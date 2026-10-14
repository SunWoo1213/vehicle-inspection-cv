# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, trainset_reload, valset_reload, testset_reload
from utils.dist_util import get_world_size

import pprint as pp
import pandas as pd

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    if args.fp16:
        checkpoint = {
            'model': model_to_save.state_dict(),
            'amp': amp.state_dict()
        }
    else:
        checkpoint = {
            'model': model_to_save.state_dict(),
        }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    elif args.dataset == "FILE_LOAD":
        num_classes = 108

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)

    model.load_from(np.load(args.pretrained_dir))
    if args.pretrained_model is not None:
        print(args.pretrained_model)
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def f1_score_list(cf):
#     print( 'Class_ID', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score')
    report = []
    for yt in range(cf.shape[0]):
        class_id = yt    
        xp = yt
        TP = cf[yt][xp]
        FP = 0
        for i in range(cf.shape[0]):
            if i != yt:
                FP += cf[i][xp]
        FN = 0
        for i in range(cf.shape[1]):
            if i != xp:
                FN += cf[yt][i]

        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)

        f1_score = (2*Precision*Recall)/(Precision+Recall)

#         print(class_id, TP, FP, FN, Precision, Recall, f1_score)
        report.append([class_id, TP, FP, FN, Precision, Recall, f1_score])
    
    report_ds=pd.DataFrame(report, columns=['Class_ID', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score'])
    report_ds.to_csv('./report.csv', index=False)

def valid(args, model, writer, test_loader, testset, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Test *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    print(test_loader)

    model.eval()
    all_preds, all_label = [], []
    res_dic = {'imgname':[], 'label':[], 'pred':[]}
    while testset.getChunkPosTest() < testset.getChunkTotalTest():
        epoch_iterator = tqdm(test_loader,
                              desc="Testing... (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        loss_fct = torch.nn.CrossEntropyLoss()        
        for step, batch in enumerate(epoch_iterator):
            imgname=batch[2]
            batch = tuple(t.to(args.device) for t in batch[:2])
            x, y = batch
            with torch.no_grad():
                logits = model(x)

                eval_loss = loss_fct(logits, y)
                eval_loss = eval_loss.mean()
                eval_losses.update(eval_loss.item())

                preds = torch.argmax(logits, dim=-1)
                #print(len(imgname))
                #print(y.shape, preds.shape)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                preds2 = preds.detach().cpu().numpy()
                label2 = y.detach().cpu().numpy()
                # print(preds2.shape, label2.shape)
                # print(preds2)
                # print(label2)
                for i, e in enumerate(imgname):
                    res_dic['imgname'].append(imgname[i])
                    res_dic['label'].append(label2[i])
                    res_dic['pred'].append(preds2[i])

                all_preds[0] = np.append(
                    all_preds[0], preds2, axis=0
                )
                all_label[0] = np.append(
                    all_label[0], label2, axis=0
                )

            epoch_iterator.set_description("Testing... (loss=%2.5f)" % eval_losses.val)
        testset.incChunkPosTest()
        if testset.getChunkPosTest() == testset.getChunkTotalTest(): break
        else: test_loader = testset_reload(testset, args)    
    testset.setChunkPosZeroTest()
    test_loader = testset_reload(testset, args)
    
    df = pd.DataFrame.from_dict(res_dic)            
    df.to_csv('res_dic.csv', index=False)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)
    val_accuracy = accuracy    
    #dist.barrier()
    #val_accuracy = reduce_mean(accuracy, args.nprocs)
    #val_accuracy = val_accuracy.detach().cpu().numpy()
    
    pp.pprint(all_preds)
    pp.pprint(all_label)
    cf = confusion_matrix(all_label, all_preds)
    f1_score_list(cf)

    fig = plt.figure()
    plt.matshow(cf)
    plt.title("Confusion matrix of TransFG")
    plt.colorbar()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig('confusion_matrix.jpg')


    logger.info("\n")
    logger.info("Test Results")
    logger.info("Global Steps: %d" % global_step)    
    logger.info("Test Accuracy: %2.5f" % val_accuracy)
    if args.local_rank in [-1, 0]:
        writer.add_scalar("test/accuracy", scalar_value=val_accuracy, global_step=global_step)

    return val_accuracy

def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, val_loader, test_loader, trainset, valset, testset = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    start_time = time.time()
    
    valid(args, model, writer, test_loader, testset, global_step)
# =============================================================================
#     while True:
#         model.train()
#         epoch_iterator = tqdm(train_loader,
#                               desc="Training (X / X Steps) (loss=X.X)",
#                               bar_format="{l_bar}{r_bar}",
#                               dynamic_ncols=True,
#                               disable=args.local_rank not in [-1, 0])
#         all_preds, all_label = [], []
#         for step, batch in enumerate(epoch_iterator):
#             batch = tuple(t.to(args.device) for t in batch)
#             x, y = batch
# 
#             loss, logits = model(x, y)
#             loss = loss.mean()
# 
#             preds = torch.argmax(logits, dim=-1)
# 
#             if len(all_preds) == 0:
#                 all_preds.append(preds.detach().cpu().numpy())
#                 all_label.append(y.detach().cpu().numpy())
#             else:
#                 all_preds[0] = np.append(
#                     all_preds[0], preds.detach().cpu().numpy(), axis=0
#                 )
#                 all_label[0] = np.append(
#                     all_label[0], y.detach().cpu().numpy(), axis=0
#                 )
# 
#             if args.gradient_accumulation_steps > 1:
#                 loss = loss / args.gradient_accumulation_steps
#             if args.fp16:
#                 with amp.scale_loss(loss, optimizer) as scaled_loss:
#                     scaled_loss.backward()
#             else:
#                 loss.backward()
# 
#             if (step + 1) % args.gradient_accumulation_steps == 0:
#                 losses.update(loss.item()*args.gradient_accumulation_steps)
#                 if args.fp16:
#                     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
#                 else:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#                 scheduler.step()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 global_step += 1
# 
#                 epoch_iterator.set_description(
#                     "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
#                 )
#                 if args.local_rank in [-1, 0]:
#                     writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
#                     writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
#                 if global_step % args.eval_every == 0:
#                     with torch.no_grad():
#                         accuracy = valid(args, model, writer, test_loader, global_step)
#                     if args.local_rank in [-1, 0]:
#                         if best_acc < accuracy:
#                             save_model(args, model)
#                             best_acc = accuracy
#                         logger.info("best accuracy so far: %f" % best_acc)
#                     model.train()
# 
#                 if global_step % t_total == 0:
#                     break
#         all_preds, all_label = all_preds[0], all_label[0]
#         accuracy = simple_accuracy(all_preds, all_label)
#         accuracy = torch.tensor(accuracy).to(args.device)
#         #dist.barrier()
#         #train_accuracy = reduce_mean(accuracy, args.nprocs)
#         #train_accuracy = train_accuracy.detach().cpu().numpy()
#         logger.info("train accuracy so far: %f" % accuracy)
#         losses.reset()
#         if global_step % t_total == 0:
#             break
# 
#     writer.close()
#     logger.info("Best Accuracy: \t%f" % best_acc)
#     logger.info("End Training!")
#     end_time = time.time()
#     logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))
# =============================================================================

def main_test(args):
    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    # args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    # Training
    train(args, model)