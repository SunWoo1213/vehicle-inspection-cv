import utils.util as util
import os
from tqdm import tqdm
import random
import argparse
import shutil


#class Name
class_name = ['Car', 'Front bumper', 'Rear bumper', 'Wheel', 'A Pillar', 'C Pillar', 'Side mirror', 'Front door', 'Back door', 'Radiator grill', 'Head lamp', 'Rear lamp', 'Bonnet', 'Trunk', 'Roof']



def json2yoloText():
    pass

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file_path',    type=str,  default='/workspace/data/091_차량외관_영상_데이터/06_품질검증_Dataset/1_원천데이터',    help='initial jpg path')
    parser.add_argument('--json_file_path',   type=str,  default='/workspace/data/091_차량외관_영상_데이터/06_품질검증_Dataset/2_라벨링데이터',  help='initial json path')
    parser.add_argument('--output_yolo_data', type=str,  default='/workspace/data/tmp_data',                                          help='initial json path')
    parser.add_argument('--data_ratio',       type=list, default=[0.8, 0.1, 0.1],                                                     help='train_list, valid_list, test_list')
    parser.add_argument('--data_set_name',    type=str,  default='Car_Part_V2',                                                       help='train_list, valid_list, test_list')
    parser.add_argument('--test_code',        type=bool, default=False,                                                               help='if test_code True make list small to test')
    parser.add_argument('--remote_server',    type=bool, default=True,                                                                help='if remote server save png plt.show() --> plt.savefig()')    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt    


def main(opt):
    # 하위 폴더의 모든 image path
    img_list = util.sub_search(opt.img_file_path, data_type='.jpg')
    random.shuffle(img_list)
    if opt.test_code:
        img_list = img_list[:10000]

    # 이미지 path 리스트를 train valid test 분포에 맞게 자르기
    # data_ratio = [0.8, 0.1, 0.1]
    train_list, valid_list, test_list = util.split_dataset(img_list, opt.data_ratio)

    # 이미지 복사 images//train, images//valid, images//test
    print("img train valid test 분류 image copy images//train, images//valid, images//test")
    util.copy_img_folder(train_list, os.path.join(opt.output_yolo_data, 'images', 'train'))
    util.copy_img_folder(valid_list, os.path.join(opt.output_yolo_data, 'images', 'valid'))
    util.copy_img_folder(test_list, os.path.join(opt.output_yolo_data, 'images', 'test'))

    # yolo jsonlabe 을 txt 라벨로 변환
    print("json label to txt file")
    util.label2txt(train_list, os.path.join(opt.output_yolo_data, 'labels', 'train'), opt)
    util.label2txt(valid_list, os.path.join(opt.output_yolo_data, 'labels', 'valid'), opt)
    util.label2txt(test_list, os.path.join(opt.output_yolo_data, 'labels', 'test'), opt)

    # bbox 포함한 이미지 저장
    # print("save ground truth bbox data img")
    # util.save_bbox_img(train_list, os.path.join(output_yolo_data, 'bbox_img', 'train'), class_name, opt)
    # util.save_bbox_img(valid_list, os.path.join(output_yolo_data, 'bbox_img', 'valid'), class_name, opt)
    # util.save_bbox_img(test_list,  os.path.join(output_yolo_data,  'bbox_img', 'test'), class_name, opt)



if __name__ == "__main__":
    random.seed(0)
    opt = parse_opt(True)
    main(opt)
