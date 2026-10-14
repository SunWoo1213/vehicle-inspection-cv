import utils.util as util
import os
from tqdm import tqdm
import random
import argparse

import shutil
def main(opt):
    # 하위 폴더의 모든 image path 가져오기
    img_list = util.sub_search(opt.img_file_path, data_type='.jpg')
    random.shuffle(img_list)

    # 이미지 path 리스트를 train valid test 분포에 맞게 자르기

    if opt.test_code:
        limit_list = 300
        print("~~~~~~~~~      test code     ~~~~~~~~~")
        print(f"limit img list to {limit_list}")
        img_list = img_list[:limit_list]

    train_list, valid_list, test_list = util.split_dataset(img_list, opt.data_ratio)

    all_dict_list =[]
    all_dict_list.append(util.check_class_names(train_list, opt)[1])
    all_dict_list.append(util.check_class_names(valid_list, opt)[1])
    all_dict_list.append(util.check_class_names(test_list,  opt)[1])

    util.make_graph(all_dict_list, opt)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file_path',  type=str,  default='data/091_차량외관_영상_데이터/06_품질검증_Dataset/1_원천데이터',    help='initial jpg path')
    parser.add_argument('--json_file_path', type=str,  default='data/091_차량외관_영상_데이터/06_품질검증_Dataset/2_라벨링데이터',  help='initial json path')
    parser.add_argument('--data_ratio',     type=list, default=[0.8, 0.1, 0.1],                                                  help='train_list, valid_list, test_list')
    parser.add_argument('--data_set_name',  type=str,  default='Car_Part_V2',                                                    help='train_list, valid_list, test_list')
    parser.add_argument('--test_code',      type=bool, default=False,                                                             help='if test_code True make list small to test')
    parser.add_argument('--remote_server',  type=bool, default=True,                                                             help='if remote server save png plt.show() --> plt.savefig()')    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt    


if __name__ == "__main__":
    opt = parse_opt(True)

    random.seed(0)
    main(opt)