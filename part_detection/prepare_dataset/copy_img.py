import utils.util as util
import os
import glob
import json
import cv2
import shutil
from tqdm import tqdm
import random

import shutil


def main():
    img_list = util.sub_search('Parts bbox', data_type='.jpg')
    random.shuffle(img_list)
    data_ratio = [0.8, 0.1, 0.1]
    train_list, valid_list, test_list = util.split_dataset(img_list, data_ratio)

    # 이미지 복사 images//train, images//valid, images//test
    util.copy_img_folder(train_list, os.path.join('images', 'train'))
    util.copy_img_folder(train_list, os.path.join('images', 'valid'))
    util.copy_img_folder(train_list, os.path.join('images', 'test'))



if __name__ == "__main__":
    random.seed(0)
    main()