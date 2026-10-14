import utils.util as util
import os
import glob
import json
import cv2
import shutil
from tqdm import tqdm
import numpy as np
import random

#class Name
class_name = ['Front bumper', 'Rear bumper', 'Wheel', 'A Pillar', 'C Pillar', 'Side mirror', 'Front door', 'Back door', 'Radiator grill', 'Head lamp', 'Rear lamp', 'Bonnet', 'Trunk', 'Roof']






def main():
    img_list = util.sub_search('Parts bbox', data_type='.jpg')
    # json_path = util.sub_search('BMW', data_type='.json')
    
    txt_label_dir = 'label_valid'
    try:
        if os.path.exists(txt_label_dir):
            shutil.rmtree(txt_label_dir)
        os.makedirs(txt_label_dir)    
    except OSError:
        print('Error: Creating directory. ' + txt_label_dir)

    for img_dir in tqdm(img_list):
        json_dir = img_dir.split('.')[0] + '.json'

        try:
            with open(json_dir, "r", encoding='UTF8') as test_json:
                json_dataset = json.load(test_json)
        except:
            print(f"no json file {json_dir}")
        
        img = cv2.imread(img_dir)
        if img == None:
            img_array = np.fromfile(img_dir, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # img = cv2.imread(img_dir)
        h,w,c = img.shape
        bbox_list = []
        bbox_dataset = json_dataset['Learning data Info.']
        
        for bbox_data in bbox_dataset:
            bbox = util.ctrbc2cxywh(bbox_data, w, h)
            bbox_list.append(bbox)
            # print(bbox)
        
        # bbox 확인용
        # util.bbox_img(img_dir, bbox_list, class_name)

        util.write_label_txt(bbox_list, txt_label_dir, os.path.basename(img_dir).split('.')[0])

if __name__ =='__main__':
    main()