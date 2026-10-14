from pprint import pprint
import os
import library.imp_file_directory as ifd
import json 
import traceback
import argparse
from shutil import copy

def remove_vacent_folder_and_part(dic):    
    rdic = {}
    for path in dic.keys():            
        files = []
        for file in dic[path]:
            if '_T_' in file['name']:
                files.append(file)
        if len(files) > 0:
            rdic[path]=files
    return rdic

def get_cid_bbox(filepath):
    with open(filepath) as json_file:
        label = json.load(json_file)
    Learning_data_Info = label['learningDataInfo']
    for e in Learning_data_Info['objects']:
        if e['classId'] == "P00.차량전체":                    
            cid = Learning_data_Info['LearningDataId'].split('_')[3]
            bbox = e['coords']
            return cid, bbox
    return None, None

def make_class(dic):   
    try:        
        cid_set = set()
        cls_num = {}        
        t_cnt = 0
        for key in dic.keys():    
            files = []
            for file in dic[key]:            
                filepath = f'{key}/{file["name"]}'
                cid, _ = get_cid_bbox(filepath)                  
                if cid is not None:
                    if cid not in cls_num.keys(): 
                        cls_num[cid] = 1
                    else:
                        cls_num[cid] += 1
                    cid_set.add(cid)   
                    t_cnt += 1
                    files.append(file)
                else:
                    print(filepath)
            dic[key] = files 
        print(f'total_number = {t_cnt}')
        cid_set = sorted(cid_set)    

        cls_dic = {}
        number = 1
        for e in cid_set:        
            cls_dic[e]={'id':number,'count':cls_num[e]}
            number += 1
        return cls_dic
    except Exception as e:
        print(filepath)
        print(str(traceback.format_exc()))

def make_label_bbox(dic, class_dic):
    for key in dic.keys():
        for e in dic[key]:
            filepath = f'{key}/{e["name"]}'
            cid, bbox = get_cid_bbox(filepath)            
            e['id']=class_dic[cid]['id']            
            e['bbox']=bbox
    return dic

def make_data_splitted(dic):
    for key in dic:
        total = len(dic[key]) 
        #print(total)
        count = 1
        for e in dic[key]:
            if count < total*0.8:
                e['tr_val_te'] = '0'
            elif total*0.8 <= count and count < total*0.9:
                e['tr_val_te'] = '1'
            elif total*0.9 <= count:
                e['tr_val_te'] = '2'
            count+=1
    return dic

def write_dic_to_text(path, dic, mode='class'):    
    with open(path, 'w') as f:
        if mode == 'class':
            for key in dic:
                f.write(f'{dic[key]["id"]} {key}\n')
        elif mode == 'images':
            path_testimages = path[:path.rfind('/')]+'/testimages'            
            if os.path.isdir(path_testimages) == False:
                os.mkdir(path_testimages)
            for key in dic:
                for e in dic[key]:
                    path = key.replace("2_라벨링데이터","1_원천데이터")
                    name = e["name"].replace(".json",".jpg")
                    if e['tr_val_te']=='2':                        
                        copy(path + '/' + name, path_testimages)
                        f.write(f'{e["number"]} {path_testimages}/{name}\n')                                            
                    else:
                        f.write(f'{e["number"]} {path}/{name}\n') 
                        
        elif mode == 'image_class_label_bboxes':
            for key in dic:
                for e in dic[key]:                    
                    f.write(f'{e["number"]} {e["id"]} {e["bbox"]}\n')
        elif mode == 'train_val_test_split':
            for key in dic:
                for e in dic[key]:
                    f.write(f'{e["number"]} {e["tr_val_te"]}\n')
    print(f'{mode} finished')

def main(args):
    print(args.data_path_to_convert)
    print(args.anno_path_to_save)   
    resL = ifd.search_last_leaves(args.data_path_to_convert) 
    print(f'folder number = {len(resL)}')
    dic, total = ifd.search_every_files(resL, '.json')
    print(f'whole number = {total}')

    dic = remove_vacent_folder_and_part(dic)
    
    cls_dic = make_class(dic)
    write_dic_to_text(args.anno_path_to_save+'/classes.txt', cls_dic, mode='class')
    
    label_dic = make_label_bbox(dic, cls_dic)
    write_dic_to_text(args.anno_path_to_save+'/image_class_label_bboxes.txt', label_dic, mode='image_class_label_bboxes')
    
    split_dic = make_data_splitted(label_dic)
    write_dic_to_text(args.anno_path_to_save+'/train_val_test_split.txt', split_dic, mode='train_val_test_split')
    
    write_dic_to_text(args.anno_path_to_save+'/images.txt', dic, mode='images')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path-to-convert', default=None, type=str, help="data path from project")
    parser.add_argument('--anno-path-to-save', default=None, type=str, help="annotation path for training")    
    args = parser.parse_args()
    main(args)    