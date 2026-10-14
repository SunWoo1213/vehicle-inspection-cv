import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


def load_img_name(label_list):
    """
    load imgs txt file return img list

    example 
    load_img_name(["partitions\X_devel.txt","partitions\X_train.txt"]) 
    """
    img_list = []
    for read_file in label_list:
        with open(read_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            img_list.append(line.split(' ')[0])
    return img_list

def ctrbc2cxywh(label_data, origin_x, origin_y, opt):
    '''
    input: label_data 딕셔너리 형태로 class_n, left, top, right, bottom이 들어옴
    origin_x, origin_y 는 with, height의 픽셀 갯수 (넓이 높이)

    output 0~1 사이의 값으로 x_center, y_center, width, height 가 나옴
    '''
    # classId = label_data['classId'].split('.')[0].split('P')[1]
    classId = label_data['classId'].split('.')[0]
    left    = label_data['coords']['tl']['x']/origin_x
    top     = label_data['coords']['tl']['y']/origin_y
    right   = label_data['coords']['br']['x']/origin_x
    bottom  = label_data['coords']['br']['y']/origin_y

    width       = right - left
    height      = bottom - top   
    x_center    = left + width/2
    y_center    = top + height/2
    return [classId, x_center, y_center, width, height]


def bbox_img(img_dir, bbox_list, copy_data_dir, class_name):
    red_color = (0,0,255)
    color_map = create_ade20k_label_colormap()

    img = cv2.imread(img_dir)
    if img is None:
        img_array = np.fromfile(img_dir, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    for bbox in bbox_list:
        label_name  = bbox[0]
        x_center    = bbox[1] * w
        y_center    = bbox[2] * h
        width       = bbox[3] * w
        height      = bbox[4] * h
        bbox = Center2LeftTop([x_center, y_center, width, height])
        rec_color = list(map(int,color_map[int(label_name[1:])]))
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), rec_color, 2)
        label_y = bbox[1]-10
        if label_y< 30:
            label_y = bbox[1] + 30
        cv2.putText(img, class_name[int(label_name[1:])], (bbox[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,100), 2)
    
    img_name = os.path.basename(img_dir)
    # imshow
    # cv2.imshow(img_name, img)
    
    # save img
    save_bbox_img_path = os.path.join(copy_data_dir, img_name)
    cv2.imwrite(save_bbox_img_path, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def Center2LeftTop(bbox):
    """
    input : x_center, y_center, width, height

    output :left, top, right, bottom
    """
    x_center    = bbox[0]
    y_center    = bbox[1]
    half_width       = bbox[2]/2
    half_height      = bbox[3]/2

    left    = x_center - half_width
    right   = x_center + half_width
    top     = y_center - half_height
    bottom  = y_center + half_height
    return int(left), int(top), int(right), int(bottom)

def write_label_txt(bbox_list, label_dir, label_name):
    # print("write_label_txt")
    # print(label_dir, label_name)
    txt_name = label_name + '.txt'
    txt_dir = os.path.join(label_dir, txt_name)
    bbox_num = len(bbox_list)
    with open(txt_dir, 'w') as f:
        for idx, bbox in enumerate(bbox_list):
            class_id    = bbox[0]
            x_center    = bbox[1]
            y_center    = bbox[2]
            width       = bbox[3]
            height      = bbox[4]
            _str = str(int(class_id[1:])) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height)
            f.write(_str)
            if idx < bbox_num-1:
                f.write('\n')

def make_folder_width_del(folder_dir):
    try:
        print(f"make folder {folder_dir}")
        if os.path.exists(folder_dir):
            shutil.rmtree(folder_dir)
        os.makedirs(folder_dir)    
    except OSError:
        print('Error: Creating directory. ' + folder_dir)

def copy_img_folder(img_list, copy_data_dir):
    """
    img path list를 받아오고 이 리스트를 copy_data_dir로 복사
    """
    make_folder_width_del(copy_data_dir)
    for img_path in tqdm(img_list):
        source =  os.path.basename(img_path)
        destination = os.path.join(copy_data_dir, source)
        shutil.copyfile(img_path, destination)

def check_class_names(img_list, opt):
    """
    img list를 받아서 받은 이미지의 json 파일명으로 고친후 json 파일을 읽어서 각 bbox의 class name을 확인한다.
    """
    class_list = set()
    if opt.data_set_name   == 'Car_Part':
        class_name = 'labeling'
    elif opt.data_set_name == 'Car_Part_V2':
        class_name = 'classId'
    else:
        print(f"error{opt.data_set_name}")
        raise


    for img_dir in tqdm(img_list):
        # 이미지의 bbox list를 로드
        bbox_list = load_label_data_from_json(img_dir, opt)

        # 각 bbox의 class만 로드
        for bbox_data in bbox_list:
            classId = bbox_data[class_name]
            class_list.add(classId)
    class_list = list(class_list)
    class_list.sort()

    class_dict = dict()
    for img_dir in tqdm(img_list):
        # 이미지의 bbox list를 로드
        bbox_list = load_label_data_from_json(img_dir, opt)

        # 각 bbox의 class만 로드
        for bbox_data in bbox_list:
            classId = bbox_data[class_name]
            if classId in class_dict:
                class_dict[classId] = class_dict[classId] +1
            else:
                class_dict[classId] = 1
    


    return class_list, class_dict
        



def label2txt(img_list, copy_data_dir, opt):
    """
    이미지 path list를 로드한후 같은 폴더에 들어 있는 json 파일을 읽어
    yolo 데이터에 맞게 변경한후 그 데이터를 txt 파일로 copy_data_dir 위치에 저장을 한다.
    """
    # txt 파일을 저장할 폴더 생성
    make_folder_width_del(copy_data_dir)
    for img_path in tqdm(img_list):
        bbox_list = load_bbox_from_label(img_path, opt)
        write_label_txt(bbox_list, copy_data_dir, os.path.basename(img_path).split('.')[0])




def save_bbox_img(img_list, copy_data_dir, class_name, opt):
    """
    이미지 path list를 로드한후 같은 폴더에 들어 있는 json 파일을 읽어
    yolo 데이터에 맞게 변경한후 그 데이터를 txt 파일로 copy_data_dir 위치에 저장을 한다.
    """
    # txt 파일을 저장할 폴더 생성
    make_folder_width_del(copy_data_dir)
    for img_path in tqdm(img_list):
        bbox_list = load_bbox_from_label(img_path, opt)
        bbox_img(img_path, bbox_list, copy_data_dir, class_name)

def load_label_data_from_json(img_path, opt):
    """
    라벨 데이터를 로드후 반환 하는 함수
    data_set_name을 이용하여 어떤 데이터셋의 json인지 구분하여 bbox를 반환한다.

    같은 data의 이미지와 json 파일이 같은 폴더안에 있는경 (확장자만 다름)

    같은 data의 이미지와 json 파일이 다른 main 폴더가 다른 경우 (/data/json/car1/tmp1/tmp.json, /data/img/car1/tmp1/tmp.jpg)

    """
    # 같은 data의 이미지와 json 파일이 다른 main 폴더가 다른 경우 (/data/json/car1/tmp1/tmp.json, /data/img/car1/tmp1/tmp.jpg)
    if opt.img_file_path != opt.json_file_path:
        img_path = opt.json_file_path + img_path.split(opt.img_file_path)[1]

    json_dir = img_path.split('.')[0] + '.json'
    
    
    with open(json_dir, "r", encoding='UTF8') as test_json:
        json_dataset = json.load(test_json)

    if opt.data_set_name == 'Car_Part':
        bbox_dataset = json_dataset['Learning data Info']
        return bbox_dataset 
    if opt.data_set_name == 'Car_Part_V2':
        bbox_dataset = json_dataset['learningDataInfo']['objects']
        return bbox_dataset  



def load_bbox_from_label(img_path, opt):
    """
    json 파일을 로드하고 bbox를 return 한다 (yolo)
    """
    # # 이미지 파일 path를 이용하여 json 파일 path 를 생성
    # json_dir = img_path.split('.')[0] + '.json'
    # with open(json_dir, "r", encoding='UTF8') as test_json:
    #     json_dataset = json.load(test_json)
    
    # # 이미지 cv2로 로드 cv2.imread가 utf8로 인하여 안될경우 if 문을 실행해서 로드
    # img = cv2.imread(img_path)
    # if img == None:
    #     img_array = np.fromfile(img_path, np.uint8)
    #     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # h,w,c = img.shape
    
    # bbox_dataset = json_dataset['Learning data Info.']
    # 이미지 cv2로 로드 cv2.imread가 utf8로 인하여 안될경우 if 문을 실행해서 로드
    
    img = cv2.imread(img_path)
    if img is None:
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    h,w,c = img.shape
    
    bbox_dataset = load_label_data_from_json(img_path, opt)
    
    bbox_list = []
    for bbox_data in bbox_dataset:
        bbox = ctrbc2cxywh(bbox_data, w, h, opt)
        bbox_list.append(bbox)
    return bbox_list

        



def search(dirname, path=True):
        """
        하위 폴더 검색
        """
        filenames = os.listdir(dirname)
        full_filename = []
        for filename in filenames:
            full_filename.append(os.path.join(dirname, filename))
        if path == True:
            # include path
            return full_filename
        else:
            # no Path
            return filenames

def sub_search(dirname, file_list = None, data_type = None):
    if file_list == None:
        file_list = []

    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                file_list = sub_search(full_filename, file_list = file_list, data_type = data_type)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == data_type: 
                    file_list.append(full_filename)
                    # print(full_filename)
        return file_list
    except PermissionError:
        pass


def split_dataset(img_list, dataset_ratio):
    """
    이미지 리스트 와 train, valid, test 비율을 받으면 img_list를 비율대로 train valid test list로 나누어 준다.
    
    """
    train_ratio = dataset_ratio[0]
    valid_ration = dataset_ratio[1]
    test_ration = dataset_ratio[2]

    img_cnt = len(img_list)
    train_cnt = int(img_cnt * train_ratio)
    img_cnt -= train_cnt
    valid_cnt = int(img_cnt * valid_ration/(valid_ration + test_ration))

    train_list = img_list[:train_cnt]
    valid_list = img_list[train_cnt:(train_cnt + valid_cnt)]
    test_list  = img_list[(train_cnt + valid_cnt):]
    print(f"데이터 숫자 \ntrain : {len(train_list)}\nvalid : {len(valid_list)}\ntest : {len(test_list)} ")
    return train_list, valid_list, test_list




def make_graph(all_dict_list, opt):
    """
    클래스별 데이터 셋을 받은 후 (딕셔너리 타입)
    막대그래프로 각 데이터셋의 갯수를 표시하고 퍼센트도 표시한다.
    """
    # 딕셔너리 sort
    def f1(x):
        x = x[0].split('.')[0].split('P')[1]
        return int(x)
    def f2(x):
        return int(x[1])
    # 리스트 key val 분류
    def dict2_keyval(data_dict):
        key = []
        val = []
        for data in data_dict:
            key.append(data[0])
            val.append(data[1])
        return key, val

    # 한글 폰트 사용
    from matplotlib import font_manager, rc
    font_path = "/usr/share/fonts/NanumGothicCoding.ttf"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
    
    # plt.rc('font', family=font_path)
    # print(plt.rcParams['font.family'])
    


    import matplotlib as mpl
    print(mpl.matplotlib_fname())

    val_list =[]
    for class_dict in all_dict_list:
        # key 값 sort
        class_list_key = sorted(class_dict.items(), key=f1)
        # value 값 sort
        class_list_value = sorted(class_dict.items(), key=f2)
        print(class_list_key)
        print(class_list_value)
        key, val = dict2_keyval(class_list_key)
        val_list.append(val)
    
    x = np.arange(len(key))
    color_list = ['r', 'b', 'g']
    bar_label = ['train', 'val', 'test']
    i=1
    for i in range(3):
        x_bias = 0.25 * i
        plt.bar(x +x_bias, val_list[i], color=color_list[i], width=0.2, label=bar_label[i])
        for idx, v in enumerate(val_list[i]):
            plt.text(idx + x_bias, v, v, fontsize=9, color='blue', horizontalalignment='center')
    plt.xticks(x, key, rotation=45)

    # bar label 표시
    plt.legend()
    plt.gcf().subplots_adjust(bottom=0.25)
    if opt.remote_server:
        plt.savefig('class count.png')
    else:
        plt.show()
    





def create_ade20k_label_colormap():
  """Creates a label colormap used in ADE20K segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])
