import cv2
import numpy as np

def rpn_cls_to_anchor(al, anchor_box_scales, anchor_box_ratios, rpn_stride, img):
    anchor_sizes = anchor_box_scales
    anchor_ratios = anchor_box_ratios
    downscale = float(rpn_stride)
    
    box_list = []
    rows = al.shape[1]
    cols = al.shape[2]
    num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
    
    for row in range(rows):
        for col in range(cols):
            for na in range(num_anchors):
                pos = False
                neg = False
                if al[0,row,col,na] == 1 and al[0,row,col,na+9] == 1:
                    pos = True
                elif al[0,row,col,na] == 1 and al[0,row,col,na+9] == 0:
                    neg = True
                
                aidx = int(na//3)
                ridx = int(na%3)
        
                anchor_x = (anchor_sizes[aidx] * anchor_ratios[ridx][0])
                anchor_y = (anchor_sizes[aidx] * anchor_ratios[ridx][1])
                x1_anc = downscale * (col + 0.5) - anchor_x / 2
                x2_anc = downscale * (col + 0.5) + anchor_x / 2
                y1_anc = downscale * (row + 0.5) - anchor_y / 2
                y2_anc = downscale * (row + 0.5) + anchor_y / 2
                
                if pos:
                    cv2.rectangle(img, (int(x1_anc),int(y1_anc)),(int(x2_anc),int(y2_anc)), (255,0,0), 1)
                elif neg:
                    cv2.rectangle(img, (int(x1_anc),int(y1_anc)),(int(x2_anc),int(y2_anc)), (0,255,0), 1)
    return img

def make_pseudo(row,col):
    scale = 10
    sh = row*scale
    sw = col*scale
    img = np.zeros((sh,sw,3),dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            y = i*10
            x = j*10
            R = np.random.randint(256)
            G = np.random.randint(256)
            B = np.random.randint(256)
            cv2.rectangle(img,(x,y),(x+scale,y+scale),(R,G,B),-1)
    return img 