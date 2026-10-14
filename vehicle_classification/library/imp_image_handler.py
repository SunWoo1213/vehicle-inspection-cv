import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_images(images, w, h, row, col):
    fig = plt.figure(figsize=(w,h))
    for i in range(1, row*col+1):
        fig.add_subplot(row,col,i)
        plt.imshow(images[i-1])
    plt.show()

def show_image(image):
    plt.imshow(image)
    plt.show()

def image_jpeg_read(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    
def image_jpeg_write(file_path, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, img)
    
def grab_foreground(img, threshold, cut):
    img_int = img.astype(int)
    
    img_sum = img_int[:,:,0] + img_int[:,:,1] + img_int[:,:,2]
    if cut == "white":
        img_filter = (img_sum < threshold).astype(int)
    elif cut == "black":
        img_filter = (img_sum > threshold).astype(int)

    cz = np.zeros(img.shape)
    cz[:,:,0]=img[:,:,0]*img_filter
    cz[:,:,1]=img[:,:,1]*img_filter
    cz[:,:,2]=img[:,:,2]*img_filter
    cz=cz.astype(np.uint8)
    #imshow.show_image(i4)

    ys, ye, xs, xe = -1, -1, -1, -1
    for y in range(cz.shape[0]):
        if np.sum(cz[y,:,:]) > 0:
            #print("ystart", y)
            ys = y
            break

    for y in reversed(range(cz.shape[0])):
        if np.sum(cz[y,:,:]) > 0:
            #print("yend", y)
            ye = y
            break

    for x in range(cz.shape[1]):
        if np.sum(cz[:,x,:]) > 0:
            #print("xstart", x)
            xs = x
            break

    for x in reversed(range(cz.shape[1])):
        if np.sum(cz[:,x,:]) > 0:
            #print("xend", x)
            xe = x
            break

    img_fore = cz[ys:ye+1,xs:xe+1,:]
    return img_fore

def stretch_image(img, width, height):
    stretched = cv2.resize(img,(width,height),interpolation=cv2.INTER_AREA)
    #print(img.shape,stretched.shape)
    return stretched
    
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def overlap(image1, image2, top, left):
  for y in range(image2.shape[0]):
    for x in range(image2.shape[1]):
      if np.sum(image2[y,x,:]) != 0:
        image1[top+y,left+x,:]=image2[y,x,:]
            
def randomly_locate_on_canvas(canvas, img, specs, label, angle, overlap_threshold=0.1):
    def iou(b1, b2):
      area_i = intersection(b1, b2)
      area_u = union(b1, b2, area_i)
      return area_i/area_u
    
    def union(b1, b2, area_intersection):
      area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
      area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
      return area1+area2-area_intersection
    
    def intersection(b1, b2):
      x = max(b1[0], b2[0])
      y = max(b1[1], b2[1])
      w = min(b1[2], b2[2]) - x
      h = min(b1[3], b2[3]) - y
      if w < 0 or h < 0:
        return 0
      return w*h
    
    def calc_max_iou(box_specs, box):
      max_iou = 0
      for e in box_specs:
        viou = iou(e['location'],box)
        if viou > max_iou:
          max_iou = viou
      return max_iou
      
    range_horz = canvas.shape[0] - img.shape[1]
    range_vert = canvas.shape[0] - img.shape[0]
    left = np.random.randint(0,range_horz)
    top = np.random.randint(0,range_vert)
    right = left+img.shape[1]
    bottom = top+img.shape[0]
    #canvas[top:bottom,left:right,:] = e['image']
    box = (left,top,right,bottom)
    if calc_max_iou(specs['box_specs'],box) < overlap_threshold:
      print('inserted')
      overlap(canvas, img, top, left)
      specs['box_specs'].append({'location':box, 
                    'label': label, 'angle':angle})
    else:
      print('not inserted')