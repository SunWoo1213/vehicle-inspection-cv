import cv2
import math
import os
import pandas as pd

def print_class(cls):
    members = [attr for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith("__")]
    for e in members:
      print(e + ' <<<<')
      print(getattr(cls, e))
      
def Hello():
    print('Hello nice to meet you')

def rotate(x,y,angle,cx=0,cy=0):    
    x = x-cx
    y = y-cy
    rad = math.pi/180*angle
    x1 = math.cos(rad)*x
    y1 = math.sin(rad)*x
    x2 = -math.sin(rad)*y
    y2 = math.cos(rad)*y
    #print(x1,y1,x2,y2)
    return cx+x1+x2, cy+y1+y2  

def rxc(r):
  return (r[0]+r[2])/2
def ryc(r):
  return (r[1]+r[3])/2
def rw(r):
  return r[2]-r[0]
def rh(r):
  return r[3]-r[1]

def draw_arrow(img, a, rect):
  cx = rxc(rect)
  cy = ryc(rect)
  if rw(rect) > rh(rect):
    L=(rect[3]-rect[1])/2
  else:
    L=(rect[2]-rect[0])/2
  al=5
  sx, sy = rotate(0,-L,a)
  ex, ey = rotate(0,L,a)
  lx, ly = rotate(-al,L-al,a)
  rx, ry = rotate(al,L-al,a)
  sy*=-1;ey*=-1;ly*=-1;ry*=-1
  sx+=cx;sy+=cy;ex+=cx;ey+=cy  
  lx+=cx;ly+=cy;rx+=cx;ry+=cy

  cv2.line(img, (int(sx),int(sy)), (int(ex),int(ey)), (255,0,0), 2)  
  cv2.line(img, (int(lx),int(ly)), (int(ex),int(ey)), (255,0,0), 2)
  cv2.line(img, (int(rx),int(ry)), (int(ex),int(ey)), (255,0,0), 2)

def get_leaf_dirs(path):
  ssL=[]
  for dirname, dirnames, filenames in os.walk(path):    
    if dirnames == []:
      ssL.append({'path':dirname})
  return ssL

def read_csv_as_pd(path):
  ds = pd.read_csv(path)
  return ds