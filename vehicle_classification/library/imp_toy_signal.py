import math
import numpy as np

def make_sin(tlen, nw, move=0., amp=1.0):
  div = tlen/nw
  inc = math.pi*2/div
  v = 0
  vl = []
  for _ in range(nw):
    for _ in range(int(div)):
      vl.append(math.sin(v-move)*amp)
      v+=inc
  #vl.append(math.sin(v-move)*amp)
  vl=np.array(vl)
  return vl

