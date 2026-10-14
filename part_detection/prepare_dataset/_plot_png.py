# import matplotlib.image as img 
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

im = np.array(Image.open("images/IMG_20201123_144800.png" ), dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)
txt_set = [0.12862401335761992, 0.4016194331983805, 0.08421052631578947, 0.026108075288403153]
left = int(txt_set[0]*1000 - (txt_set[3]/2)*1000)
top = int(txt_set[1]*750 - (txt_set[2]/2)*750)
print((txt_set[3]/2)*750)
# left = int(txt_set[0]*1000 )
# top = int(txt_set[1]*750)
print(top, left)
width = int(txt_set[2]*1000)
height = int(txt_set[3]*750)
# Create a Rectangle patch  왼쪽 오른쪽 넓이 높이
rect = patches.Rectangle((left,top),height,width,linewidth=1,edgecolor='r',facecolor='none')

# rect = patches.Rectangle((115,269),26,63,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()




# fig, ax = plt.subplots()

# ax.plot([1,4],[1,4])

# ax.add_patch(
#      patches.Rectangle(
#         (1, 1),
#         0.5,
#         0.5,
#         edgecolor = 'blue',
#         facecolor = 'red',
#         fill=True
#      ) )

# plt.show()

# fileName = "images/IMG_20201123_144800.png" 
# ndarray = img.imread(fileName) 
# plt.imshow(ndarray) 
# plt.show()

