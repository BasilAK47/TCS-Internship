import os
import pathlib

vessels=[]
cup=[]
disk=[]

for filepath in pathlib.Path('data/vessels').glob('**/*'):
    vessels.append(str(filepath.absolute()))

c=0
for filepath in pathlib.Path('data/disk_cup').glob('**/*'):
    if c%2==0:
        cup.append(str(filepath.absolute()))
    else:    
        disk.append(str(filepath.absolute()))
    c+=1

# for i in vessels:
#     if i not in cd:
#         print('==')
#         print(i)
#     else:
#         pass  
print(len(vessels))
print(len(cup))
print(len(disk))

import cv2
import numpy as np
for i in range(50):
    v=vessels[i]
    c=cup[i]
    d=disk[i]

    v=cv2.imread(v,0)
    v=cv2.resize(v,(150,150))
    ret,v = cv2.threshold(v,80,255,cv2.THRESH_BINARY)
    v=cv2.cvtColor(v,cv2.COLOR_GRAY2RGB)
    v[np.all(v == [255, 255, 255], axis=-1)] = [0,0,255]


    d=cv2.imread(d)
    d=cv2.resize(d,(150,150))
    d[np.all(d == [255, 255, 255], axis=-1)] = [0,255,0]

    print(v.shape)
    c=cv2.imread(c)
    c=cv2.resize(c,(150,150))
    c[np.all(c == [255, 255, 255], axis=-1)] = [255,0,0]
    
    print(c.shape)
    img1=cv2.add(v,c)
    img=cv2.add(img1,d)
    cv2.imwrite('data/fused/%s.png'%(str(i)), img) 
    print(i)
