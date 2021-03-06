from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import pathlib

def load_images(size=(256,256)):
    src_list, tar_list = [],[]
    src=[]
    tar=[]
    c=0
    for filepath in pathlib.Path('data/Images').glob('**/*'):
        # if c>3:
        #     break
        tar.append(str(filepath.absolute()))
        # print(str(filepath.absolute()))
        c+=1
    c=0    
    for filepath in pathlib.Path('data/fused').glob('**/*'):
        # if c>3:
        #     break
        # print(str(filepath.absolute()))
        src.append(str(filepath.absolute())) 
        c+=1   
    tar=sorted(tar) 
    c=0   
    for i in range(50):
        # if c>3:
        #     break
        # load and resize the image
        s_img = load_img(src[i], target_size=size)
        # convert to numpy array
        s_img = img_to_array(s_img)
        # load and resize the image
        t_img = load_img(tar[i], target_size=size)
        # print(tar[i])
        # convert to numpy array
        t_img = img_to_array(t_img)
        # split into satellite and map
        src_list.append(s_img)
        tar_list.append(t_img)
        c+=1
    return [asarray(src_list), asarray(tar_list)]

[s,t]=load_images()   
print(s.shape,t.shape) 

filename = 'fundus_150.npz'
savez_compressed(filename, s, t)




 



