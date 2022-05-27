from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
 
# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
 
# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()
 
# load dataset
#[X1, X2] = load_real_samples('fundus_150.npz')

import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
def predict(path):
    s_img = load_img(src[i], target_size=size)
        # convert to numpy array
    s_img = img_to_array(s_img)
    img=(img - 127.5) / 127.5

    #print('Loaded', img.shape, X2.shape)
    # load model
    model = load_model('model1.h5')
    # # select random example
    # ix = randint(0, len(X1), 1)
    src_image = img.reshape(1,256,256,3)

    # generate image from source
    gen_image = model.predict(src_image)[0]
    gen_image=(gen_image*127.5)+127.5
	
    cv2.imwrite('result.png',gen_image)
    # pyplot.imshow(gen_image)
    # pyplot.show()

#predict(1)

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
# # plot all three images
# plot_images(src_image, gen_image, tar_image)
def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels
def delayed_insert(label,index,message):
    label.insert(index,message)  
def predict1(path,lb1):

    lb1.after(0,delayed_insert,lb1,0,'Loading image')

    src_image = load_image(path)
    lb1.after(100,delayed_insert,lb1,1,'Preprocess the image')
    # load model
    model = load_model('model1.h5')
    lb1.after(200,delayed_insert,lb1,2,'Generating fundus image')
    # generate image from source
    gen_image = model.predict(src_image)
    lb1.after(300,delayed_insert,lb1,3,'Cleaning fundus image')
    gen_image = (gen_image + 1) / 2.0
    # plot the image
    #cv2.imwrite('result.png',gen_image[0]*255)
    pyplot.imshow(gen_image[0])
    pyplot.axis('off')
    pyplot.savefig('result.png',bbox_inches='tight')
    # pyplot.axis('off')
    # pyplot.show()
import pathlib
def predict2(path,lb1):
    paths1=[]
    paths2=[]
    for filepath in pathlib.Path('data/Images').glob('**/*'):
        # if c>3:
        #     breaks
        paths1.append(str(filepath.absolute()))
        # print(str(filepath.absolute()))
        
    for filepath in pathlib.Path('data/fused').glob('**/*'):
        # if c>3:
        #     break
        paths2.append(str(filepath.absolute()))
        # print(str(filepath.absolute()))
       
    paths1=sorted(paths1)
    #paths2=sorted(paths2)
    ind=paths1.index(path.replace('/','\\'))
    print(path)
    path=paths2[ind]
    print(path)
    

    

    
    lb1.after(0,delayed_insert,lb1,0,'Loading image')

    src_image = load_image(path)
    lb1.after(100,delayed_insert,lb1,1,'Preprocess the image')
    # load model
    model = load_model('model1.h5')
    lb1.after(200,delayed_insert,lb1,2,'Generating fundus image')
    # generate image from source
    gen_image = model.predict(src_image)
    lb1.after(300,delayed_insert,lb1,3,'Cleaning fundus image')
    gen_image = (gen_image + 1) / 2.0
    # plot the image
    #cv2.imwrite('result.png',gen_image[0]*255)
    pyplot.imshow(gen_image[0])
    pyplot.axis('off')
    pyplot.savefig('result.png',bbox_inches='tight')
    # pyplot.axis('off')
    # pyplot.show()

#predict1('data/fused/4.png')



	