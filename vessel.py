from keras.models import model_from_json
import tensorflow as tf
tf.python.control_flow_ops = tf

json_file = open('test_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("test_best_weights.h5")

import cv2
import copy
from matplotlib import pyplot as plt
import numpy as np


def prepocess_data(the_img):
    #normalize
    #std:55.851990770984536, img_mean:83.96562255652191
    img_std = 55.851990770984536 #np.std(the_img)
    img_mean = 83.96562255652191#np.mean(the_img)
    img_normalized = (the_img - img_mean)/img_std
    img_normalized = ((img_normalized-np.min(img_normalized)) / (np.max(img_normalized)-np.min(img_normalized)))*255

    #clahe preporcess
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_processed = clahe.apply(np.array(img_normalized, dtype=np.uint8))

    #gama correction
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gama_img =  cv2.LUT(np.array(clahe_processed, dtype = np.uint8), table)

    #finally return res
    return gama_img

def predictImage(inImg, theModel):
    stepSize = 24
    windowSize = 48
    prob_threshold = 0.99

    #create zero img
    outImg = np.zeros((inImg.shape[0], inImg.shape[1]), dtype=inImg.dtype)#outImg with zeros
    print(inImg.shape)
    for x in range(48, inImg.shape[0]-windowSize, stepSize):
        for y in range(48, inImg.shape[1]-windowSize, stepSize):
            #sub image rectangle
            subImg =copy.copy(inImg[x-windowSize//2:x+windowSize//2, y-windowSize//2:y+windowSize//2])
            subImg = prepocess_data(subImg)
            cnnShape = np.reshape(subImg,(1,1,48,48))
            predictions = theModel.predict(cnnShape, batch_size=32, verbose=2) #(1,2304,2)
            #reshape out to img
            positives = (predictions[0,:,1] > prob_threshold).astype(inImg.dtype)
            positives = np.reshape(positives,(48,48))
            outImg[x-windowSize//2:x+windowSize//2, y-windowSize//2:y+windowSize//2] = positives
    return outImg


if __name__=='__main__':
    import cv2
    img=cv2.imread("Drishti-GS1_files/Drishti-GS1_files/Training/Images/drishtiGS_004.png",0)
    import numpy as np
    img=prepocess_data(img)


    outImg = predictImage(img, loaded_model)

    cv2.imwrite('predicted.jpg',outImg[:,:]*255 )
    plt.subplot(121),plt.imshow(g,'gray'), plt.title('ORIGINAL')
    plt.subplot(122),plt.imshow(outImg[:,:],'gray'), plt.title('PREDICTED1')
    plt.show()