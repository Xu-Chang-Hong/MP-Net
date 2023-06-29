# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 17:19:28 2022
@author: XCH
Environment: PY38
Run: python predict.py --model XX.h5
"""

import time
start = time.time()

import numpy as np
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import argparse
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEST_SET = ['sample.png']

image_size = 256

classes = [0.0, 175.0, 239.0, 255.0]
labelencoder = LabelEncoder()  
labelencoder.fit(classes) 

from keras import backend as K

def p(y_true, y_pred):
    """Precision"""
    tp= K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    pp= K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = tp/ (pp+ K.epsilon())
    return precision
    
def r(y_true, y_pred):
    """Recall"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = tp / (pp + K.epsilon())
    return recall
 
def f(y_true, y_pred):
    """F1-score"""
    precision = p(y_true, y_pred)
    recall = r(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
    ap.add_argument("-s", "--stride", required=False,
        help="crop slide stride", type=int, default=image_size)
    args = vars(ap.parse_args())    
    return args

    
def predict(args):
    print("[INFO] loading network...")
    model = load_model(args["model"],custom_objects={'p': p, 'r': r, 'f': f})
    stride = args['stride']
    for n in range(len(TEST_SET)):
        path = TEST_SET[n]
        image = cv2.imread('./' + path)
        h,w,_ = image.shape
        padding_h = (h//stride + 1) * stride 
        padding_w = (w//stride + 1) * stride
        padding_img = np.zeros((padding_h,padding_w,3),dtype=np.uint8)
        padding_img[0:h,0:w,:] = image[:,:,:]
        padding_img = padding_img.astype("float") / 255.0
        padding_img = img_to_array(padding_img)
        print('src:',padding_img.shape)
        mask_whole = np.zeros((padding_h,padding_w),dtype=np.uint8)
        for i in range(padding_h//stride):
            for j in range(padding_w//stride):
                crop = padding_img[:3,i*stride:i*stride+image_size,j*stride:j*stride+image_size]
                _,ch,cw = crop.shape
                if ch != 256 or cw != 256:
                    print('invalid size!')
                    continue
                    
                crop = np.expand_dims(crop, axis=0)
                y_predict = model.predict(crop)
                pred = np.argmax(y_predict,axis=-1)
                pred = labelencoder.inverse_transform(pred[0])   
                pred = pred.reshape((256,256)).astype(np.uint8)
                mask_whole[i*stride:i*stride+image_size,j*stride:j*stride+image_size] = pred[:,:]

        cv2.imwrite('./'+str(n+1)+'.png',mask_whole[0:h,0:w])
    
if __name__ == '__main__':
    args = args_parse()
    predict(args)

end = time.time()
runTime = end - start
print("Run Time:", runTime)