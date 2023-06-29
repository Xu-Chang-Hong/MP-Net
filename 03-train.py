# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:03:00 2022
@author: XCH
Environment: PY38
Run: python train_updatev1.py --model XX.h5
"""

import time
start = time.time()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Conv2DTranspose,Input,Dropout,Reshape,Permute,Activation,add
from tensorflow.keras.layers import AveragePooling2D,UpSampling2D,Concatenate
from tensorflow.keras.utils import to_categorical  
from tensorflow.keras.preprocessing.image import img_to_array  
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder    
import cv2
import random
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 7  
np.random.seed(seed)  
 
img_w = 256  
img_h = 256   
n_label = 4

classes = [0.0, 175.0, 239.0, 255.0] 
labelencoder = LabelEncoder()  
labelencoder.fit(classes)  

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img

def get_train_val(val_rate = 0.2,test_rate = 0.2):
    train_url = []    
    train_set = []
    test_set = []
    val_set  = []
    for pic in os.listdir(filepath + 'sample'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    test_num = int(test_rate * total_num)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i]) 
        elif val_num <= i < val_num+test_num:
            test_set.append(train_url[i]) 
        else:
            train_set.append(train_url[i])
    return train_set,val_set,test_set

def generateData(batch_size,data=[]):  
    while True:  
        train_data = []  
        train_label = []  
        batch = 0  
        for i in (range(len(data))): 
            url = data[i]
            batch += 1 
            img = load_img(filepath + 'sample/' + url)
            img = img_to_array(img) 
            train_data.append(img)  
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))  
            # print label.shape  
            train_label.append(label)  
            if batch % batch_size==0: 
                train_data = np.array(train_data)  
                train_label = np.array(train_label).flatten()  
                train_label = labelencoder.transform(train_label)  
                train_label = to_categorical(train_label, num_classes=n_label)  
                train_label = train_label.reshape((batch_size,img_w * img_h,n_label))  
                yield (train_data,train_label)  
                train_data = []  
                train_label = []  
                batch = 0  
 
def generateValidData(batch_size,data=[]):  
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath + 'sample/' + url)
            img = img_to_array(img)  
            valid_data.append(img)  
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))    
            valid_label.append(label)  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label).flatten()  
                valid_label = labelencoder.transform(valid_label)  
                valid_label = to_categorical(valid_label, num_classes=n_label)  
                valid_label = valid_label.reshape((batch_size,img_w * img_h,n_label))  
                yield (valid_data,valid_label)  
                valid_data = []  
                valid_label = []  
                batch = 0  

def generateTestData(batch_size,data=[]):  
    while True:  
        valid_data = []  
        valid_label = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = load_img(filepath + 'sample/' + url)
            img = img_to_array(img)  
            valid_data.append(img)  
            label = load_img(filepath + 'label/' + url, grayscale=True)
            label = img_to_array(label).reshape((img_w * img_h,))    
            valid_label.append(label)  
            if batch % batch_size==0:  
                valid_data = np.array(valid_data)  
                valid_label = np.array(valid_label).flatten()  
                valid_label = labelencoder.transform(valid_label)  
                valid_label = to_categorical(valid_label, num_classes=n_label)  
                valid_label = valid_label.reshape((batch_size,img_w * img_h,n_label))  
                yield (valid_data,valid_label)  
                valid_data = []  
                valid_label = []  
                batch = 0   

import tensorflow.keras.backend as K

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

def mpnet():
    
    inputs = Input((3, img_w, img_h)) # (3, 256, 256)

    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    x = Concatenate(axis=1)([inputs, conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(x) # (35, 128, 128)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    x = Concatenate(axis=1)([pool1, conv2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(x) # (99, 64, 64)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv3)
    x = Concatenate(axis=1)([pool2, conv3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(x) # (227, 32, 32)
    
    # ---------Pyramid Pooling Module---------
    x_31 = AveragePooling2D(pool_size=32, strides=32)(pool3)
    x_31 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(x_31)
    x_31 = Activation(activation='relu')(x_31)
    x_31 = UpSampling2D(size=(32, 32))(x_31)

    x_32 = AveragePooling2D(pool_size=16, strides=16)(pool3)
    x_32 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(x_32)
    x_32 = Activation(activation='relu')(x_32)
    x_32 = UpSampling2D(size=(16, 16))(x_32)

    x_33 = AveragePooling2D(pool_size=8, strides=8)(pool3)
    x_33 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(x_33)
    x_33 = Activation(activation='relu')(x_33)
    x_33 = UpSampling2D(size=(8, 8))(x_33)

    x_34 = AveragePooling2D(pool_size=4, strides=4)(pool3)
    x_34 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(x_34)
    x_34 = Activation(activation='relu')(x_34)
    x_34 = UpSampling2D(size=(4, 4))(x_34)

    x_35 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(pool3)
    x_35 = Activation(activation='relu')(x_35)

    x = Concatenate(axis=1)([x_31, x_32, x_33, x_34, x_35])
    
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    c = Activation(activation='relu')(x)
    # ---------Pyramid Pooling Module---------
    score_pool3 = Conv2D(filters=n_label, kernel_size=(3, 3), padding='same', activation='relu')(c)

    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(c)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
    x = Concatenate(axis=1)([pool3, conv4])
    pool4 = MaxPooling2D(pool_size=(2, 2))(x) # (256, 16, 16)
    
    # ---------Pyramid Pooling Module---------
    x_41 = AveragePooling2D(pool_size=16, strides=16)(pool4)
    x_41 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(x_41)
    x_41 = Activation(activation='relu')(x_41)
    x_41 = UpSampling2D(size=(16, 16))(x_41)

    x_42 = AveragePooling2D(pool_size=8, strides=8)(pool4)
    x_42 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(x_42)
    x_42 = Activation(activation='relu')(x_42)
    x_42 = UpSampling2D(size=(8, 8))(x_42)

    x_43 = AveragePooling2D(pool_size=4, strides=4)(pool4)
    x_43 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(x_43)
    x_43 = Activation(activation='relu')(x_43)
    x_43 = UpSampling2D(size=(4, 4))(x_43)

    x_44 = AveragePooling2D(pool_size=2, strides=2)(pool4)
    x_44 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(x_44)
    x_44 = Activation(activation='relu')(x_44)
    x_44 = UpSampling2D(size=(2, 2))(x_44)

    x_45 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool4)
    x_45 = Activation(activation='relu')(x_45)

    x = Concatenate(axis=1)([x_41, x_42, x_43, x_44, x_45])
    
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    b = Activation(activation='relu')(x)
    # ---------Pyramid Pooling Module---------
    score_pool4 = Conv2D(filters=n_label, kernel_size=(3, 3), padding='same', activation='relu')(b)
    
    # BN_5 = BatchNormalization(axis=1, momentum=0.90, epsilon=0.001)(pool4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(b)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv5)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv5) 
    x = Concatenate(axis=1)([pool4, conv5])
    pool5 = MaxPooling2D(pool_size=(2, 2))(x) # (256, 8, 8)
    
    # ---------Pyramid Pooling Module---------
    x_51 = AveragePooling2D(pool_size=8, strides=8)(pool5)
    x_51 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(x_51)
    x_51 = Activation(activation='relu')(x_51)
    x_51 = UpSampling2D(size=(8, 8))(x_51)

    x_52 = AveragePooling2D(pool_size=4, strides=4)(pool5)
    x_52 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(x_52)
    x_52 = Activation(activation='relu')(x_52)
    x_52 = UpSampling2D(size=(4, 4))(x_52)

    x_53 = AveragePooling2D(pool_size=2, strides=2)(pool5)
    x_53 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(x_53)
    x_53 = Activation(activation='relu')(x_53)
    x_53 = UpSampling2D(size=(2, 2))(x_53)

    x_54 = AveragePooling2D(pool_size=1, strides=1)(pool5)
    x_54 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(x_54)
    x_54 = Activation(activation='relu')(x_54)
    x_54 = UpSampling2D(size=(1, 1))(x_54)

    x_55 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(pool5)
    x_55 = Activation(activation='relu')(x_55)

    x = Concatenate(axis=1)([x_51, x_52, x_53, x_54, x_55])
    
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(x)
    a = Activation(activation='relu')(x)
    # ---------Pyramid Pooling Module---------
    
    fc6 = Conv2D(filters=1024, kernel_size=(1, 1), padding='same', activation='relu')(a)
    fc6 = Dropout(0.3)(fc6) # (1024, 8, 8)
    
    fc7 = Conv2D(filters=1024, kernel_size=(1, 1), padding='same', activation='relu')(fc6)
    fc7 = Dropout(0.3)(fc7) # (1024, 8, 8)
    
    score_fr = Conv2D(filters=n_label, kernel_size=(1, 1), padding='same', activation='relu')(fc7)
    
    score2 = Conv2DTranspose(filters=n_label, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None)(score_fr)
    
    add1 = add(inputs=[score2, score_pool4]) # (n_label, 16, 16)
    
    score4 = Conv2DTranspose(filters=n_label, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None)(add1)
    
    add2 = add(inputs=[score4, score_pool3]) # (n_label, 32, 32)
    
    UpSample = Conv2DTranspose(filters=n_label, kernel_size=(8, 8), strides=(8, 8),
                               padding="valid", activation=None)(add2)
    
    outputs = Conv2D(n_label, (1, 1))(UpSample) # (n_label, 256, 256)
    res = Reshape((n_label,img_w*img_h))(outputs)
    per = Permute((2,1))(res)
    act = Activation('softmax')(per)
    
    model = Model(inputs=inputs, outputs=act)
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['acc', p, r, f,
                           tf.keras.metrics.MeanIoU(num_classes=n_label)])
    model.summary()
    return model
  
def train(args): 
    EPOCHS = 100
    BS = 10
    model = mpnet()  
    modelcheck = ModelCheckpoint(args['model'],monitor='val_acc',save_best_only=True,mode='max')  
    # earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, mode='min', verbose=1, restore_best_weights = True)
    callable = [modelcheck]  
    train_set,val_set,test_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    test_numb = len(test_set)
    print ("The number of train data is",train_numb)  
    print ("The number of val data is",valid_numb)
    print ("The number of test data is",test_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=callable)
    
    # Evaluate the model on the test data using 'evaluate'
    print("Evaluate on test data")
    test = generateTestData(BS,test_set)
    test_data = []
    test_label = []
    for i in range(test_numb//BS):
        data, label = next(test)
        test_data.append(data)
        test_label.append(label)
    
    a = np.array(test_data).shape[0]
    b = np.array(test_data).shape[1]
    c = np.array(test_data).shape[2]
    d = np.array(test_data).shape[3]
    e = np.array(test_data).shape[4]
    
    f = np.array(test_label).shape[2]
    g = np.array(test_label).shape[3]

    test_data = np.array(test_data).reshape((a*b,c,d,e))
    test_label = np.array(test_label).reshape((a*b,f,g))
    
    results = model.evaluate(test_data, test_label, batch_size=BS)
    print("Evaluation Results:", results)
    
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_accuracy")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_accuracy")
    plt.title("Training Loss and Accuracy on winter crops in Erhai")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="./",
                    help="path to training data")
    ap.add_argument("-m", "--model", default="Train-Model.h5",
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args

if __name__=='__main__':
    args = args_parse()
    filepath = args['data']
    train(args)  

end = time.time()
runTime = end - start
print("Run Time:", runTime)