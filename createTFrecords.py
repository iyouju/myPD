# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:34:28 2017

@author: root
"""
import os
import random
import numpy as np
#from scipy import io #io.savemat

import tensorflow as tf
from PIL import Image

#-------------------------------
IMG_H = 128
IMG_W = 64
NEGLABEL = [0,1]
POSLABEL = [1,0]
PROPORTION_TRAIN_DATA = 0.9
negPath = '/home/iyouju/pythonPro/DATASETS/INRIAPerson/train_64x128_H96/neg/'
posPath = '/home/iyouju/pythonPro/DATASETS/INRIAPerson/train_64x128_H96/pos/'
#--------------------------------

#
def addTFrecords2(pathList,labelList,writer):
    negUnst = 0
    posUnst = 0
    l = len(pathList)
    for i in range(l):
        path = pathList[i]
        index = labelList[i]
#        print(len(index))
#        print(label)
        img = Image.open(path)
        img = img.resize((IMG_W, IMG_H),Image.BILINEAR)
        img_arr = np.asarray(img)
        sh = img_arr.shape
        # detect the channels of png
        if sh[2] == 4:
            img = Image.fromarray(img_arr[:,:,0:3])
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=index)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
        elif sh[2] == 3:
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=index)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
        else:
            if index[1]:
                posUnst = posUnst + 1
            else:
                negUnst = negUnst + 1
    print('posUnst: %d,\tnegUnst: %d' %(posUnst,negUnst))

#create tfrecords
def createTFrecords():
    negNameList = []
    negNameList = os.listdir(negPath)
    posNameList = []
    posNameList = os.listdir(posPath)

    imgPath = []
    imgLabel = []
    for img in negNameList:
        imgPath.append(negPath+img)
        imgLabel.append(NEGLABEL)
    for img in posNameList:
        imgPath.append(posPath+img)
        imgLabel.append(POSLABEL)
    numTotal = len(imgPath)
    numTrainData = np.floor(numTotal * PROPORTION_TRAIN_DATA)
#    numTestData = numTotal - numTrainData
    
    index = range(numTotal)
    random.shuffle(index)
    pathListTrain = []
    labelListTrain = []
    pathListTest = []
    labelListTest = []
    
    cnt = 0
    for i in index:
        if cnt < numTrainData:
            pathListTrain.append(imgPath[i])
            labelListTrain.append(imgLabel[i])
        else:
            pathListTest.append(imgPath[i])
            labelListTest.append(imgLabel[i])
        cnt += 1

    writer = tf.python_io.TFRecordWriter("trainData_64x128.tfrecords")
    addTFrecords2(pathListTrain,labelListTrain,writer)
    writer.close()
    writer = tf.python_io.TFRecordWriter("testData_64x128.tfrecords")
    addTFrecords2(pathListTest,labelListTest,writer)
    writer.close()
#-------------------------------

if __name__ == '__main__':
    createTFrecords()
    print('createTFrecords is over.')
