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

negPath = '/home/iyouju/pythonPro/DATASETS/INRIAPerson/train_64x128_H96/neg/'
posPath = '/home/iyouju/pythonPro/DATASETS/INRIAPerson/train_64x128_H96/pos/'
#--------------------------------

#==============================================================================
# #
# def addTFrecords(path,NameList,index,writer):
#     for imgName in NameList:
#         imgPath = path + imgName
#         img = Image.open(imgPath)
#         img_arr = np.asarray(img)
#         sh = img_arr.shape
# #        print('sh:%d' %sh(2))
#         if sh(2) == 3:
#             img = img.resize((IMG_W, IMG_H),Image.BILINEAR)
#             img_raw = img.tobytes()              #将图片转化为原生bytes
# #            print(img_raw.size)
#             example = tf.train.Example(features=tf.train.Features(feature={
#                 "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
#                 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#             }))
#             writer.write(example.SerializeToString())  #序列化为字符串
#         else:
#             print()
# #-------------------------------
#==============================================================================

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
    
    index = range(len(imgPath))
    random.shuffle(index)
    pathList = []
    labelList = []
    for i in index:
        pathList.append(imgPath[i])
        labelList.append(imgLabel[i])
    
    writer = tf.python_io.TFRecordWriter("train2_64x128.tfrecords")
    addTFrecords2(pathList,labelList,writer)
#    addTFrecords(negPath,negNameList,0,writer)
#    addTFrecords(posPath,posNameList,1,writer)
    writer.close()
#-------------------------------

if __name__ == '__main__':
    createTFrecords()
    print('createTFrecords is over.')
