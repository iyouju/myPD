# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:12:37 2017

@author: root
"""
import numpy as np
import tensorflow as tf
from PIL import Image



IMG_H = 128
IMG_W = 64

class TFreader:
    #-------------
    fileName = None
    batchSize = None
    numEpochs = None
    graph = None
    session = None
#    img = None
#    labels = None
    imgBatch = None
    labelsBatch = None
    coord = tf.train.Coordinator()
    threads = None

    # controle the consumption of RAM
    setGPU = True
    sessConfig = tf.ConfigProto()
    sessConfig.gpu_options.allow_growth=True

    def __init__(self,fileName,batch_size,num_epochs=None,setGPU=True):
        self.fileName = fileName
        self.batchSize = batch_size
        self.numEpochs = num_epochs
        self.setGPU = setGPU
        self.graph = tf.Graph()
        self.coord = tf.train.Coordinator()

        self.defineGraph()

        print('TFreader is initialized.')


    def read_and_decode(self,filename,num_epochs=None):
        filename_queue = tf.train.string_input_producer([filename],num_epochs=num_epochs)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([2], tf.int64),
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                           })

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [IMG_H, IMG_W, 3])
    #    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        index=tf.cast(features['label'], tf.int32)
    #    print label[1]
        return img, index

    def defineGraph(self):
        with self.graph.as_default():
            img,labels = self.read_and_decode(self.fileName,num_epochs=self.numEpochs)
            self.imgBatch,self.labelsBatch = tf.train.shuffle_batch([img,labels],
                                                                    batch_size = self.batchSize,
#                                                                    num_threads=2,
                                                                    capacity = 500,
                                                                    min_after_dequeue = 100)
#            print type(self.labelsBatch)
    def reBatch(self):
        with self.graph.as_default():

            #--------------
            if self.setGPU:
                self.session = tf.Session(graph=self.graph,config=self.sessConfig)
            else:
                self.session = tf.Session(graph=self.graph)
    #        with self.session as sess:
            tf.local_variables_initializer().run(session=self.session) # epoch计数变量是local variable
            tf.global_variables_initializer().run(session=self.session)
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.session, coord=self.coord)
#            if not self.coord.should_stop():
#                img,label = self.session.run([self.imgBatch,self.labelsBatch])
#            else:
#                img = None
#                label = None
            try:
                img,label = self.session.run([self.imgBatch,self.labelsBatch])
            except tf.errors.OutOfRangeError:    # 文件队列关闭后，终止循环
                img = None
                label = None
                print('None')

        self.coord.request_stop()
        self.coord.join(self.threads)

        return img,label



if __name__ == '__main__':
    BATCH_SIZE = 5
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE,IMG_H,IMG_W,1))

    train_labels_node = tf.placeholder(tf.int64,shape=(BATCH_SIZE,))

    tfReader = TFreader("train2_64x128.tfrecords",BATCH_SIZE,num_epochs=1,useGPU=False)
    i = 0
    while True:
        img,label = tfReader.reBatch()
        i += 1
        if img != None :
            print label[0]
            print label.shape
            print i
        else:
            print('Queue is empty.')
            break
    print('test is over.')
