# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 08:52:21 2017

@author: root
"""
#from __feature__ import division
import numpy as np
import tensorflow as tf
from PIL import Image
#from TFreader import *
import time

IMG_H = 128
IMG_W = 64
IMG_CH = 3
LABEL_W = 2

NUM_EPOCHS = 40
BATCH_SIZE = 50
DECAY_STEP = 4000
SEED = 66478  # Set to None for random seed.

def data_type():
    return tf.float32

class pdNet:
  # Set to None for random seed.

    fileName = None
    batchSize = None
    numEpochs = None
    graph = None
    session = None
    coord = None
    threads = None

    merge = None
    
    def __init__(self,file_name,batch_size,num_epochs=None):
        self.fileName = file_name
        self.batchSize = batch_size
        self.numEpochs = num_epochs
        self.graph = tf.Graph()
        self.coord = tf.train.Coordinator()

        self.defineGraph()
        self.session = tf.Session(graph=self.graph)
        
        self.writer = tf.summary.FileWriter('./board', self.graph)
        self.merge = tf.summary.merge_all()
        
        print('TFreader is initialized.')

#    def __del__(self):


        
    def read_and_decode(self,filename,num_epochs=None):
        filename_queue = tf.train.string_input_producer([filename],
                                                        num_epochs=num_epochs)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([2], tf.int64),
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                           })

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [IMG_H, IMG_W, 3])
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        index=tf.cast(features['label'], tf.int32)
    #    print label[1]
        return img, index

    def defineGraph(self):
        with self.graph.as_default():
            img,labels = self.read_and_decode(self.fileName,num_epochs=self.numEpochs)
            self.imgBatch,self.labelsBatch = tf.train.shuffle_batch([img,labels],
#                                                            num_epochs = self.numEpochs,
                                                            batch_size = self.batchSize,
                                                            capacity = 500,
                                                            min_after_dequeue = 100)  
            #----		net variables
            #    eval_data = tf.place
            with tf.name_scope('model'):
                #This is where training samples and labels are fed to the graph.
                #These placeholder nodes will be fed a batch of training data at each
                #training step using the (feed_dict) argument to the Run() call below.
                self.train_data_node = tf.placeholder(
                    data_type(),
                    shape=(BATCH_SIZE,IMG_H,IMG_W,IMG_CH),
                    name = 'train_data_node')
            
                self.train_labels_node = tf.placeholder(data_type(),shape=(BATCH_SIZE,LABEL_W),
                                                        name = 'train_labels_node')
                c1_w = tf.Variable(
                    tf.truncated_normal([8,8,IMG_CH,64],#5x5 filter depth 32.
                                        stddev=0.1,
                                        seed=SEED,dtype=data_type(),
                                        name = 'c1_w'))
                c1_b = tf.Variable(tf.zeros([64],dtype = data_type(),
                                        name = 'c1_b'))
                c2_w = tf.Variable(
                    tf.truncated_normal([4,4,64,32],
                                        stddev=0.1,
                                        seed=SEED,dtype = data_type(),
                                        name = 'c2_w'))
                c2_b = tf.Variable(tf.zeros([32],dtype = data_type(),
                                        name = 'c2_b'))
                fc1_w = tf.Variable(
                    tf.truncated_normal([IMG_H//4*IMG_W//4*32,1024],
                                        stddev=0.1,
                                        seed=SEED,dtype = data_type(),
                                        name = 'fc1_w'))
                fc1_b = tf.Variable(tf.constant(0.1,shape=[1024],dtype=data_type(),
                                        name = 'fc1_b'))
                fc2_w = tf.Variable(
                    tf.truncated_normal([1024,LABEL_W],
                                        stddev=0.1,
                                        seed=SEED,dtype = data_type(),
                                        name = 'fc2_w'))
                fc2_b = tf.Variable(tf.constant(0.1,shape=[LABEL_W],dtype=data_type(),
                                        name = 'fc2_b'))
                
                #-----------------------------
                tf.summary.histogram('c1_w',c1_w)
                #---------------------	end net variables
                #
                def model(data,train=False):
                    # shape matches the data layout:[image index,y,x,depth].
                    c1 = tf.nn.conv2d(data,
                                      c1_w,
                                      strides=[1,1,1,1],
                                        padding = 'SAME')
                    # Bias and rectified linear non-linearity.
                    relu1 = tf.nn.relu(tf.nn.bias_add(c1,c1_b))
                    # Max pooling
        
                    pool1 = tf.nn.max_pool(relu1,
                                          ksize=[1,2,2,1],
                                            strides=[1,2,2,1],
                                            padding='SAME')
                    c2 = tf.nn.conv2d(pool1,
                                      c2_w,
                                      strides=[1,1,1,1],
                                        padding='SAME')
                    relu2 = tf.nn.relu(tf.nn.bias_add(c2,c2_b))
                    pool2 = tf.nn.max_pool(relu2,
                                           ksize=[1,2,2,1],
                                            strides=[1,2,2,1],
                                            padding='SAME')
                    # Reshape the feature map cuboid into a 2D matrix to feed it to the fully connected layers.
                    poolShape = pool2.get_shape().as_list()
                    reshape = tf.reshape(
                                        pool2,
                                        [poolShape[0],poolShape[1]*poolShape[2]*poolShape[3]])
                    # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
                    fc1 = tf.nn.tanh(tf.matmul(reshape,fc1_w) + fc1_b)
                    # Add a 50% dropout during training training only.
                    # Dropout also scales activations such that no rescaling is needed at evaluation time
                    if train:
                        fc1 = tf.nn.dropout(fc1,0.5,seed=SEED)
                    return tf.nn.sigmoid(tf.matmul(fc1,fc2_w) + fc2_b)
    #                return (tf.matmul(fc1,fc2_w) + fc2_b)
                #--------------------end model

            # Training computation: logits + cross-entropy loss
            logits = model(self.train_data_node,True)
            self.loss1 = tf.reduce_mean(
                                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=self.train_labels_node))
            # L2 regularization for the fully connected parameters.
            regularizers = (tf.nn.l2_loss(fc1_w) + tf.nn.l2_loss(fc1_b) + tf.nn.l2_loss(fc2_w) + tf.nn.l2_loss(fc2_b))
            # Add the regularization term to the loss.
            loss = self.loss1 + 5e-4*regularizers
            tf.summary.histogram('loss1',self.loss1)
            tf.summary.histogram('loss',loss)

            # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
            batch = tf.Variable(0,dtype=data_type())
            # Decay once per epoch, using an exponential schedule starting at 0.01
            self.learningRate = tf.train.exponential_decay(
                0.1,               # Base learning rate.
                batch * self.batchSize, # Current index into the dataset
                DECAY_STEP,         # Decay step.
                0.99,               # Decay rate.
                staircase=True)
            # learningRate = 0.1
            self.optimizer = tf.train.MomentumOptimizer(self.learningRate,0.9).minimize(loss,global_step=batch)
    
            # Predictions for the current training minibatch
            self.trainPrediction = tf.nn.sigmoid(logits)
#            self.trainPrediction = (logits)

#        self.merge = tf.merge_v2_checkpoints()
#        self.merge = tf.merge_all_summaries()
    #------------   END def defineGraph()

        #
    def error_rate(self,predictions, labels):
      """Return the error rate based on dense predictions and sparse labels."""
#      num = len(predictions)
#      pmax = np.argmax(predictions, 1)
#      lmax = np.argmax(labels, 1)
#      s = np.sum(pmax==lmax)
#      rate = 1 - s/num
#      return rate
      return 100.0 - (
          100.0 *
          np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
          np.float32(len(predictions)))
      
    def train(self):

#        self.session = tf.Session(graph=self.graph)
        with self.session :
            
            tf.local_variables_initializer().run(session=self.session) # epoch计数变量是local variable
            tf.global_variables_initializer().run(session=self.session)

            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.session, coord=self.coord)
            try:
                step = 0
                while not self.coord.should_stop():
                    startTime = time.time()
                    with self.session.as_default():
                        imgBatch_r,labelBatch_r = self.session.run([self.imgBatch,self.labelsBatch])#session.run([tfReader.imgBatch,tfReader.labelsBatch])

                    feed_dict = {self.train_data_node: imgBatch_r,
                                 self.train_labels_node: labelBatch_r}
                    _,lrate,l,prediction = self.session.run([self.optimizer,
#                                                             self.merge,
                                                             self.learningRate,
                                                             self.loss1,
                                                             self.trainPrediction],
                                                             feed_dict = feed_dict)
                    elapsed_time = time.time() - startTime
                    err = self.error_rate(prediction,labelBatch_r)
                    if not step%10:
                        print('step:%d time:%f s' %(step,elapsed_time))
#                        print('time:%f s' %elapsed_time)
                        print('loss1:%f ,learnrate:%f' %(l,lrate))
                        print('error_rate: %f' %err)
                    
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Train is over.')
            finally:
                self.coord.request_stop()
                self.coord.join(self.threads)

        self.session.close()
    
#    def test(self):
#        img,labels = self.read_and_decode(self.fileName,num_epochs=self.numEpochs)
        
if __name__ == '__main__':
    net = pdNet("trainData_64x128.tfrecords",BATCH_SIZE,num_epochs=NUM_EPOCHS)
    net.train()
    del net