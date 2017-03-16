# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 08:52:21 2017

@author: root
"""
from __feature__ import division
import numpy as np
import tensorflow as tf
from PIL import Image
import tfReadTest2
import time



IMG_H = 128
IMG_W = 64
IMG_CH = 3
LABEL_W = 2

NUM_EPOCHS = 10
BATCH_SIZE = 10
DECAY_STEP = 4000 
SEED = 66478  # Set to None for random seed.

#
def data_type():
    return tf.float32

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])
      
#    eval_data = tf.place
c1_w = tf.Variable(
    tf.truncated_normal([5,5,IMG_CH,32],#5x5 filter depth 32.
                        stddev=0.1,
                        seed=SEED,dtype=data_type()))
c1_b = tf.Variable(tf.zeros([32],dtype = data_type()))
c2_w = tf.Variable(
    tf.truncated_normal([5,5,32,64],
                        stddev=0.1,
                        seed=SEED,dtype = data_type()))
c2_b = tf.Variable(tf.zeros([64],dtype = data_type()))
fc1_w = tf.Variable(
    tf.truncated_normal([IMG_H//4*IMG_W//4*64,512],
                        stddev=0.1,
                        seed=SEED,dtype = data_type()))
fc1_b = tf.Variable(tf.constant(0.1,shape=[512],dtype=data_type()))
fc2_w = tf.Variable(
    tf.truncated_normal([512,LABEL_W],
                        stddev=0.1,
                        seed=SEED,dtype = data_type()))
fc2_b = tf.Variable(tf.constant(0.1,shape=[LABEL_W],dtype=data_type()))

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
    fc1 = tf.nn.relu(tf.matmul(reshape,fc1_w) + fc1_b)
    # Add a 50% dropout during training training only.
    # Dropout also scales activations such that no rescaling is needed at evaluation time
    if train:
        fc1 = tf.nn.dropout(fc1,0.5,seed=SEED)
    return tf.matmul(fc1,fc2_w) + fc2_b
#--------------------end model      
      
#def main(_):
if __name__ == '__main__':
    #This is where training samples and labels are fed to the graph.
    #These placeholder nodes will be fed a batch of training data at each
    #training step using the (feed_dict) argument to the Run() call below.
    train_data_node = tf.placeholder(
        data_type(),
        shape=(BATCH_SIZE,IMG_H,IMG_W,IMG_CH))
        
    train_labels_node = tf.placeholder(tf.int64,shape=(BATCH_SIZE,))

    # Training computation: logits + cross-entropy loss
    logits = model(train_data_node,True)
    loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits,train_labels_node))
    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_w) + tf.nn.l2_loss(fc1_b) + tf.nn.l2_loss(fc2_w) + tf.nn.l2_loss(fc2_b))
    # Add the regularization term to the loss.
    loss += 5e-4*regularizers
    
    # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
    batch = tf.Variable(0,dtype=data_type())
    # Decay once per epoch, using an exponential schedule starting at 0.01
    learningRate = tf.train.exponential_decay(
        0.01,               # Base learning rate.    
        batch * BATCH_SIZE, # Current index into the dataset
        DECAY_STEP,         # Decay step.
        0.95,               # Decay rate.
        staircase=True)
    optimizer = tf.train.MomentumOptimizer(learningRate,0.9).minimize(loss,global_step=batch)
    
    # Predictions for the current training minibatch
    trainPrediction = tf.nn.softmax(logits)

    # Train data
#    img,label = tfReadTest2.read_and_decode("train2_64x128.tfrecords",num_epochs=NUM_EPOCHS)
    init = tf.initialize_all_variables()
    print('Initialized.')
    # Create a local session to run the training.
    with tf.Session() as sess:
        sess.run(init)
        # Start the quen
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
        for i in range(1000):
            print('start')
            startTime = time.time()  
#            imgBatch,labelBatch = tf.train.shuffle_batch([img,label],
#                                                   batch_size=BATCH_SIZE,
#                                                   capacity=2000,min_after_dequeue=100)
            img_batch, label_batch = tf.train.shuffle_batch([img,label],
                                                   batch_size=BATCH_SIZE,
                                                   capacity=2000,min_after_dequeue=1000)
            print('test1')             
            imgBatch,labelBatch= sess.run([img_batch, label_batch])                                       
            print('test2')            
            feed_dict = {train_data_node: imgBatch,
                         train_labels_node: labelBatch}
#            feed_dict = {imgBatch,labelBatch}
            #
#            sess.run([optimizer,loss],feed_dict=feed_dict)
            
            l,prediction = sess.run([loss,trainPrediction],feed_dict=feed_dict)     
            
            
            elapsed_time = time.time() - startTime
            print('time:%.1f ms' %elapsed_time)
            print('l:%f' %l)
            
            
#            print()
        coord.request_stop()
        coord.join(threads)
        sess.close()               
                                
                            
                        
    