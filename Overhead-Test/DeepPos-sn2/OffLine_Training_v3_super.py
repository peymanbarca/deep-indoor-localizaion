from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
#from read_from_db import read_csi_from_db
import time
from get_v3 import get_csi
import os
import random
import psutil
import shutil


#num_of_SPs=[6,10,14,17]
num_of_SPs=17

for sp in [num_of_SPs]:
    total_elapsed_time=[]
    total_memory_used=[]
    total_memory_perc=[]
    random_list=random.sample(range(17), sp)
    if os.path.isdir(os.getcwd()+r'\Models'):
            shutil.rmtree(os.getcwd()+r'\Models')
    os.mkdir(os.getcwd()+r'\Models')

    for test_i in [random_list.index(random_list[-1])]:  # we must create a session for each of loc --> model all other sps with one net!!
        csi = get_csi(random_list[test_i],random_list)
        csi = np.squeeze(csi)
        st = time.time()
        points=[]
        for t in range(sp):
            points.append(csi[t*20:(t+1)*20])

        total=points

        # Parameters
        learning_rate = 0.01
        training_epochs = 1000
        display_step = 50
        n_labels = sp-1
        #

        # Network Parameters
        n_hidden_1 = 45 # 1st layer num features
        n_hidden_2 = 20 # 2nd layer num features
        n_hidden_3=10
        n_hidden_4=5
        n_input = 90

        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_labels],name='Labels')

        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name='w1'),
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name='w2'),
            'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]),name='w3'),
            'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4]),name='w4'),

            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4+n_labels, n_hidden_3]),name='w5'),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2]),name='w6'),
            'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1]),name='w7'),
            'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input]),name='w8'),
        }


        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1]),name='b1'),
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2]),name='b2'),
            'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3]),name='b3'),
            'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4]),name='b4'),

            'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3]),name='b5'),
            'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2]),name='b6'),
            'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1]),name='b7'),
            'decoder_b4': tf.Variable(tf.random_normal([n_input]),name='b8'),
        }



        # Building the encoder
        def encoder(x):
            # Encoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                           biases['encoder_b1']))
            # Decoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                           biases['encoder_b2']))

            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                           biases['encoder_b3']))

            layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                           biases['encoder_b4']))
            return layer_4


        # Building the decoder
        def decoder(x):
            # Encoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                           biases['decoder_b1']))
            # Decoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                           biases['decoder_b2']))

            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                           biases['decoder_b3']))

            layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                           biases['decoder_b4']))
            return layer_4

        # Construct model
        encoder_op = encoder(X)
        decoder_input = tf.concat([encoder_op,y],1,name="op_to_restore")

        decoder_op = decoder(decoder_input)

        # Prediction
        y_pred = decoder_op

        # Targets (Labels) are the input data.
        y_true = X

        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


        # Initializing the variables
        init = tf.global_variables_initializer()


        saver = tf.train.Saver()


        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                # Loop over all batches
                x=csi
                rows=x.shape[0] # == sp-1 * num_of_pcakets(20)
                labels=np.array(([0]*(sp-1))*rows)
                labels=labels.reshape(rows,n_labels)  # sp-1*20 , sp-1
                for t in range(sp-1):
                    labels[t*20:(t+1)*20,t]=1


                label=labels
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={X: x,y: label})
                tf.add_to_collection("predict", y_pred)
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
                    # print("Epoch:", '%04d' % (epoch+1),"accuracy", "{:.9f}".format(accuracy))

            print("Optimization Finished for point " + str(test_i+1) + "! ")
            os.mkdir(os.getcwd()+'/Models/Test'+str(test_i+1))
            saver.save(sess,os.path.join(os.getcwd()+r'\Models\Test'+str(test_i+1), 'trained_variables'+'.ckpt'))
        tf.reset_default_graph()
        total_elapsed_time.append( float("{0:.2f}".format(time.time()-st)))
        total_memory_used.append(float("{0:.2f}".format(list(psutil.virtual_memory())[3]/1e9)))
        total_memory_perc.append(list(psutil.virtual_memory())[2])

print(float("{0:.2f}".format(np.mean(total_elapsed_time))),float("{0:.2f}".format(np.sum(total_elapsed_time))))
print(float("{0:.2f}".format(np.mean(total_memory_used))))
print(float("{0:.2f}".format(np.mean(total_memory_perc))))
