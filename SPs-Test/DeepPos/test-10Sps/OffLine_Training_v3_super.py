from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

from get_v3 import get_csi
import os
import shutil

if os.path.isdir(os.getcwd()+'/Models'):

        shutil.rmtree(os.getcwd()+'/Models')

os.mkdir(os.getcwd()+'/Models')

random_points=list(np.array([2, 5, 6,8, 10,12, 14,16,17, 19])-1)
for k in range(1,11):
    csi = get_csi(k,random_points)
    csi = np.squeeze(csi)
    print(csi.shape)


    point1=csi[0:20]
    point2=csi[20:40]
    point3=csi[40:60]
    point4=csi[60:80]
    point5=csi[80:100]
    point6 = csi[100:120]
    point7 = csi[120:140]
    point8 = csi[140:160]
    point9 = csi[160:180]
    point10 = csi[180:200]




    points=[point1,point2,point3,point4,point5,point6,point7,point8,point9,point10]
    total=points

    # Parameters
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 50
    n_labels = 9
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
    #Z={'Z': tf.Variable(decoder_input,name='z')}

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
            rows=x.shape[0]
            labels=np.array([0,0,0,0,0,0,0,0,0]*rows)
            labels=labels.reshape(rows,n_labels)
            labels[0:20,0]=1  #label for point 1
            labels[20:40,1]=1  #label for point 2
            labels[40:60,2]=1  #label for point 3
            labels[60:80,3]=1  #label for point 4
            labels[80:100,4]=1  #label for point 5
            labels[100:120, 5] = 1
            labels[120:140, 6] = 1
            labels[140:160, 7] = 1
            labels[160:180, 8] = 1



            label=labels
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: x,y: label})
            tf.add_to_collection("predict", y_pred)
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
                # print("Epoch:", '%04d' % (epoch+1),"accuracy", "{:.9f}".format(accuracy))

        print("Optimization Finished for point " + str(k) + "! ")
        os.mkdir(os.getcwd()+'/Models/Test'+str(k))
        saver.save(sess,os.path.join(os.getcwd()+'/Models/Test'+str(k), 'trained_variables'+'.ckpt'))
    tf.reset_default_graph()
    #

