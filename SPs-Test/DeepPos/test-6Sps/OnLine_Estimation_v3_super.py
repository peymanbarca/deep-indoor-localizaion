import tensorflow as tf
import numpy as np
from get_v3 import get_test_csi
import os
import multiprocessing
from Detemine_error import compute_error_metric
import time


random_points=list(np.array([2, 5, 8, 10, 14, 18])-1)
param=6
test_loc=random_points[param-1]
csi=get_test_csi(test_loc)
csi=np.squeeze(csi)
print(csi.shape)



n_hidden_1 = 45 # 1st layer num features
n_hidden_2 = 20 # 2nd layer num features
n_hidden_3=10
n_hidden_4=5
n_input = 90
n_labels = 5

X = tf.placeholder("float", [None, n_input])

locs = list(np.linspace(1 ,6, 6))


new_input = csi[1:6, :]  # for example give some packets input
#print(new_input.shape)

errors = []
times=[]


# Applying encode and decode over test set
for i in range(1):
    with tf.Session() as sess:
        saver1 = tf.train.import_meta_graph('Models/Test' + str(param) + '/trained_variables' + '.ckpt.meta')
        saver1.restore(sess, tf.train.latest_checkpoint('Models/Test' + str(param) + '/'))
        for j in range(5):              #ye session darim & label haye mokhtakef ra dar an micharkhanim
            X = tf.placeholder("float", [None, n_input])
            y = tf.placeholder("float", [None, n_labels])
            #sess.run(tf.global_variables_initializer())

            st=time.time()
            w1=sess.run('w1:0')     #90*30  inha trian shode va fix hastand
            w2 = sess.run('w2:0')   #30*20
            w3 = sess.run('w3:0')   #20*10
            w4 = sess.run('w4:0')   #10*5
            w5 = sess.run('w5:0')
            w6 = sess.run('w6:0')
            w7 = sess.run('w7:0')
            w8 = sess.run('w8:0')
            b1 = sess.run(('b1:0'))
            b2 = sess.run(('b2:0'))
            b3 = sess.run(('b3:0'))
            b4 = sess.run(('b4:0'))
            b5 = sess.run(('b5:0'))
            b6 = sess.run(('b6:0'))
            b7 = sess.run(('b7:0'))
            b8 = sess.run(('b8:0'))


            def encoder(x):
                # Encoder Hidden layer with sigmoid activation #1
                layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w1),
                                              b1))
                # Decoder Hidden layer with sigmoid activation #2
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2),
                                               b2))

                layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w3),
                                               b3))

                layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, w4),
                                               b4))
                return layer_4

            def decoder(x):
                # Encoder Hidden layer with sigmoid activation #1
                layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, w5),
                                               b5))
                # Decoder Hidden layer with sigmoid activation #2
                layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w6),
                                               b6))

                layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w7),
                                               b7))

                layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, w8),
                                               b8))
                return layer_4


            # Construct model
            encoder_op = encoder(X)
            decoder_input = tf.concat([encoder_op,y],1)
            decoder_op = decoder(decoder_input)
            rows=new_input.shape[0]
            labels=np.array([0,0,0,0,0]*rows)
            labels=labels.reshape(rows,n_labels)
            labels[:,j]=1  #charkhondan label

            label=labels
            y_pred = decoder_op
            predictions = sess.run(y_pred, feed_dict={X: new_input,y: label}) 
            a=(np.mean(np.square(predictions - new_input)))
            errors.append(a)
            times.append(time.time()-st)

    tf.reset_default_graph()
    print('***')

print(test_loc)
errors=np.array(errors)
print(errors)




sorted_errors=sorted(errors)
a = sorted_errors[0:3]
errors=list(errors)
ers=[]
keys =[]
for i in a:
    ers.append(i)
    keys.append(errors.index(i))

true_keys=[]
for i in keys:
    if i<test_loc-1:
        true_keys.append(i+1)
    else:
        true_keys.append(i+2)

print('3 best candidate locations:')
print(true_keys)

print('\n')
er=compute_error_metric(true_keys[0],true_keys[1],param,ers[0],ers[1])
print('Error is ' + str(er) +'  meters !')
print('Took : ' + str(np.sum(times)))


