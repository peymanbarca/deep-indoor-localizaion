import tensorflow as tf
import numpy as np
from get_v3 import get_test_csi
import os
import time


random_point=list(np.array([1,2,4,5, 6, 7,9, 10,11,12, 13,15,16,17, 18,19]))
param=16
test_loc=random_point[param-1]
print(test_loc)
csi=get_test_csi(test_loc) # 10*90
csi=np.squeeze(csi)



n_hidden_1 = 30 # 1st layer num features
n_hidden_2 = 20 # 2nd layer num features
n_hidden_3=10
n_hidden_4=5
n_input = 90


ers=[]
times=[]

# Applying encode and decode over test set
for k in range(1,16):  # give to all trained network for each point ---> DeepFi
    X = tf.placeholder("float", [None, n_input])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver1 = tf.train.import_meta_graph('model/Test'+str(param)+'/'+str(k)+
                                            '/trained_variables'+str(param)+str(k)+'.ckpt.meta')
        saver1.restore(sess, tf.train.latest_checkpoint('model/Test'+str(param)+'/'+str(k)+'/'))
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
        decoder_op = decoder(encoder_op)

        # Prediction
        y_pred = decoder_op

        new_input=csi
        print(new_input.shape)
        predictions = sess.run(y_pred, feed_dict={X: new_input})

        print((predictions.shape))

        print('The error is\n' + 'test : ' + str(test_loc) + ' point ' + str(k))
        errpr=np.mean(np.square(predictions-new_input))
        print(errpr)
        ers.append(errpr)
        times.append(time.time()-st)
    tf.reset_default_graph()
    print('***************')

from Detemine_error import compute_error_metric

sorted_errors=sorted(ers)
a = sorted_errors[0:3]
errors=list(ers)
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

