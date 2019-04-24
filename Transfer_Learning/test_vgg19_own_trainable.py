"""
Simple tester for the vgg19_own_trainable
"""

import tensorflow as tf

import vgg19_own_trainable as vgg19
import utils

img1= utils.load_image_rwanda_zip("1.0.png")
img1_true_result = [1 if i == 1 else 0 for i in range(3)]  # 1-hot result for tiger

batch1 = img1.reshape((1, 400, 400, 3))

with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [None, 400, 400, 3])
    true_out = tf.placeholder(tf.float32, [None, 3])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)
    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())

    # test classification
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.prediction(prob[0])

    # simple 1-step training
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_out, logits=vgg.fc8))
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

    # test classification again, should have a higher probability about tiger
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.prediction(prob[0])

    # test save
    #vgg.save_npy(sess, './test-save.npy')
