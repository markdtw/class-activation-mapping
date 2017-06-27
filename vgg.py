from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb

import numpy as np 
import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer, l2_regularizer

class VGG16_GAP:
    # Train on CALTECH256
    def __init__(self, lr=1e-4, num_batches=1e4, train_mode=True):
        self.g_step = tf.train.get_or_create_global_step()
        self.lr = lr
        self.num_batches = num_batches
        self.train_mode = train_mode
        if self.train_mode:
            self.params = np.load(os.path.join('caffe_pretrained', 'VGG_ILSVRC_16_layers.npy')).item()
        else:
            self.params = np.load(os.path.join('caffe_pretrained', 'vgg16CAM_train_iter_90000.npy')).item()

        self.l2_beta = 1e-2
        self.n_labels = 257 if self.train_mode else 1000    # CALTECH256 or IMAGENET
        self.means = [ 123.68, 116.779, 103.939]            # IMAGENET in r, g, b order

    def build(self, images):
        r, g, b = tf.split(images, 3, 3)
        images = tf.concat([b - self.means[2], g - self.means[1], r - self.means[0]], 3)

        net = self.conv_layer(images, 3, 3, 64, 'conv1_1')
        net = self.conv_layer(net, 3, 64, 64, 'conv1_2')
        net = self.max_pool(net, name='pool1')

        net = self.conv_layer(net, 3, 64, 128, 'conv2_1')
        net = self.conv_layer(net, 3, 128, 128, 'conv2_2')
        net = self.max_pool(net, name='pool2')
        
        net = self.conv_layer(net, 3, 128, 256, 'conv3_1')
        net = self.conv_layer(net, 3, 256, 256, 'conv3_2')
        net = self.conv_layer(net, 3, 256, 256, 'conv3_3')
        net = self.max_pool(net, name='pool3')

        net = self.conv_layer(net, 3, 256, 512, 'conv4_1')
        net = self.conv_layer(net, 3, 512, 512, 'conv4_2')
        net = self.conv_layer(net, 3, 512, 512, 'conv4_3')
        net = self.max_pool(net, name='pool4')

        net = self.conv_layer(net, 3, 512, 512, 'conv5_1')
        net = self.conv_layer(net, 3, 512, 512, 'conv5_2')
        net = self.conv_layer(net, 3, 512, 512, 'conv5_3')

        with tf.variable_scope('CAM'):
            self.cam_conv = self.conv_layer(net, 3, 512, 1024, 'CAM_conv')
            self.gap = tf.reduce_mean(self.cam_conv, [1, 2], name='CAM_GAP')
            if self.train_mode:
                self.gap = tf.nn.dropout(self.gap, 0.5)

            self.cam_fc = self.fc_layer(self.gap, 1024, self.n_labels, 'CAM_fc')

        with tf.variable_scope('CAM', reuse=True):
            self.cam_conv_resize = tf.image.resize_images(self.cam_conv, [224, 224])
            self.cam_fc_value = tf.nn.bias_add(tf.get_variable('CAM_fc_w'), tf.get_variable('CAM_fc_b'))


    def loss(self, labels):
        self.xen_loss_op = tf.losses.sparse_softmax_cross_entropy(labels, self.logits)
        self.reg_loss_op = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss_op = tf.add(self.xen_loss_op, self.reg_loss_op, name='total_loss')
        self.correct_op = tf.equal(tf.argmax(self.logits, 1), labels)

    def train(self):
        return tf.train.AdamOptimizer(self.lr).minimize(self.loss_op, global_step=self.g_step)

    def max_pool(self, bottom, name, k=2, s=2):
        return tf.nn.max_pool(bottom, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)
    def conv_layer(self, bottom, f_size, in_c, out_c, name, stride=1):
        f, b = self.get_conv_var(f_size, in_c, out_c, name)
        conv = tf.nn.conv2d(bottom, f, [1, stride, stride, 1], padding='SAME')
        return tf.nn.relu(tf.nn.bias_add(conv, b))
    def fc_layer(self, bottom, in_size, out_size, name):
        w, b = self.get_fc_var(in_size, out_size, name)
        x = tf.reshape(bottom, [-1, in_size])
        return tf.nn.xw_plus_b(x, w, b)
    def get_conv_var(self, f_size, in_c, out_c, name):
        if name in self.params.keys():
            w_initializer = tf.constant_initializer(self.params[name][0].transpose((2, 3, 1, 0)))
            b_initializer = tf.constant_initializer(self.params[name][1])
        else:
            b_initializer = w_initializer = xavier_initializer()
        f = tf.get_variable(name+'_f', [f_size, f_size, in_c, out_c],
                initializer=w_initializer, regularizer=l2_regularizer(self.l2_beta))
        b = tf.get_variable(name+'_b', [out_c], initializer=b_initializer)
        return f, b
    def get_fc_var(self, in_size, out_size, name):
        if name in self.params.keys():
            w_initializer = tf.constant_initializer(self.params[name][0].transpose((1, 0)))
            b_initializer = tf.constant_initializer(self.params[name][1])
        else:
            b_initializer = w_initializer = xavier_initializer()
        w = tf.get_variable(name+'_w', [in_size, out_size],
                initializer=w_initializer, regularizer=l2_regularizer(self.l2_beta))
        b = tf.get_variable(name+'_b', [out_size], initializer=b_initializer)
        return w, b
