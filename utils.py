from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import glob

#import caffe
import scipy.misc
import numpy as np
import tensorflow as tf

from tqdm import tqdm

def extract(model, weights):
    """extract from .caffemodel base on .prototxt to .npy, don't extract original vgg's fc layers
    Args:
        model: the .prototxt file path
        weights: the .caffemodel file path
    """
    net = caffe.Net(model, 1, weights=weights)

    parameters = {}
    for layer, param in net.params.iteritems():
        if 'fc6' == layer or 'fc7' == layer or 'fc8' == layer:
            continue
        w = param[0].data
        b = param[1].data
        parameters[layer] = [w, b]

    if 'CAM' in model:
        filename = 'vgg16CAM_train_iter_90000.npy'
    else:
        filename = 'VGG_ILSVRC_16_layers.npy'
    np.save(os.path.join('caffe_pretrained', filename), parameters)


class CALTECH256_loader():

    def __init__(self, batch_size, num_epochs, num_threads=4):
        filename = 'calt256_224_224.tfrecords'
        if not os.path.exists(os.path.join('CALTECH256', filename)):
            print ('Generating {}'.format(filename))
            images, labels = self.readCALTECH256()
            self.convertToTFRecords(images, labels, filename)

        self.images, self.labels = self.readFromTFRecords(os.path.join('CALTECH256', filename),
                batch_size, num_epochs, [224, 224, 3], num_threads)
        self.num_batches = 30607 // batch_size

    def readCALTECH256(self):
        category_list = sorted(glob.glob('CALTECH256/*'))
        images = []
        labels = []
        for c in tqdm(category_list):
            filename_list = sorted(glob.glob(c + '/*'))
            for f in filename_list:
                image = scipy.misc.imread(f, mode='RGB')
                image = scipy.misc.imresize(image, (224, 224))
                assert image.shape == (224, 224, 3)
                images.append(image)

                label = int(f.split('/')[1].split('.')[0]) - 1 # f = 'CALTECH256/001.ak47/001_0001.jpg', label = 1 - 1 = 0
                labels.append(label)

        images = np.asarray(images)
        labels = np.asarray(labels)
        return images, labels

    def convertToTFRecords(self, images, labels, filename):
        writer = tf.python_io.TFRecordWriter(os.path.join('CALTECH256', filename))
        for i in xrange(images.shape[0]):
            image_raw = images[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
            }))
            writer.write(example.SerializeToString())
        writer.close()

    def readFromTFRecords(self, filename, batch_size, num_epochs, img_shape, num_threads, min_after_dequeue=10000):

        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)

        def read_and_decode(filename_queue, img_shape):
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(
                    serialized_example,
                    features={
                        'label': tf.FixedLenFeature([], tf.int64),
                        'image_raw': tf.FixedLenFeature([], tf.string)
                    }
            )
            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image = tf.reshape(image, img_shape)    # THIS IS IMPORTANT
            image = tf.cast(image, tf.float32) 
            
            sparse_label = features['label']        # tf.int64
            return image, sparse_label

        image, sparse_label = read_and_decode(filename_queue, img_shape) # share filename_queue with multiple threads

        # tf.train.shuffle_batch internally uses a RandomShuffleQueue
        images, sparse_labels = tf.train.shuffle_batch(
                [image, sparse_label], batch_size=batch_size, num_threads=num_threads,
                min_after_dequeue=min_after_dequeue,
                capacity=min_after_dequeue + (num_threads + 1) * batch_size
        )
        return images, sparse_labels    

