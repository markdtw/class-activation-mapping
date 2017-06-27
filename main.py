from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import argparse

import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from vgg import VGG16_GAP
from utils import CALTECH256_loader

def test(args):
    
    if args.imgpath is None: raise SystemExit('imgpath is None')
    image = scipy.misc.imread(args.imgpath, mode='RGB')
    image = scipy.misc.imresize(image, (224, 224))
    image = np.expand_dims(image, 0)
    assert image.shape == (1, 224, 224, 3)

    image_ph = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))

    model = VGG16_GAP(train_mode=False)
    model.build(image_ph)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    if args.modelpath is not None:
        print ('Using model: {}'.format(args.modelpath))
        saver.restore(sess, args.modelpath)
    else:
        print ('Using pretrained caffemodel')

    feed_dict = {image_ph: image}
    logits, CAM_conv_resize, CAM_fc = sess.run([
        model.cam_fc, model.cam_conv_resize, model.cam_fc_value], feed_dict=feed_dict)
    pred = np.argmax(logits)
    pdb.set_trace()

    CAM_heatmap = np.matmul(CAM_conv_resize.reshape(-1, 1024), CAM_fc.transpose()[pred])
    CAM_heatmap = np.reshape(CAM_heatmap, [224, 224])
    
    fig = plt.figure()
    plt.imshow(image.squeeze())
    plt.imshow(CAM_heatmap, cmap=plt.cm.jet, alpha=0.5, interpolation='bilinear')
    #plt.show()
    fig.savefig(os.path.join('result', args.imgpath.split('/')[-1]))

    sess.close()
    print ('Bye')

def train(args):

    queue_loader = CALTECH256_loader(batch_size=args.bsize, num_epochs=args.ep)

    model = VGG16_GAP(args.lr, train_mode=True)
    model.build(queue_loader.images)
    model.loss(queue_loader.labels)
    train_op = model.train()

    tf.summary.scalar('cross entropy loss', model.xen_loss_op)
    tf.summary.scalar('regularization loss', model.reg_loss_op)
    tf.summary.scalar('total loss', model.loss_op)
    merged_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    writer = tf.summary.FileWriter('log', sess.graph)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    if args.modelpath is not None:
        print ('Using model: {}'.format(args.modelpath))
        saver.restore(sess, args.modelpath)

    print ('Start training')
    print ('batch size: %d, epoch: %d, initial learning rate: %.3f' % (args.bsize, args.ep, args.lr))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        ep = 0
        step = 1
        correct_all = 0
        while not coord.should_stop():
            logits, xloss, rloss, loss, correct, _, summary = sess.run([
                model.cam_fc, model.xen_loss_op, model.reg_loss_op, model.loss_op, 
                model.correct_op, train_op, merged_op
            ])
            writer.add_summary(summary, ep * queue_loader.num_batches + step)
            correct_all += correct.sum()
            if step % 40 == 0:
                print ('epoch: %2d, step: %3d, xloss: %.2f, rloss: %.2f, loss: %.3f' % (ep+1, step, xloss, rloss, loss))

            if step % queue_loader.num_batches == 0:
                print ('epoch: %2d, step: %3d, xloss: %.2f, rloss: %.2f, loss: %.3f, epoch %2d done.' %
                        (ep+1, step, xloss, rloss, loss, ep+1))
                print ('EPOCH %2d ACCURACY: %.2f%%.' % (ep+1, correct_all * 100 / 30607))
                checkpoint_path = os.path.join('log', 'vgg16CAM_cal256')
                saver.save(sess, checkpoint_path, global_step=ep+1)
                ep += 1
                step = 1
                correct_all = 0
            else:
                step += 1
    except tf.errors.OutOfRangeError:
        print ('\nDone training, epoch limit: %d reached.' % (args.ep))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    print ('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='set this to train.')
    parser.add_argument('--test', action='store_true', help='set this to test.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=180, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int, default=64, help='batch size.')
    parser.add_argument('--modelpath', metavar='', type=str, default=None, help='trained tensorflow model path.')
    parser.add_argument('--imgpath', metavar='', type=str, default=None, help='Test image path.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: raise SystemExit('Unknown argument: {}'.format(unparsed))
    if args.train:
        train(args)
    if args.test:
        test(args)
    if not args.train and not args.test:
        parser.print_help()
