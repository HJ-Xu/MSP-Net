import tensorflow as tf
import numpy as np
import math
import sys
import os
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import *

def placeholder_inputs(batch_size, num_point, sk_point, sk):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_sk = tf.placeholder(tf.int32, shape=(batch_size, sk_point, 1))
    # pointclouds_hsk = tf.placeholder(tf.float32, shape=(batch_size, int(sk_point/(num_point/sk_point)), 3))
    pointclouds_h = tf.placeholder(tf.int32, shape=(batch_size, int(sk_point/8), 1))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, pointclouds_sk, pointclouds_h, labels_pl


def get_model(point_cloud, point_skid, high_id, higher_id, is_training, bn_decay=None):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    k = 20

    point,sk = get_f(point_cloud,k=4,s=True)

    # h_p = tf.squeeze(find_xyz(sk,high_id))
    # adj_matrix = distance_matrix(h_p,sk)
    # _,_idx = knn(adj_matrix, k, sort=True)
    # _idx = tf.gather(_idx,np.arange(0,k-1),axis=2)

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point, is_training, bn_decay, K=3)
    point_transformed = tf.matmul(point_cloud, transform)

    input_point,skp = get_f(point_transformed,k=k,s=True)
    net_1 = conv(input_point, [1,1], mlp = [64], is_training=is_training, scope='net_1convl', bn_decay=bn_decay)
    net_1l = tf.reduce_max(net_1, axis=[2], keep_dims=True, name='net_1_maxpool')
    net_1 = conv(net_1, [1,1], mlp = [196], is_training=is_training, scope='net_1_conv', bn_decay=bn_decay)
    net_1 = tf.reduce_max(net_1, axis=[2], keep_dims=True, name='net_1_maxpool1')
    net_1 = conv(net_1, [1,1], mlp = [896], is_training=is_training, scope='net_1_conv1', bn_decay=bn_decay)
    net_1 = tf_util.max_pool2d(net_1, [net_1.get_shape()[1],1],stride=[1,1],padding='VALID', scope='net1_maxpool')

    input_point,skp = get_f(net_1l,skp,point_skid,k=k)
    net_2 = conv(input_point, [1,1], mlp = [196], is_training=is_training, scope='net_2conv', bn_decay=bn_decay)
    net_2l = tf.reduce_max(net_2, axis=[2], keep_dims=True, name='net_2_maxpool')
    net_2 = conv(net_2l, [1,1], mlp = [896], is_training=is_training, scope='net_2_conv', bn_decay=bn_decay)
    net_2 = tf_util.max_pool2d(net_2, [net_2.get_shape()[1],1],stride=[1,1],padding='VALID', scope='net2_maxpool')

    input_point,skp = get_f(net_2l,skp,high_id,k=k)
    net_3 = conv(input_point, [1,1], mlp = [896], is_training=is_training, scope='net_3conv', bn_decay=bn_decay)
    # net_3l = tf.reduce_max(net_3, axis=[2], keep_dims=True, name='net_3_maxpool')
    # net_3 = conv(net_3, [1,1], mlp = [64,128], is_training=is_training, scope='net_3_conv', bn_decay=bn_decay)
    net_3 = tf_util.max_pool2d(net_3, [net_3.get_shape()[1],net_3.get_shape()[2]],stride=[1,1],padding='VALID', scope='net3_maxpool')

    # input_point,skp = get_f(net_3l,skp,higher_id,k=k)
    # net_4 = conv(input_point, [1,1], mlp = [64], is_training=is_training, scope='net_4conv', bn_decay=bn_decay)
    # net_4l = tf.reduce_max(net_4, axis=[2], keep_dims=True, name='net_4_maxpool')
    # net_4 = conv(net_4, [1,1], mlp = [64,196], is_training=is_training, scope='net_4_conv', bn_decay=bn_decay)
    # net_4 = tf.reduce_max(net_4, axis=[1,2], keep_dims=True, name='net_4_maxpool1')

    # net = conv(tf.concat([net_1,net_2,net_3,net_4], axis = 1), [1,1], mlp = [1024], is_training=is_training, scope='net_conv2', bn_decay=bn_decay, concat = False, concatdata=net_2)
    # net = tf_util.max_pool2d(net, [net.get_shape()[1],1],stride=[1,1],padding='VALID', scope='net_maxpool')
    net = tf.concat([net_1,net_2,net_3], axis = -1)

    net = tf.reshape(net, [batch_size, -1]) 
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net,end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    return classify_loss


if __name__=='__main__':
    batch_size = 2
    num_pt = 124
    pos_dim = 3

    input_feed = np.random.rand(batch_size, num_pt, pos_dim)
    label_feed = np.random.rand(batch_size)
    label_feed[label_feed>=0.5] = 1
    label_feed[label_feed<0.5] = 0
    label_feed = label_feed.astype(np.int32)

    # # np.save('./debug/input_feed.npy', input_feed)
    # input_feed = np.load('./debug/input_feed.npy')
    # print input_feed

    with tf.Graph().as_default():
      input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
      pos, ftr = get_model(input_pl, tf.constant(True))
      # loss = get_loss(logits, label_pl, None)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {input_pl: input_feed, label_pl: label_feed}
        res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
        print(res1.shape)
        print(res1)

        print(res2.shape)
        print(res2)