import os
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
slim = tf.contrib.slim


def upscale2d(x):
    shape = x.getshape().as_list()
    _, h, w, _ = shape
    x = tf.image.resize_nearest_neighbor(x, (h*2, w*2))
    return x

def data_loader():

    # dataset_name = os.path.basename(root)
    # #x = Image
    x = []
    # #y = Label
    y = []
    return x, y

def Encoder(Inputs, is_train=True, reuse=None):
    # C16 - C32 - C64 - C128 - C256 - C512 - C512
    # 256x256 - 256x256 - 128x128 - 128x128 - 64x64 - 32x32 - 16x16
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.01)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("G", reuse=reuse) as vs:
        tl.layers.set_name_resuse(reuse)
        n = tl.layers.InputLayer(Inputs, name='in')
        #256x256
        net_c16 = tl.layers.conv2d(n, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   B_init=b_init, name='g_c1')
        net_bn1 = tl.layers.BatchNormLayer(net_c16, act=tf.identity, gamma_init=g_init, name='g_bn1')
        #256x256
        net_c32 = tl.layers.conv2d(net_bn1, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   B_init=b_init, name='g_c2')
        net_bn2 = tl.layers.BatchNormLayer(net_c32, act=tf.identity, gamma_init=g_init, name='g_bn2')
        net_pool1 = tl.layers.MaxPool2d(net_bn2,filter_size=(2, 2), name='g_pool1')
        #128x128
        net_c64 = tl.layers.conv2d(net_pool1, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   B_init=b_init, name='g_c3')
        net_bn3 = tl.layers.BatchNormLayer(net_c64, act=tf.identity, gamma_init=g_init, name='g_bn3')
        #128x128
        net_c128 = tl.layers.conv2d(net_bn3, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    B_init=b_init, name='g_c4')
        net_bn4 = tl.layers.BatchNormLayer(net_c128, act=tf.identity, gamma_init=g_init, name='g_bn4')
        net_pool2 = tl.layers.MaxPool2d(net_bn4, filter_size=(2, 2), name='g_pool2')
        #64x64
        net_c256 = tl.layers.conv2d(net_pool2, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    B_init=b_init, name='g_c5')
        net_bn5 = tl.layers.BatchNormLayer(net_c256, act=tf.identity, gamma_init=g_init, name='g_bn5')
        net_pool3 = tl.layers.MaxPool2d(net_bn5, filter_size=(2, 2), name='g_pool3')
        #32x32
        net_c512 = tl.layers.conv2d(net_pool3, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    B_init=b_init, name='g_c6')
        net_bn6 = tl.layers.BatchNormLayer(net_c512, act=tf.identity, gamma_init=g_init, name='g_bn6')
        net_pool4 = tl.layers.MaxPool2d(net_bn6, filter_size=(2, 2), name='g_pool4')
        #16x16
        net_c512 = tl.layers.conv2d(net_pool4, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    B_init=b_init, name='g_c7')
    variables = tf.contrib.framework.get_variables(vs)
    output = net_c512
    return output


def Decoder(inputs, reuse=None):
    # C512+2N - C512+2N - C256+2N - C128+2N - C64+2N - C32+2N - C16+2N  
    # x_de is a 256x256 image
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.01)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("D") as vs :
        tl.layers.set_name_resuse(reuse)
        n = tl.layers.InputLayer(inputs, name='in')
        # 16x16
        net_c1 = tl.layers.conv2d(n, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   B_init=b_init, name='d_c1')
        net_bn1 = tl.layers.BatchNormLayer(net_c1, act=tf.identity, gamma_init=g_init, name='d_bn1')
        net_upsca1 = upscale2d(net_bn1)
        #32x32
        net_c2 = tl.layers.conv2d(net_upsca1, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   B_init=b_init, name='d_c2')
        net_bn2 = tl.layers.BatchNormLayer(net_c2, act=tf.identity, gamma_init=g_init, name='d_bn2')
        net_upsca2 = upscale2d(net_bn2)
        # 64x64
        net_c3 = tl.layers.conv2d(net_upsca1, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   B_init=b_init, name='d_c3')
        net_bn3 = tl.layers.BatchNormLayer(net_c3, act=tf.identity, gamma_init=g_init, name='d_bn3')
        net_upsca3 = upscale2d(net_bn3)
        # 128x128
        net_c4 = tl.layers.conv2d(net_upsca3, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    B_init=b_init, name='d_c4')
        net_bn4 = tl.layers.BatchNormLayer(net_c4, act=tf.identity, gamma_init=g_init, name='d_bn4')
        net_upsca4 = upscale2d(net_bn4)
        # 256x256
        net_c5 = tl.layers.Conv2d(net_upsca4, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    B_init=b_init, name='d_c5')
        net_bn5 = tl.layers.BatchNormLayer(net_c5, act=tf.identity, gamma_init=g_init, name='d_bn5')
        net_upsca5 = upscale2d(net_bn5)
        # 512x512
        net_c6 = tl.layers.conv2d(net_upsca5, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    B_init=b_init, name='d_c6')
        net_bn6 = tl.layers.BatchNormLayer(net_c6, act=tf.identity, gamma_init=g_init, name='d_bn6')

        # 512x512
        net_c512 = tl.layers.conv2d(net_bn6, 3, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                                    B_init=b_init, name='d_c7')

    variables = tf.contrib.framework.get_variables(vs)
    output = net_c512
    return output, variables

def Discriminator(x):

    # Z = C512+2N

    # C512 + FC512 + 1
    return 


def model(x, y):

    # encoder

    # decoder
    return 0

def trainer():
    optimizer = tf.train.AdamOptimizer


def main():
    data = data_loader()
    


if __name__ == '__main__':
    main()