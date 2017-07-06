import os
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
slim = tf.contrib.slim


def data_loader(root):

    #添加带标签数据,制作 tfdrecorder
    dataset_name = os.path.basename(root)
    #x = Image
    x = []
    #y = Label
    y = []

    return x,y

def Encoder(Inputs, is_train=True, reuse=None):
    # C16 - C32 - C64 - C128 - C256 - C512 - C512
    # 256x256 - 256x256 - 128x128 - 128x128 - 64x64 - 32x32 - 16x16
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.01)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("G", reuse=reuse) as vs:
        tl.layers.set_name_resuse(reuse)
        n = tl.layers.InputLayer(Inputs, name = 'in')
        net_c16 = tl.layers.conv2d(n, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   B_init=b_init, name='g_1c')
        net_c32 = tl.layers.conv2d(net_c16, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   B_init=b_init, name='g_2c')
        net_c64 = tl.layers.conv2d(net_c32, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   B_init=b_init, name='g_3c')
        net_c128 = tl.layers.conv2d(net_c64, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    B_init=b_init, name='g_4c')
        net_c256 = tl.layers.conv2d(net_c128, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    B_init=b_init, name='g_5c')
        net_c512 = tl.layers.conv2d(net_c256, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    B_init=b_init, name='g_6c')
        net_c512 = tl.layers.conv2d(net_c512, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAMW', W_init=w_init, B_init = b_init, name = 'g_7c')

    # z is 112 tensor
    z = []
    return z

def Decoder():
    # C512+2N - C512+2N - C256+2N - C128+2N - C64+2N - C32+2N - C16+2N  
    # x_de is a 256x256 image
    with tf.variable_scope("D") as vs :

        x_d = []
        return x_d

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