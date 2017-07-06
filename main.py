import os
import tensorflow as tf
import tensorlayer as tl
slim = tf.contrib.slim


def data_loader(root):

    #添加带标签数据,制作 tfdrecorder
    dataset_name = os.path.basename(root)
    #x = Image
    x = []
    #y = Label
    y = []

    return x,y

def Encoder(x):
    # C16 - C32 - C64 - C128 - C256 - C512 - C512
    # x is 256x256 tensor
    with tf.variable_scope("G") as vs:
        z = tf.nn.conv2d()
        #relu
        net = tf.nn.conv2d()
        net = tf.nn.conv2d()
        net = tf.nn.conv2d()
        net = tf.nn.conv2d()
        net = tf.nn.conv2d()
        net = tf.nn.conv2d() 

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