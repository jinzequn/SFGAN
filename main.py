import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np

epoch = 50000
batch_size = 1
learning_rate = 10e-5
save_step = 500


def upscale2d(x):
    x = x.outputs
    size = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, (size[1]*2, size[2]*2))
    x = InputLayer(x, name='upscale_inputs')
    tl.layers.set_name_reuse(True)
    return x


def data_loader(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.float32)
    label = tf.reshape(label, [1])
    label = tf.cast(label, tf.float32)
    return img, label


def Encoder(Inputs, is_train=True, reuse=None):

    '''

    Input a 256x256 image and output a 16x16 tensor

    '''
    # C16 - C32 - C64 - C128 - C256 - C512 - C512
    # 256x256 - 256x256 - 128x128 - 128x128 - 64x64 - 32x32 - 16x16
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.01)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("G", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(Inputs, name='in_E')
        #256x256
        net_c16 = Conv2d(n, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   b_init=b_init, name='g_c1')
        net_bn1 = BatchNormLayer(net_c16, act=tf.nn.relu, gamma_init=g_init, is_train=is_train, name='g_bn1')
        #256x256
        net_c32 = Conv2d(net_bn1, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   b_init=b_init, name='g_c2')
        net_bn2 = BatchNormLayer(net_c32, act=tf.identity, gamma_init=g_init, is_train=is_train, name='g_bn2')
        net_pool1 = MaxPool2d(net_bn2,filter_size=(2, 2), name='g_pool1')
        #128x128
        net_c64 = Conv2d(net_pool1, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   b_init=b_init, name='g_c3')
        net_bn3 = BatchNormLayer(net_c64, act=tf.identity, gamma_init=g_init, is_train=is_train, name='g_bn3')
        #128x128
        net_c128 = Conv2d(net_bn3, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    b_init=b_init, name='g_c4')
        net_bn4 = BatchNormLayer(net_c128, act=tf.identity, gamma_init=g_init, is_train=is_train, name='g_bn4')
        net_pool2 = MaxPool2d(net_bn4, filter_size=(2, 2), name='g_pool2')
        #64x64
        net_c256 = Conv2d(net_pool2, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    b_init=b_init, name='g_c5')
        net_bn5 = BatchNormLayer(net_c256, act=tf.identity, gamma_init=g_init, is_train=is_train, name='g_bn5')
        net_pool3 = MaxPool2d(net_bn5, filter_size=(2, 2), name='g_pool3')
        #32x32
        net_c512 = Conv2d(net_pool3, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    b_init=b_init, name='g_c6')
        net_bn6 = BatchNormLayer(net_c512, act=tf.identity, gamma_init=g_init, is_train=is_train, name='g_bn6')
        net_pool4 = MaxPool2d(net_bn6, filter_size=(2, 2), name='g_pool4')
        #16x16
        net_c512 = Conv2d(net_pool4, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    b_init=b_init, name='g_c7')
    # variables = tf.contrib.framework.get_variables(vs)
    logits = net_c512.outputs
    output = net_c512

    return output, logits


def Decoder(inputs, is_train=True, reuse=None):

    '''

    input a 16x16 tensor and output a 256x256 image

    '''
    # C512+2N - C512+2N - C256+2N - C128+2N - C64+2N - C32+2N - C16+2N  
    # x_de is a 256x256 image
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.01)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("D", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(inputs, name='in_D')
        # 16x16
        net_c1 = Conv2d(n, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   b_init=b_init, name='d_c1')
        net_bn1 = BatchNormLayer(net_c1, act=tf.identity, gamma_init=g_init, is_train=is_train, name='d_bn1')
        net_upsca1 = upscale2d(net_bn1)
        #32x32
        net_c2 = Conv2d(net_upsca1, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   b_init=b_init, name='d_c2')
        net_bn2 = BatchNormLayer(net_c2, act=tf.identity, gamma_init=g_init, is_train=is_train, name='d_bn2')
        net_upsca2 = upscale2d(net_bn2)
        # 64x64
        net_c3 = Conv2d(net_upsca2, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                   b_init=b_init, name='d_c3')
        net_bn3 = BatchNormLayer(net_c3, act=tf.identity, gamma_init=g_init, is_train=is_train, name='d_bn3')
        net_upsca3 = upscale2d(net_bn3)
        # 128x128
        net_c4 = Conv2d(net_upsca3, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    b_init=b_init, name='d_c4')
        net_bn4 = BatchNormLayer(net_c4, act=tf.identity, gamma_init=g_init, is_train=is_train, name='d_bn4')
        # 128x128
        net_c5 = Conv2d(net_bn4, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    b_init=b_init, name='d_c5')
        net_bn5 = BatchNormLayer(net_c5, act=tf.identity, gamma_init=g_init, is_train=is_train, name='d_bn5')
        net_upsca4 = upscale2d(net_bn5)
        # 256x256
        net_c6 = Conv2d(net_upsca4, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init,
                                    b_init=b_init, name='d_c6')
        net_bn6 = BatchNormLayer(net_c6, act=tf.identity, gamma_init=g_init, is_train=is_train, name='d_bn6')

        # 256x256
        net_c512 = Conv2d(net_bn6, 3, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init,
                                    b_init=b_init, name='d_c7')

    # variables = tf.contrib.framework.get_variables(vs)
    logits = net_c512.outputs
    output = net_c512
    return output, logits


def Discriminator(inputs, reuse=False):
    # Z = C512+2N
    # C512 + FC512 + FC112 + 1
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.01)

    with tf.variable_scope("Di", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(inputs, name='input_layer')
        net = FlattenLayer(n, name='flatten')
        net_1 = DenseLayer(net, n_units=512, act=tf.identity, W_init=w_init, b_init=b_init, name='dense_layer1')
        net_2 = DenseLayer(net_1, n_units=112, act=tf.identity, W_init=w_init, b_init=b_init, name="dense_layer2")
        net_3 = DenseLayer(net_2, n_units=1, act=tf.identity, W_init=w_init, b_init=b_init, name='dense_layer3')

    # variables = tf.contrib.framework.get_variables(vs)
    logits = net_3.outputs
    output = net_3
    return output, logits


def main():
    iter_counter = 0
    real_image_data, label_data = data_loader('train.tfrecords')
    img_batch, label_batch = tf.train.shuffle_batch([real_image_data, label_data],
                                                    batch_size=batch_size,
                                                    capacity=200,
                                                    min_after_dequeue=100
                                                    )
    print("img_batch   : %s" % img_batch._shape)
    print("label_batch : %s" % label_batch._shape)

    #==============================MODEL=====================================

    Enc_z_fake, Enc_logits_fake = Encoder(img_batch, is_train=True, reuse=False)
    Enc_z_real, Enc_logits_real = Encoder(img_batch, is_train=False, reuse=True)

    Dis_z_fake, Dis_logits_fake = Discriminator(Enc_logits_fake, reuse=False)
    Dis_z_real, Dis_logits_real = Discriminator(Enc_logits_real, reuse=True)

    Dec_z, Dec_logits = Decoder(Enc_logits_fake, is_train=True, reuse=False)

    Dis_loss_fake = tl.cost.sigmoid_cross_entropy(Dis_logits_fake, label_batch, name='discriminator_loss_fake')
    Dis_loss_real = tl.cost.sigmoid_cross_entropy(Dis_logits_real, label_batch, name='discriminator_loss_real')
    Dis_loss = Dis_loss_fake+Dis_loss_real

    Dec_loss = tl.cost.mean_squared_error(Dec_logits, img_batch)

    Dec_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(Dec_loss)
    Dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(Dis_loss)

    # ===============================TRAIN=====================================================

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # start_time = time.time()
        for i in range(epoch):
            print('epoch %d' % i)
            errDis, _ = sess.run([Dis_loss, Dis_optimizer])
            for _ in range(2):
                errDec, _ = sess.run([Dec_loss, Dec_optimizer])
            print('errDis:', errDis)
            print('errDec:', errDec)

            iter_counter += 1
            if np.mod(iter_counter, save_step) == 0:
                img = sess.run([Dec_z])
                img = img[0]
                tl.visualize.save_images(img, [1,1], './train_{:02d}_{:04d}.png'.format(epoch,i))
            
            if np.mod(iter_counter, save_step) == 0:
                tl.files.save_npz(Enc_z_fake.all_params, name='Enc_model.npz')
                tl.files.save_npz(Dec_z.all_params, name='Dec_model.npz')
                print ('model saved!')
                
        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    main()
