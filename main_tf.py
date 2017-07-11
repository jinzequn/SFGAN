import tensorflow as tf

epoch = 50000
batch_size = 1
learning_rate = 10e-5
save_step = 50

def weight_variable(shape):
    with tf.name_scope('weights'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


def bias_variable(shape):
    with tf.name_scope('biases'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def upscale2d(x):
    size = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, (size[1]*2, size[2]*2))
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


def Encoder(inputs, is_train=True, reuse=None):

    '''

    Input a 256x256 image and output a 16x16 tensor

    '''
    # C16 - C32 - C64 - C128 - C256 - C512 - C512
    # 256x256 - 256x256 - 128x128 - 128x128 - 64x64 - 32x32 - 16x16

    with tf.variable_scope("G", reuse=reuse) as vs:
        #256x256
        W_conv1 = tf.get_variable('W_conv1', shape=[3, 3, 3, 16], initializer=tf.contrib.layers.xavier_initializer())
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(inputs, W_conv1) + b_conv1)


        #256x256
        W_conv2 = tf.get_variable('W_conv2', shape=[3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        h_pool1 = max_pool_2x2(h_conv2)
        #128x128
        W_conv3 = tf.get_variable('W_conv3', shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
        #128x128
        W_conv4 = tf.get_variable('W_conv4', shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
        b_conv4 = bias_variable([128])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
        h_pool2 = max_pool_2x2(h_conv4)
        #64x64
        W_conv5 = tf.get_variable('W_conv5', shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
        b_conv5 = bias_variable([256])
        h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)
        h_pool3 = max_pool_2x2(h_conv5)
        #32x32
        W_conv6 = tf.get_variable('W_conv6', shape=[3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
        b_conv6 = bias_variable([512])
        h_conv6 = tf.nn.relu(conv2d(h_pool3, W_conv6) + b_conv6)
        h_pool4 = max_pool_2x2(h_conv6)
        #16x16
        W_conv7 = tf.get_variable('W_conv7', shape=[3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
        b_conv7 = bias_variable([512])
        h_conv7 = tf.nn.relu(conv2d(h_pool4, W_conv7) + b_conv7)

        W_fc1 = weight_variable([16 * 16 * 512, 256])
        b_fc1 = bias_variable([256])
        h_pool3_flat = tf.reshape(h_conv7, [-1, 16 * 16 * 512])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    variables = tf.contrib.framework.get_variables(vs)


    return h_fc1, variables


def Decoder(inputs, is_train=True, reuse=None):

    '''

    input a 16x16 tensor and output a 256x256 image

    '''
    # C512+2N - C512+2N - C256+2N - C128+2N - C64+2N - C32+2N - C16+2N
    # x_de is a 256x256 image

    with tf.variable_scope("D", reuse=reuse) as vs:

        W_fc1 = weight_variable([256, 16 * 16 * 512])
        b_fc1 = bias_variable([131072])
        h_pool3_flat = tf.reshape(inputs, [-1, 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        h_re = tf.reshape(h_fc1, [-1, 16, 16, 512])
        # 16x16
        W_conv8 = tf.get_variable('W_conv8', shape=[3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
        b_conv8 = bias_variable([512])
        h_conv8 = tf.nn.relu(conv2d(h_re, W_conv8) + b_conv8)
        net_upsca1 = upscale2d(h_conv8)
        #32x32
        W_conv9 = tf.get_variable('W_conv9', shape=[3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
        b_conv9 = bias_variable([512])
        h_conv9 = tf.nn.relu(conv2d(net_upsca1, W_conv9) + b_conv9)
        net_upsca2 = upscale2d(h_conv9)
        # 64x64
        W_conv10 = tf.get_variable('W_conv10', shape=[3, 3, 512, 256], initializer=tf.contrib.layers.xavier_initializer())
        b_conv10 = bias_variable([256])
        h_conv10 = tf.nn.relu(conv2d(net_upsca2, W_conv10) + b_conv10)
        net_upsca3 = upscale2d(h_conv10)
        # 128x128
        W_conv11 = tf.get_variable('W_conv11', shape=[3, 3, 256, 128], initializer=tf.contrib.layers.xavier_initializer())
        b_conv11 = bias_variable([128])
        h_conv11 = tf.nn.relu(conv2d(net_upsca3, W_conv11) + b_conv11)
        # 128x128
        W_conv12 = tf.get_variable('W_conv12', shape=[3, 3, 128, 64], initializer=tf.contrib.layers.xavier_initializer())
        b_conv12 = bias_variable([64])
        h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12) + b_conv12)
        net_upsca4 = upscale2d(h_conv12)
        # 256x256
        W_conv13 = tf.get_variable('W_conv13', shape=[3, 3, 64, 32], initializer=tf.contrib.layers.xavier_initializer())
        b_conv13 = bias_variable([32])
        h_conv13 = tf.nn.relu(conv2d(net_upsca4, W_conv13) + b_conv13)

        # 256x256
        W_conv14 = tf.get_variable('W_conv14', shape=[3, 3, 32, 16], initializer=tf.contrib.layers.xavier_initializer())
        b_conv14 = bias_variable([16])
        h_conv14 = tf.nn.relu(conv2d(h_conv13, W_conv14) + b_conv14)

        W_conv15 = tf.get_variable('W_conv15', shape=[3, 3, 16, 3], initializer=tf.contrib.layers.xavier_initializer())
        b_conv15 = bias_variable([3])
        h_conv15 = tf.nn.relu(conv2d(h_conv14, W_conv15) + b_conv15)
    variables = tf.contrib.framework.get_variables(vs)

    return h_conv15, variables


def Discriminator(inputs, reuse=False):
    # Z = C512+2N
    # C512 + FC512 + FC112 + 1

    with tf.variable_scope("Di", reuse=reuse) as vs:

        W_fc2 = weight_variable([256,   112])
        b_fc2 = bias_variable([112])
        h_fc2 = tf.nn.relu(tf.matmul(inputs, W_fc2)+b_fc2)

        W_fc3 = weight_variable([112,   1])
        b_fc3 = bias_variable([1])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3)+b_fc3)

    variables = tf.contrib.framework.get_variables(vs)

    return h_fc3, variables


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

    Enc_z_fake, Enc_val_fake = Encoder(img_batch, is_train=True, reuse=False)
    Enc_z_real, Enc_val_real = Encoder(img_batch, is_train=False, reuse=True)

    Dis_z_fake, Dis_val_fake = Discriminator(Enc_z_fake, reuse=False)
    Dis_z_real, Dis_val_real = Discriminator(Enc_z_real, reuse=True)

    Dec_z, Dec_val = Decoder(Enc_z_fake, is_train=True, reuse=False)

    Dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dis_z_fake, label_batch, name='discriminator_loss_fake'))
    Dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Dis_z_real, label_batch, name='discriminator_loss_real'))
    Dis_loss = Dis_loss_fake+Dis_loss_real

    Dec_loss = tf.reduce_mean(tf.abs(Dec_z - img_batch))

    Dec_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(Dec_loss,
                                                                                                var_list=Dec_val)
    Dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(Dis_loss,
                                                                                                var_list=Dis_val_fake)

    # ===============================TRAIN=====================================================

    with tf.Session() as sess:
        # tl.layers.initialize_global_variables(sess)
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

        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    main()
