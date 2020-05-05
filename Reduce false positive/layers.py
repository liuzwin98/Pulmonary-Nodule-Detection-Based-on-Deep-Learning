import tensorflow as tf
import settings as st

# def initializer():
#     # return tf.variance_scaling_initializer(scale=0.01, mode="fan_avg", distribution="normal")
#     # return tf.truncated_normal_initializer(stddev=0.01)
#     # return tf.random_normal_initializer(mean=0.0, stddev=0.01)
#     return lambda:tf.truncated_normal_initializer(mean=0.0, stddev=0.001)

initializer = lambda: tf.truncated_normal_initializer(mean=0.0, stddev=0.001)


def sigma_constraint(input):
    return tf.exp(input)

def dense(inputs, filters, use_bias=True, activation=None, freeze=False):
    trainable = st.train
    if freeze:
        trainable=False
    return tf.layers.dense(inputs=inputs, units=filters,
                           activation=activation,
                           use_bias=use_bias,
                           kernel_initializer=initializer(),
                           trainable=trainable)
def conv2d(inputs, filters, kernel_size=3, strides=1, use_bias=True, activation=None, freeze=False):
    trainable = st.train
    if freeze:
        trainable=False

    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            data_format='channels_last',
                            activation=activation,
                            use_bias=use_bias,
                            kernel_initializer=initializer(),
                            trainable=trainable)


def deconv2d(inputs, filters, kernel_size=3, strides=1, activation=None, use_bias=True, freeze=False):
    trainable = st.train
    if freeze:
        trainable=False

    return tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size,
                                      strides=strides,
                                      padding='same',
                                      data_format='channels_last',
                                      activation=activation,
                                      use_bias=use_bias,
                                      kernel_initializer=initializer(),
                                      trainable=trainable)


def batch_norm(inputs, freeze=False):

    if st.batch_norm:
        if freeze or not st.train:
            trainable = False
            training = False
        else:
            trainable = st.train
            training = st.is_training

        return tf.layers.batch_normalization(inputs,
                                             momentum=0.9,
                                             epsilon=1e-5,
                                             training=training,
                                             trainable=trainable
                                             )
    else:
        return inputs


def leaky_relu(inputs, alpha=st.alpha):
    return tf.nn.leaky_relu(inputs, alpha)


def dropout(inputs, rate, freeze=False):
    if st.dropout:
        train = st.is_training
        if freeze:
            train=False
        return tf.layers.dropout(inputs=inputs, rate=rate, training=train)
    else:
        return inputs


def max_pool(inputs, pool_size, strides):
    return tf.layers.max_pooling2d(inputs = inputs, pool_size=pool_size, strides=strides, padding='same')


def dense_layer(inputs, filters, bn=True, do=0, use_bias=True, activation=leaky_relu, freeze=False):
    out = inputs
    if dropout:
        out = dropout(out, rate=do, freeze=freeze)
    out = dense(inputs=out, filters=filters, use_bias=use_bias, freeze=freeze)
    if bn:
        out = batch_norm(out, freeze=freeze)

    if activation:
        out = activation(out)

    return out


def conv2d_layer(inputs, filters, kernel_size, strides, bn=True, do=0, use_bias=True, activation=leaky_relu, freeze=False):
    out = inputs
    if dropout:
        out = dropout(out, rate=do, freeze=freeze)
    out = conv2d(inputs=out, filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, freeze=freeze)
    if bn:
        out = batch_norm(out, freeze=freeze)

    if activation:
        out = activation(out)

    return out


def deconv2d_layer(inputs, filters, kernel_size, strides, bn=True, do=0, use_bias=True, activation=leaky_relu, freeze=False):
    out = inputs
    if do:
        out = dropout(out, rate=do, freeze=freeze)
    out = deconv2d(inputs=out, filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, freeze=freeze)

    if bn:
        out = batch_norm(out, freeze=freeze)

    if activation:
        out = activation(out)

    return out