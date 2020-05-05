
from layers import *

class MGICNN:
    def __init__(self, placeholders):
        self.placeholders = placeholders

    def _GFE(self, x1, x2, x3):
        layers = []
        out = conv2d_layer(inputs=x1, filters=64, kernel_size=3, strides=1, activation=None)
        layers.append(out)
        out = leaky_relu(out)
        out = conv2d_layer(inputs=out, filters=64, kernel_size=3, strides=1, activation=None)
        layers.append(out)
        out = leaky_relu(out)

        out = conv2d_layer(inputs=tf.concat([x2, out], axis=-1), filters=128, kernel_size=3, strides=1, activation=None)
        layers.append(out)
        out = leaky_relu(out)
        out = conv2d_layer(inputs=out, filters=128, kernel_size=3, strides=1, activation=None)
        layers.append(out)
        out = leaky_relu(out)

        out = conv2d_layer(inputs=tf.concat([x3, out], axis=-1), filters=192, kernel_size=3, strides=1, activation=None)
        layers.append(out)
        out = leaky_relu(out)
        out = conv2d_layer(inputs=out, filters=192, kernel_size=3, strides=1, activation=None)
        layers.append(out)
        out = leaky_relu(out)

        out = max_pool(inputs=out, pool_size=3, strides=2)

        out = conv2d_layer(inputs=out, filters=256, kernel_size=3, strides=1)
        out = conv2d_layer(inputs=out, filters=256, kernel_size=3, strides=1)

        return out, layers

    def _out_layer(self, inputs):
        out = max_pool(inputs=inputs, pool_size=3, strides=2)

        out = conv2d_layer(inputs=out, filters=512, kernel_size=3, strides=1)
        out = conv2d_layer(inputs=out, filters=512, kernel_size=3, strides=1)

        out = tf.layers.flatten(out)
        out = dense_layer(inputs=out, filters=1024, do=0.5)
        out = dense_layer(inputs=out, filters=1024, do=0.5)

        logit = dense_layer(inputs=out, filters=1, activation=None)

        pred_sig = tf.nn.sigmoid(logit)        # 输出概率值
        prediction = tf.cast(tf.round(pred_sig), tf.int64)[..., 0]   # 大于0.5的会被认为是正例

        return logit, pred_sig, prediction

    def build_proposed(self):
        zoom_in, self.zi_layers = self._GFE(x1=self.placeholders["bdat"],
                            x2=self.placeholders["mdat"],
                            x3=self.placeholders["tdat"])
        zoom_out, self.zo_layers = self._GFE(x1=self.placeholders["tdat"],
                             x2=self.placeholders["mdat"],
                             x3=self.placeholders["bdat"])
        ######### Proposed ###################
        if st.model_mode ==0:
            if st.multistream_mode==0:
                out = zoom_in + zoom_out
            elif st.multistream_mode == 1:
                out = tf.concat([zoom_in, zoom_out], axis=-1)
            else:
                zoom_in_ = conv2d_layer(inputs=zoom_in, filters=zoom_in.shape[-1], kernel_size=1, strides=1, bn=True)
                zoom_out_ = conv2d_layer(inputs=zoom_out, filters=zoom_out.shape[-1], kernel_size=1, strides=1, bn=True)
                out = zoom_in_ + zoom_out_
        ############ RI and LR #################
        elif st.model_mode == 1 or st.model_mode==2:
            if st.model_mode==1:
                out = tf.concat([self.placeholders["bdat"], self.placeholders["mdat"], self.placeholders["tdat"]], axis=-1)
                out = conv2d_layer(inputs=out, filters=64, kernel_size=3, strides=1)
                out = conv2d_layer(inputs=out, filters=64, kernel_size=3, strides=1)

            elif st.model_mode==2:
                out1 = conv2d_layer(inputs=self.placeholders["bdat"], filters=64, kernel_size=1, strides=1)
                out1 = conv2d_layer(inputs=out1, filters=64, kernel_size=1, strides=1)
                out2 = conv2d_layer(inputs=self.placeholders["mdat"], filters=64, kernel_size=1, strides=1)
                out2 = conv2d_layer(inputs=out2, filters=64, kernel_size=1, strides=1)
                out3 = conv2d_layer(inputs=self.placeholders["tdat"], filters=64, kernel_size=1, strides=1)
                out3 = conv2d_layer(inputs=out3, filters=64, kernel_size=1, strides=1)

                out = tf.concat([out1, out2, out3], axis=-1)

            out = conv2d_layer(inputs=out, filters=128, kernel_size=3, strides=1)
            out = conv2d_layer(inputs=out, filters=128, kernel_size=3, strides=1)

            out = conv2d_layer(inputs=out, filters=192, kernel_size=3, strides=1)
            out = conv2d_layer(inputs=out, filters=192, kernel_size=3, strides=1)

            out = max_pool(inputs=out, pool_size=3, strides=2)

            out = conv2d_layer(inputs=out, filters=256, kernel_size=3, strides=1)
            out = conv2d_layer(inputs=out, filters=256, kernel_size=3, strides=1)

        elif st.model_mode==3:
            out = zoom_in
        elif st.model_mode == 4:
            out = zoom_out

        logit, self.pred_sig, self.prediction = self._out_layer(inputs=out)

        self.loss = tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=self.placeholders["lbl"],
                                               logits=logit)
        if st.train:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optim = tf.train.AdamOptimizer(learning_rate=st.lr, beta1=st.beta1, beta2=st.beta2).minimize(
                    self.loss)

        self.acc = tf.count_nonzero(tf.equal(self.prediction, self.placeholders["lbl"][...,0]))

        self.summary_op = tf.compat.v1.summary.merge([
            tf.compat.v1.summary.scalar("loss", self.loss),
            tf.compat.v1.summary.scalar("acc", self.acc),
        ])