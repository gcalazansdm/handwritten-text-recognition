"""
Gated implementations
    GatedConv2D: Introduce a Conv2D layer (same number of filters) to multiply with its sigmoid activation.
    FullGatedConv2D: Introduce a Conv2D to extract features (linear and sigmoid), making a full gated process.
                     This process will double number of filters to make one convolutional process.
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Conv2D, Multiply, Activation, ReLU, Lambda, Subtract, SpatialDropout2D, Dropout
from tensorflow.keras.initializers import RandomUniform, Constant
from tensorflow.keras import initializers,regularizers
import math
import numpy as np
import random 

class DoubleDropout(Layer):
    def __init__(self,rate=.2,rate_b=.2, **kwargs):
        super().__init__(**kwargs)
        self.dropout2d = SpatialDropout2D(rate_b)
        self.dropout = Dropout(rate)

    def call(self, inputs):
        if random.random() < 0.5:
            return self.dropout(inputs)
        return self.dropout2d(inputs)

class Attention(Layer):
    def __init__(self,
                 filters,
                 kernel_initializer="glorot_uniform",
                 **kwargs):

        super().__init__(filters,**kwargs)
        self.filters = filters
        
        self.kernel_size = (3,3)
        self.kernel_value_size = (1,1)
        
        self.query_kernel = self.add_weight(name="query_kernel",
                                           shape=(*self.kernel_size,1, self.filters),
                                           initializer=kernel_initializer)
                                           
        self.key_kernel = self.add_weight(name="key_kernel",
                                           shape=(*self.kernel_size,1, self.filters),
                                           initializer=kernel_initializer)
                                           
        self.value_kernel = self.add_weight(name="value_kernel",
                                           shape=(*self.kernel_value_size,1, self.filters),
                                           initializer=kernel_initializer)
        self.final_kernel = self.add_weight(name="final_kernel",
                                           shape=(*self.kernel_value_size,1, self.filters),
                                           initializer=kernel_initializer)
        self.negative = tf.constant(-1.)
        self.Bq = tf.keras.layers.BatchNormalization(renorm=True)
        self.Bk = tf.keras.layers.BatchNormalization(renorm=True)
        self.Bv = tf.keras.layers.BatchNormalization(renorm=True)

    def call(self, inputs):
        #Query
        q = K.conv2d(inputs,self.query_kernel,padding="same",data_format="channels_last")
        
        #Key
        k = K.conv2d(inputs,self.key_kernel,padding="same",data_format="channels_last")
        
        #Value
        v = K.conv2d(inputs,self.value_kernel,padding="same",data_format="channels_last")
        
        # N = h * wi
        s = tf.matmul(self.Bq(q), self.Bk(k),transpose_b=True)
        
        # attention map
        attention_map = K.softmax(s)

        s = tf.matmul(attention_map,self.Bv(v))
        
        o = K.conv2d(s,self.final_kernel,padding="same",data_format="channels_last")

        return o + inputs



class pushPull2D(Layer):
    """Gated Convolutional Class"""
    def __init__(self,
                 filters,
                 kernel_size=(3,3),
                 input_shape=(1,1,1,3),
                 strides=(1,1),
                 padding="same",
                 kernel_initializer="glorot_uniform",
                 scale=2,
                 alpha=1,
                 train_alpha=True,
                 **kwargs):
        assert alpha >= 0 and alpha <= 10

        super().__init__(filters,**kwargs)
        self.train_alpha = train_alpha
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.scale_factor = scale
        
        self.push_kernel = self.add_weight(name="push_kernel",
                                           shape=(*self.kernel_size,1, self.filters),
                                           initializer=kernel_initializer)
                                           
        self.pull_kernel = self.add_weight(name="pull_kernel",
                                           shape=(*self.kernel_size,1, self.filters),
                                           initializer=kernel_initializer)
                                           
        if self.train_alpha:
            k = filters
            r = 1. / math.sqrt(input_shape[3] * filters)
            alpha_weight = tf.Variable(tf.random_uniform_initializer(minval=0.5-r, maxval=0.5+r)(shape=[1], dtype=tf.float32))
            self.alpha = K.relu(alpha_weight)
        else:
            self.alpha = tf.Variable(alpha, dtype=np.float32, trainable=False)
        self.negative = tf.constant(-1.)

    def call(self, inputs):
        #Push
        push_first_step = K.conv2d(inputs,self.push_kernel,strides=self.strides,padding=self.padding,data_format="channels_last")
        push_final_step = K.relu(push_first_step)
        
        #Calculate pull variables
        push_size = push_first_step.shape[3]
        
        if self.scale_factor != 1:
            pull_size = math.floor(push_size * self.scale_factor)
            if pull_size % 2 == 0:
                pull_size += 1
            
        #Pull
        pull_step_zero = K.conv2d(inputs,self.pull_kernel,strides=self.strides, padding=self.padding,data_format="channels_last") 
        
        if self.scale_factor != 1:
            pull_first_step = pull_step_zero
        else:
            pull_first_step = K.UpSampling2D(size=(pull_size, pull_size),
                                      interpolation='bilinear')(pull_step_zero)
                                    
        pull_second_step = K.relu(pull_first_step)
        pull_third_step = pull_second_step * self.alpha
        pull_final_step = pull_third_step * self.negative 
        
        #Push x Pull        
        return push_final_step - pull_final_step


"""
Tensorflow Keras layer implementation of the gated convolution.
    Args:
        kwargs: Conv2D keyword arguments.
    Reference:
        T. Bluche, R. Messina,
        Gated convolutional recurrent neural networks for multilingual handwriting recognition.
        14th IAPR International Conference on Document Analysis andRecognition (ICDAR),
        p. 646–651, 11 2017.
"""


class GatedConv2D(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, **kwargs):
        super(GatedConv2D, self).__init__(**kwargs)

    def call(self, inputs):
        """Apply gated convolution"""

        output = super(GatedConv2D, self).call(inputs)
        linear = Activation("linear")(inputs)
        sigmoid = Activation("sigmoid")(output)

        return Multiply()([linear, sigmoid])

    def get_config(self):
        """Return the config of the layer"""

        config = super(GatedConv2D, self).get_config()
        return config


"""
Tensorflow Keras layer implementation of the gated convolution.
    Args:
        filters (int): Number of output filters.
        kwargs: Other Conv2D keyword arguments.
    Reference (based):
        Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier,
        Language modeling with gated convolutional networks, in
        Proc. 34th Int. Conf. Mach. Learn. (ICML), vol. 70,
        Sydney, Australia, pp. 933–941, 2017.

        A. van den Oord and N. Kalchbrenner and O. Vinyals and L. Espeholt and A. Graves and K. Kavukcuoglu
        Conditional Image Generation with PixelCNN Decoders, 2016
        NIPS'16 Proceedings of the 30th International Conference on Neural Information Processing Systems
"""


class FullGatedConv2D(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        """Apply gated convolution"""
        output = super(FullGatedConv2D, self).call(inputs)
        linear = Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, self.nb_filters:])

        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""

        output_shape = super(FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters* 2,)
    
    def get_config(self):
        """Return the config of the layer"""

        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config


"""
Tensorflow Keras layer implementation of the octave convolution.

Reference (based):
    Yunpeng Chen, Haoqi Fan, Bing Xu, Zhicheng Yan, Yannis Kalantidis, Marcus Rohrbach, Shuicheng Yan, Jiashi Feng.
    Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution.

    OctConv-TFKeras
    Github: https://github.com/koshian2/OctConv-TFKeras
"""


class OctConv2D(Layer):
    """Octave Convolutional Class"""

    def __init__(self,
                 filters,
                 alpha,
                 kernel_size=(3,3),
                 strides=(1,1),
                 padding="same",
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        assert alpha >= 0 and alpha <= 1
        assert filters > 0 and isinstance(filters, int)

        super().__init__(**kwargs)

        self.alpha = alpha
        self.filters = filters
        # optional values
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint

        # --> low channels
        self.low_channels = int(self.filters * self.alpha)
        # --> high channels
        self.high_channels = self.filters - self.low_channels

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 4 and len(input_shape[1]) == 4
        # assertion for high inputs
        assert input_shape[0][1] // 2 >= self.kernel_size[0]
        assert input_shape[0][2] // 2 >= self.kernel_size[1]
        # assertion for low inputs
        assert input_shape[0][1] // input_shape[1][1] == 2
        assert input_shape[0][2] // input_shape[1][2] == 2

        assert K.image_data_format() == "channels_last"
        high_in = int(input_shape[0][3])
        low_in = int(input_shape[1][3])

        # High -> Low
        self.high_to_high_kernel = self.add_weight(name="high_to_high_kernel",
                                                   shape=(*self.kernel_size, high_in, self.high_channels),
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.kernel_regularizer,
                                                   constraint=self.kernel_constraint)
        # High -> Low
        self.high_to_low_kernel = self.add_weight(name="high_to_low_kernel",
                                                  shape=(*self.kernel_size, high_in, self.low_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)

        # Low -> High
        self.low_to_high_kernel = self.add_weight(name="low_to_high_kernel",
                                                  shape=(*self.kernel_size, low_in, self.high_channels),
                                                  initializer=self.kernel_initializer,
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        # Low -> Low
        self.low_to_low_kernel = self.add_weight(name="low_to_low_kernel",
                                                 shape=(*self.kernel_size, low_in, self.low_channels),
                                                 initializer=self.kernel_initializer,
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)
        super().build(input_shape)

    def call(self, inputs):
        # Input=[x^H, x^L]
        assert len(inputs) == 2
        high_input, low_input = inputs
        # High -> High conv
        high_to_high = K.conv2d(high_input, self.high_to_high_kernel,
                                strides=self.strides, padding=self.padding,
                                data_format="channels_last")
        # High -> low conv
        high_to_low = K.pool2d(high_input, (2, 2), strides=(2, 2), pool_mode="avg")
        high_to_low = K.conv2d(high_to_low, self.high_to_low_kernel, strides=self.strides,
                               padding=self.padding, data_format="channels_last")

        # Low -> high conv
        low_to_high = K.conv2d(low_input, self.low_to_high_kernel,
                               strides=self.strides, padding=self.padding,
                               data_format="channels_last")
        low_to_high = K.repeat_elements(low_to_high, 2, axis=1)
        low_to_high = K.repeat_elements(low_to_high, 2, axis=2)

        # Low -> low conv
        low_to_low = K.conv2d(low_input, self.low_to_low_kernel,
                              strides=self.strides, padding=self.padding,
                              data_format="channels_last")

        # cross add
        high_add = high_to_high + low_to_high
        low_add = low_to_low + high_to_low

        return [high_add, low_add]

    def compute_output_shape(self, input_shapes):
        high_in_shape, low_in_shape = input_shapes
        high_out_shape = (*high_in_shape[:3], self.high_channels)
        low_out_shape = (*low_in_shape[:3], self.low_channels)
        return [high_out_shape, low_out_shape]

    def get_config(self):
        base_config = super().get_config()
        out_config = {
            **base_config,
            "filters": self.filters,
            "alpha": self.alpha,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
        }
        return out_config
