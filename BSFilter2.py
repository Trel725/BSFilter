from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.constraints import Constraint


class ZeroOne(Constraint):
    """Constrains the weights to lie in range [0, 1].
    """

    def __call__(self, w):
        w = K.clip(w, 0.0, 1.0)
        return w


class BSFilter(Layer):
    '''
    Binary stochastic filter for feature selection problem
    '''

    def __init__(self, regularizer=None, initializer=0.5, share_axis=None, threshold=0.1, **kwargs):
        '''
        regularizer: regularizer to use, l1 is recommended

        initialized: constant value to initialize the weights, not the class instance

        share_axis: axis, along which filtering coefficients will be shared.
                    it is mainly useful e.g. to force network select same features for 
                    every channel.

        threshold: used at prediction phase, features with weight lower than 
                   threshold are determinstically dropped, with higher values
                   are passed
        '''
        self.regularizer = regularizers.get(regularizer)
        self.initializer = initializer
        self.axis = share_axis
        self.threshold = threshold
        super(BSFilter, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable representing parameters of Bernoulli distribution
        # for each input variable

        # if axis is None the shape of kernel equal to the input shape
        if self.axis is None:
            shape = input_shape[1:]

        # else the weights will share axis
        else:
            shape = [1] * len(input_shape)
            shape[self.axis] = input_shape[self.axis]
            shape = shape[1:]
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      regularizer=self.regularizer,
                                      initializer=Constant(self.initializer),
                                      constraint=ZeroOne(),
                                      trainable=True)

        super(BSFilter, self).build(input_shape)

    # core function, implementing the filtering
    @tf.custom_gradient
    def bernoulli_pass(self, x, kernel):
        batch_size = tf.shape(x)[0]
        shape = tf.concat([(batch_size,), tf.shape(kernel)], 0)
        # a binary tensor, such that probability of one
        # is equal to the weights of the layer
        R = tf.math.ceil(
            kernel - tf.random.uniform(shape))

        # grad through the layer is simply R
        # grad w.r.t. to weights is X*weights
        def grad(dy):
            res = tf.math.reduce_mean(dy * kernel * x, axis=0)
            if self.axis is not None:
                return R * dy, tf.math.reduce_mean(res, axis=self.axis, keepdims=True)
            else:
                return R * dy, res

        return x * R, grad

    # define different beahviour at training and prediction phases
    def call(self, x, training=None):
        if training:
            return self.bernoulli_pass(x, self.kernel)
        else:
            R = self.kernel > self.threshold
            return x * tf.cast(R, tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape
