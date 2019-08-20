from keras import backend as K
from keras.backend import tf
from keras.engine.topology import Layer
from keras import regularizers
from keras.initializers import Constant
from tensorflow.python.framework import ops
from keras.constraints import Constraint


@ops.RegisterGradient("UnityGrad")
def bernoulliSample_ST(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]


class ZeroOne(Constraint):
    """Constrains the weights to lie in range [0, 1].
    """

    def __call__(self, w):
        w = K.clip(w, 0.0, 1.0)
        return w


class BSConvFilter(Layer):
    '''
    Binary stochastic filter for kernel pruning
    '''

    def __init__(self, regularizer=None, initializer=0.5, **kwargs):
        self.regularizer = regularizers.get(regularizer)
        self.initializer = initializer
        super(BSConvFilter, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable representing parameters of Bernoulli distribution
        # for each input variable
        if len(input_shape) != 4:
            raise Exception("BSConvFilter must be placed after convolutional layer")
        self.kernel = self.add_weight(name='kernel',
                                      shape=input_shape[3:],
                                      regularizer=self.regularizer,
                                      initializer=Constant(self.initializer),
                                      constraint=ZeroOne(),  # keeps values in sensible range
                                      trainable=True)

        super(BSConvFilter, self).build(input_shape)  # parent init

    def call(self, x):
        G = K.get_session().graph
        with G.gradient_override_map({"Ceil": "Identity", "Sub": "UnityGrad"}):
            prob = self.kernel  # kernel represents probabilities
            coef = K.tf.ceil(prob - K.tf.random_uniform(K.tf.shape(prob)))  # sample from
            # uniform distribution in range [0, 1], round to greater value, i.e. to zero or one
            return K.tf.multiply(x, coef)

    def compute_output_shape(self, input_shape):
        return input_shape
