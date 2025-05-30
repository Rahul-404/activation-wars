import tensorflow as tf

#  Skip softmax in hidden layers. It’s for the output layer only and will break training if used internally.

activation_configs = {
    'sigmoid': 'sigmoid',
    'tanh': 'tanh',
    'relu': 'relu',
    'leaky_relu': tf.keras.layers.LeakyReLU(),
    'prelu': tf.keras.layers.PReLU(),
    'elu': 'elu',
    'swish': tf.keras.activations.swish,
}
