import tensorflow as tf
import numpy as np

from layers import Oper1D

np.random.seed(10)
tf.random.set_seed(10)

### SLR-OL
def SLRol(n_bands, q):
  input = tf.keras.Input((n_bands, 1), name='input')
  x_0 = Oper1D(n_bands, 3, activation = 'tanh', q = q)(input)
  y = tf.matmul(x_0, input)

  model = tf.keras.models.Model(input, y, name='OSEN')

  optimizer = tf.keras.optimizers.Adam(lr=0.001)

  model.compile(optimizer = optimizer, loss = 'mse')

  model.summary()

  return model