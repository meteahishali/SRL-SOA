import tensorflow as tf

####################################################
# Operational Layers.
class Oper1D(tf.keras.Model):
  def __init__(self, filters, kernel_size, activation = None, q = 1):
    super(Oper1D, self).__init__(name='')

    self.diagonal = tf.zeros(filters)
    self.activation = activation
    self.q = q
    self.all_layers = []

    for i in range(0, q):  # q convolutional layers.
      self.all_layers.append(tf.keras.layers.Conv1D(filters,
                                                    kernel_size,
                                                    padding='same', activation=None))

  @tf.function
  def call(self, input_tensor, training=False):

    def diag_zero(input):
      x_0 = tf.linalg.set_diag(input, self.diagonal)
      return x_0

    
    x = self.all_layers[0](input_tensor)  # First convolutional layer.

    if self.q > 1:
      for i in range(1, self.q):
        x += self.all_layers[i](tf.math.pow(input_tensor, i + 1))
    
    if self.activation is not None:
      x = eval('tf.nn.' + self.activation + '(x)')

    x = tf.vectorized_map(fn=diag_zero, elems = x) # Diagonal constraint.

    x = tf.keras.layers.ActivityRegularization(l1=0.01)(x) # Sparse regularization.
    
    return x