# Deep Learning with TensorFlow
* Giancarlo Zaccone, Md. Rezaul Karim
* Packt Publishing (2018)
* https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-Second-Edition

## 2:
* TensorFlow does everything on the C++ engine, which
  means not even a little multiplication or addition is executed in Python.
  Python is just a wrapper.

* Variables: Used to contain values for the weights and biases between TensorFlow sessions.
  * are TensorFlow objects used to hold and update parameters
* Tensors: A set of values that pass between nodes to perform operations (aka. op).
* Placeholders: Used to send data between the program and the TensorFlow graph.
  * Shape of the placeholder is the shape of the tensor to be fed (optional). If the shape is not specified, you can feed a tensor of any shape.
* Session: When a session is started, TensorFlow automatically calculates gradients for all the operations in the graph and uses them in a chain rule. In fact, a session is invoked when the graph is to be executed.
* rank:
unit of dimensionality called rank describes each tensor. It identifies the number of dimensions of the tensor. For this reason, a rank is known as order or n–dimensions of a tensor. A rank zero tensor is a scalar, a rank one tensor is a vector, and a rank two tensor is a matrix.
* tf.global_variables_initializer() is used to initialize all the ops we created before. If you need to create a variable with an initial value dependent on another variable, use the other variable's initialized_value(). This ensures that variables are initialized in the right order.

* Feeding using feed_dict argument is the least efficient way to feed data into a TensorFlow execution graph and should only be used for small experiments needing small dataset

* MSE: `loss = tf.reduce_mean(tf.square(y - y_data))`
* tensorboard --logdir path/to/logs

* RMSPropOptimizer is an advanced form of gradient descent that divides the learning rate by an exponentially decaying average of squared gradients. The suggested setting value of the decay parameter is 0.9, while a good default value for the learning rate is 0.001

* The AdamOptimizer offers several advantages over the simple GradientDescentOptimizer. In fact, it uses a larger effective step size, and the algorithm will converge to this step size without fine-tuning:

## 3:
* even though we do not care about the backward pass, TensorFlow automatically creates derivatives for
all the operations in a top to bottom fashion. When we start a session, TensorFlow automatically calculates gradients for all the deferential operations in the graph and uses them in the chain rule

* the forward pass is what we create, but TensorFlow automatically creates a backward pass, which makes the training process run by transferring data when doing the chain rule.

* no matter what the input was, if all the weights are the same then all the units in the hidden layer will be the same too. To get rid of this issue, one of the most common initialization techniques in training FNNs is the random initialization. The idea of using the random initialization is just to sample each weight from a normal distribution of the input dataset, with low deviation

* The good thing when using Xavier initialization is that it makes sure the weights are "just right," keeping the signal in a reasonable range of values through many layers. In summary, it can automatically determine the scale of initialization based on the number of input and output neurons.

* When it comes to initializing the biases, it is possible, and common, to initialize the biases to be zero since the asymmetry breaking is provided by the small random numbers in the weights.

### Activation functions
* A sigmoid function is a bounded differentiable real function that is defined for all real input values and has a non-negative derivative at each point. In general, a sigmoid function is real-valued, monotonic, and differentiable, having a non-negative first derivative, which is bell-shaped.

* Tanh squashes a real-valued number to the range [-1, 1]. Like the sigmoid neuron, its activations saturate, but unlike the sigmoid neuron, its output is zero-centered. Therefore, in practice, the tanh non-linearity is always preferred over the sigmoid nonlinearity

* softmax function is a generalization of the logistic function, that "squashes" a K-dimensional vector of arbitrary real values to a K-dimensional vector  of real values in the range [0, 1] that add up to 1:

* All activation ops apply component-wise and produce a tensor of the same shape as the input tensor.

* tf.nn.softmax(logits, axis=None, name=None)
  * why using logit(p)=log(p/(1-p))? why not, log(p)

### Softmax classifier
* `cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y_)`
  * `cost_op = tf.reduce_mean(cross_entropy)*100`
  * Ylogits = tf.matmul(Y4, W5) + B5 # computing the logits
  * Ypred = tf.nn.softmax(Ylogits)# output from layer 5

<!--
Xavier Glorot , Yoshua Bengio, Understanding the difficulty of training deep feedforward neural networks, Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS) 2010
 -->
