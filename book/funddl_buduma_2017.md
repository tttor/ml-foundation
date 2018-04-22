# Fundamentals of Deep Learning
* Nikhil Buduma and Nicholas Lacascio
* O’Reilly Media, Inc.,
* 2017-05-25: First Release
* https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book

## 3: Implementing Neural Networks in TensorFlow
* allows users to express arbitrary computation as a graph of data flows
* Nodes
  * represent mathematical operations,
* edges
  * represent data that is communicated from one node to another.
* Data in TensorFlow is represented as tensors, which are multidimensional array
  * allows us to take advantage of the speedups afforded by modern hardware
    (i.e., GPU acceleration of parallel tensor operations) and
  * provides us with a clean, but expressive, method for implementing models.

### Creating and Manipulating TensorFlow Variables
* TensorFlow variables are
  * in-memory buffers that contain tensors;
  * variables survive across multiple executions of a graph.
    * unlike normal tensors that are only instantiated when a graph is run and
      that are immediately wiped clean afterward,

### Placeholder Tensors
* `x = tf.placeholder(tf.float32, name="x", shape=[None, 784])`
  * We also notice that x has an undefined number of rows.
    * means that x can be initialized with an arbitrary number of data samples.

### Sessions in TensorFlow
* A TensorFlow program interacts with a computation graph using a session.
* The TensorFlow session is responsible for
  * building the initial graph, and
  * can be used to initialize all variables appropriately and
  * to run the computational graph

### Specifying the Logistic Regression Model in TensorFlow

## 9: Deep Reinforcement Learning
* In reinforcement learning, we don’t have a true label, only reward signals.
  * However, we can still use SGD to optimize our weights using something called policy gradients.
  * We can use the actions the agent takes, and the returns associated with those actions,
    to encourage our model weights to take good actions that lead to high reward, and to
    avoid bad ones that lead to low reward.
  * scale our loss by the value of our return, so
    if the model chose an action that led to negative return, this would lead to greater loss
