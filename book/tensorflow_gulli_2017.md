# TensorFlow 1.x Deep Learning Cookbook
* Antonio Gulli, Amita Kapoor
* December 2017
* https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-1x-deep-learning-cookbook

## ch 1: intro
* vs numpy for writing code for a simple one-layer feedforward network
* DL lib: 
  * have pretrained models, and 
  * support commonly used NN architectures like 
    recurrent neural networks (RNNs), convolutional neural networks (CNNs), and deep belief networks (DBNs)
* To execute the graph element, 
  we need to define the Session using `with` and run the session using `run`
* dividing the program into two separate parts, namely, 
  * computational graph 
    * is a network of nodes and edges
    * Nodes in the network represent Objects (tensors and Operations), and 
    * edges represent the Tensors that flow between operations
  * Execution of the grap
    * The execution of the graph is performed using Session Object
    * Session Objectencapsulates the environment in which tensor and Operation Objects are evaluated.
    * In the same program code, we can have many session objects.
* `tf.device()`: 
   to use specific devices (CPU/GPU) with different objects of the computation graph
* All types of data, that is, scalar, vectors, and matrices are special types of tensors
  * scalar: 0-D tensor
  * vectors: 1-D tensor
  * matrices: 2-D tensor
  * tensor: n-D tensor
* TensorFlow supports three types of tensors:
  * Constants
  * Variables
    * values require updating within a session
    * often used to represent weights and biases in a neural network.
  * Placeholders
    * used to feed values into a TensorFlow graph
    * to feed new training examples while training a neural network
    * do not contain any data and thus there is no need to initialize them as well.
* TensorFlow sequences are not iterable
* tf.div returns a tensor of the same type as the first argument.
* TensorFlow has TensorBoard to provide a graphical image of the computation graph
* supports both CPUs and GPUs. It also supports distributed computation
  * if both CPU and GPU are present, TensorFlow gives priority to GPU.
  * https://stackoverflow.com/questions/38836269/does-tensorflow-view-all-cpus-of-one-machine-as-one-device
* fn
  * tf.set_random_seed()
  * tf.convert_to_tensor()
  * tf.cast() # to convert Tensors from one data type to another
  * tf.truediv(a,b) # first casts the integer tensors to floating points and then performs element-wise division.
