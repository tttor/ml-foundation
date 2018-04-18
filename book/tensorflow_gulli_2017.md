# TensorFlow 1.x Deep Learning Cookbook
* Antonio Gulli, Amita Kapoor
* December 2017
* https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-1x-deep-learning-cookbook
* https://github.com/PacktPublishing/TensorFlow-1x-Deep-Learning-Cookbook
* https://github.com/tttor/TensorFlow-1x-Deep-Learning-Cookbook/tree/devel

## 1: intro
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
  * execution of the graph
    * The execution of the graph is performed using Session Object
    * Session Objectencapsulates the environment in which tensor and Operation Objects are evaluated.
    * In the same program code, we can have many session objects.
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
* TensorFlow 
  * has TensorBoard to provide a graphical image of the computation graph
  * uses C/C++ at the backend, which makes it computationally fast.
* supports both CPUs and GPUs. It also supports distributed computation
  * if both CPU and GPU are present, TensorFlow gives priority to GPU.
  * https://stackoverflow.com/questions/38836269/does-tensorflow-view-all-cpus-of-one-machine-as-one-device
* Reading the data: 
  * feeding through feed_dict, 
    * using the feed_dict argument in the run() or eval() function call
  * reading from files,
    * Filename queue: to keep the filenames until the reader needs them
    * Reader is defined and used to read from files from the filename queue.
    * Decoder: to decode the value string into Tensors that make up the training example
  * using preloaded data
    * store data either in a constant or variable
    * While using a variable, we need to set the trainable flag to False so that the data does not change while training
* TensorFlow Estimators
  * provides the option of either using pre-made Estimators or one can write their own custom Estimators
  * no longer have to worry about building the computational or creating a session, it handles it all.
  * https://www.tensorflow.org/programmers_guide/estimators
* misc
  * TensorFlow sequences are not iterable
  * compare to numpy for writing code for a simple one-layer feedforward network
* fn
  * tf.set_random_seed()
  * tf.convert_to_tensor()
  * tf.cast() # to convert Tensors from one data type to another
  * tf.truediv(a,b) # first casts the integer tensors to floating points and then performs element-wise division.
  * tf.div() # returns a tensor of the same type as the first argument.
  * tf.device() # to use specific devices (CPU/GPU) with different objects of the computation graph

## 9: Reinforcement Learning
* nice starting point for policy rep using nets: `02_nn_random_agent.py`
* naive q learning: `03_Q_learning_cartpole.py`

