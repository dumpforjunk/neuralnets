Neural Nets
===========

multilayerperceptron.py
-----------------------

Provides the class Perceptron which represents a multi-layer perceptron and is initialised with a list giving the number of nodes in each layer.
    
    from multilayerperceptron import Perceptron

    neural_net = Perceptron([3,2,3])

The multi-layer perceptrons provided by this package always output one answer strictly between 0 and 1, and take inputs of any size. The number of inputs taken is the number of nodes in the first layer.

The weights in the network are initialised randomly. To train the neural net, provide an example input and the correct answer along with a constant for how quickly the weights should be changed. The weights will be adjusted using backpropagation.

    neural_net.train([0.1, 0.2, 0.3], 0.6, 0.01)

After sufficient training, you can use your neural net for computations:

    neural_net.compute_result([0.2, 0.3, 0.7])
