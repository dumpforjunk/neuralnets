"""
multilayerperceptron.py - Eliot Ball <eb@eliotball.com>

Provides the class Perceptron which represents a multi-layer perceptron and
is initialised with a list giving the number of nodes in each layer.

    neural_net = Perceptron([3,2,3])

The multi-layer perceptrons provided by this package always output one
answer strictly between 0 and 1, and take inputs of any size. The number
of inputs taken is the number of nodes in the first layer.

The weights in the network are initialised randomly. To train the neural
net provide an example input and the correct answer, along with a constant 
for how quickly the weights should be changed. The weights will be adjusted 
using backpropagation.

    neural_net.train([0.1, 0.2, 0.3], 0.6, 0.01)

After sufficient training you can use your neural net for computations:

    neural_net.compute_result([0.2, 0.3, 0.7])

"""


import math
import random


class Perceptron:
    """
    Implements a multi-layer perceptron, initialised with a list of the number
    of nodes in each layer.
    """
    def __init__(self, layer_node_counts):
        """
        Create a new multi-layer perceptron with layers of the specified size, 
        and an output layer with one node.
        """
        self.input_count = layer_node_counts[0]
        self.layers = [Layer(layer_node_counts[i], layer_node_counts[i + 1])
                       for i in range(len(layer_node_counts) - 1)]
        self.layers += [Layer(layer_node_counts[-1], 1)]
    
    def compute_result(self, input_signals):
        """
        Compute the output of the neural network for a given set of input 
        signals.
        """
        # Pass the input through each layer
        signals = input_signals[:]
        for layer in self.layers:
            signals = layer.process_input_signals(signals)
        # Now we have just one signal which is the output
        return signals[0]

    def train(self, input_signals, correct_answer, adjustment_strength):
        """
        Uses backpropagation to adjust the weights in the network to better
        fit an example correct answer.
        """
        # First propagate the input signals through the whole network
        layer_input_signals = [input_signals]
        for layer in self.layers:
            layer_input_signals += [
                layer.process_input_signals(layer_input_signals[-1])]
        # Now backpropagate the deltas through the whole network
        layer_deltas = [[correct_answer - layer_input_signals[-1][0]]]
        for layer in reversed(self.layers):
            layer_deltas = [
                layer.process_deltas(layer_deltas[0])] + layer_deltas
        # Finally, adjust the weights on the layers accrodingly
        for layer_number in range(len(self.layers)):
            self.layers[layer_number].improve_weights(
               layer_input_signals[layer_number], layer_deltas[layer_number + 1],
               adjustment_strength) 


class Layer:
    def __init__(self, input_count, output_count):
        """
        Create a new layer with the specified number of inputs and outputs.
        """
        self.input_count = input_count
        self.output_count = output_count
        self.weights = dict([(str(input) + "-" + str(output), random.random())
                             for input in range(input_count)
                             for output in range(output_count)])
    
    def get_weight(self, input, output):
        """
        Get the weight of the arc between a practicular input and output.
        """
        return self.weights[str(input) + "-" + str(output)]

    def set_weight(self, input, output, value):
        """
        Set the weight between an input to a layer and the output node.
        """
        self.weights[str(input) + "-" + str(output)] = value

    def activation_function(self, input):   
        """
        The logistic function.
        """
        return 1.0 / (1.0 + math.exp(-input))
    
    def process_input_signals(self, input_signals):
        """
        Take a list of input signals and produce the output signals of the
        layer.
        """
        # Combine weighted input signals with horrible/awesome nested list 
        # comprehension
        weighted_signals = [sum([input_signals[input] * self.get_weight(input, output)
                                 for input in range(self.input_count)])
                            for output in range(self.output_count)]
        # Apply the activation function to all of the weighted signals and 
        # return outputs
        return map(self.activation_function, weighted_signals)

    def process_deltas(self, deltas):
        """
        Take a list of deltas for this layer and produce the back-propagated
        deltas for the previous layer.
        """
        # Backpropagate by combining weighted deltas from this layer
        return [sum([deltas[output] * self.get_weight(input, output)
                     for output in range(self.output_count)])
                for input in range(self.input_count)]
    
    def improve_weights(self, input_signals, deltas, adjustment_strength):
        """
        Adjusts the weights according to the backpropagated deltas and the
        input signals recieved.
        """
        for input in range(self.input_count):
            for output in range(self.output_count): 
                self.weights[str(input) + "-" + str(output)] += (
                    adjustment_strength * input_signals[input] * deltas[output])
