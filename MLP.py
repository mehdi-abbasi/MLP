from numpy.random import rand
from numpy import zeros,ones,dot
from sys import stdout
import time



class Neuron:
    def __init__(self,weights_num,active_func,active_func_derivative):
        self.out = 0
        self.weights_num = weights_num
        self.active_func = active_func
        self.active_func_derivative = active_func_derivative
        self.target = []
        self.weights = rand(self.weights_num)
        self.new_weights = ones(self.weights_num)
        self.delta = 0


    def output(self,input_vector):
        # net = dot(self.weights,input_vector)

        net = 0
        for i in range(len(input_vector)):
            net += self.weights[i] * input_vector[i]
        net += self.weights[-1]

        self.out = self.active_func(net)
        self.derivative = self.active_func_derivative(net)

        return self.out



class Layer:
    def __init__(self,neurons_num,active_func,active_func_derivative):
        self.out = ones(neurons_num)
        self.neurons_num = neurons_num
        self.active_func = active_func
        self.active_func_derivative = active_func_derivative

    def build(self,before_layer_out):
        self.neurons = [Neuron(before_layer_out,self.active_func,self.active_func_derivative) for i in range(self.neurons_num)]


class Mlp:
    def __init__(self,layers):
        self.layers = layers


    def initialize(self,train_data,target):
        self.train_data = train_data
        self.target = target

        for l in range(len(self.layers)):
            if l == 0: self.layers[l].build(len(self.train_data[0]))
            else: self.layers[l].build(self.layers[l-1].neurons_num)

        
        for i in range(len(self.target)):
            for n in range(self.layers[-1].neurons_num):
                self.layers[-1].neurons[n].target.append(self.target[i][n])


    def feed_forward(self,input_vector):
        for l in range(len(self.layers)):
            if l == 0:
                for n in range(self.layers[l].neurons_num):
                    self.layers[l].out[n] = self.layers[l].neurons[n].output(input_vector)
            else:
                for n in range(self.layers[l].neurons_num):
                    self.layers[l].out[n] = self.layers[l].neurons[n].output(self.layers[l-1].out)
        return self.layers[-1].out


    def train(self,learning_rate,epoch):
        for item in range(epoch):
          for d in range(len(self.train_data)):
            self.feed_forward(self.train_data[d])
            for layer in reversed(self.layers):
                # last layer
                if self.layers.index(layer) == len(self.layers) - 1:
                    for neuron in layer.neurons:
                        neuron.delta = neuron.derivative * (neuron.target[d] - neuron.out)

                        for w in range(neuron.weights_num):
                            neuron.new_weights[w] = neuron.weights[w] + learning_rate * neuron.delta * self.layers[self.layers.index(layer)-1].out[w]

                # first layer
                elif self.layers.index(layer) == 0:
                    for neuron in layer.neurons:
                        delta_sum = 0                            
                        for n in self.layers[self.layers.index(layer)+1].neurons:
                            delta_sum += n.delta * n.weights[layer.neurons.index(neuron)]

                        neuron.delta = neuron.derivative * delta_sum

                        for w in range(neuron.weights_num):
                            neuron.new_weights[w] = neuron.weights[w] + learning_rate * neuron.delta * self.train_data[d][w]

                # hidden layers
                else:
                    for neuron in layer.neurons:
                        delta_sum = 0
                        for n in self.layers[self.layers.index(layer)+1].neurons:
                            delta_sum += n.delta * n.weights[layer.neurons.index(neuron)]

                        neuron.delta = neuron.derivative * delta_sum

                        for w in range(neuron.weights_num):
                            neuron.new_weights[w] = neuron.weights[w] + learning_rate * neuron.delta * self.layers[self.layers.index(layer) - 1].out[w]

            for layer in self.layers:
                for neuron in layer.neurons:
                    neuron.weights = neuron.new_weights.copy()

            stdout.write('\r' + "Training: "+ F"|{'█'*int(item/epoch*50)}{'-'*int(50*(1-item/epoch))}|" + f"{float((item+1)/epoch*100):.2f}%")
        print()