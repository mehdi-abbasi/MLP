import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.utils import shuffle
import MLP
import matplotlib.pyplot as plt
import time


# ----------------preprocessing dataset------------------
iris = load_iris()
features = iris.data  
target = pd.get_dummies(iris.target).to_numpy()
dataset = np.hstack((features,target,np.reshape(iris.target,(-1,1))))
dataset = shuffle(dataset)
test_data = dataset[120:150,0:4]
test_target = dataset[120:150,4:7]


# -------------------Active Functions--------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s) 

def linear(x):
    return x

def linear_derivative(x):
    return 1

def relu(x):
    return max(0,x)

def relu_derivative(x):
    if x < 0: return 0 
    else: return 1


# ---------------MLP Model Implementation---------------
layers = []
for l in range(1): # Hidden Layers
    layers.append(MLP.Layer(5,sigmoid,sigmoid_derivative))
    # layers.append(MLP.Layer(2,linear,linear_derivative))
    # layers.append(MLP.Layer(5,relu,relu_derivative))

# layers.append(MLP.Layer(3,sigmoid,sigmoid_derivative))
# layers.append(MLP.Layer(3,linear,linear_derivative))
layers.append(MLP.Layer(3,relu,relu_derivative)) # Output Layer


# ---------------Initialize Neural Network---------------
mlp = MLP.Mlp(layers)
mlp.initialize(dataset[0:120,0:4],dataset[0:120,4:7])


# ---------------Training---------------
eta = 0.01
epoch = 200
t_start = time.time()
mlp.train(eta,epoch)
t_end = time.time()


# ----------------------Testing----------------------
err = 0
for d in range(len(test_data)):
    output = mlp.feed_forward(test_data[d])
    err += int(np.argmax(output) != np.argmax(test_target[d]))

accuracy = ((1-err/len(test_data)) * 100) 
print(f'Accuracy= {accuracy:.2f}%')
print(f"time to learn: {(t_end - t_start):.2f} sec" )




# -------------------Finding Best Config.---------------
# comment last two line in MLP.py
# eta = [0.1,0.01,0.001]
# epoch = 400
# plot_data = []
# for e in range(len(eta)):
#     mlp = MLP.Mlp(layers)
#     mlp.initialize(dataset[0:120,0:4],dataset[0:120,4:7])
#     accuracy = []
#     for i in range(epoch):
#         mlp.train(eta[e],1)
#         err = 0
#         for d in range(len(test_data)):
#             output = mlp.feed_forward(test_data[d])
#             err += int(np.argmax(output) != np.argmax(test_target[d]))

#         accuracy.append((1-err/len(test_data)) * 100) 
#     plot_data.append(accuracy)

# plt.plot(plot_data[0],label='η = 0.1')
# plt.plot(plot_data[1],label='η = 0.01')
# plt.plot(plot_data[2],label='η = 0.001')
# plt.title("Accuracy on 3 different Learning rates with 400 epochs")
# plt.legend(loc="lower right")
# plt.xlabel("Eepochs No.")
# plt.ylabel("Accuracy")
# plt.grid()
# plt.show()