import numpy as cp
from LeafNetwork import *
from LeafNetwork.Layers import *
from LeafNetwork.Activations import *
# Données d'entrée
input_data = [1, 2]
train_data = [[1, 0]]

# Convertir en tableaux numpy
input_data = cp.array(input_data)
train_data = cp.array(train_data)

# Initialiser le réseau
nn = LeafNetwork(2)
nn.add(Dense(2, 3))
nn.add(ReLU())
nn.add(Dense(3, 2))

# Entraîner le réseau
nn.train(input_data, train_data, epochs=100, learning_rate=0.01)
