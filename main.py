import numpy as np
from LeafNetwork import *

# Données d'entrée
input_data = [1, 2]
train_data = [[1, 0]]

# Convertir en tableaux numpy
input_data = np.array([1, 2])
train_data = np.array(train_data)

# Initialiser le réseau
nn = LeafNetwork(2)
nn.add(Dense(2, 3))
nn.add(ReLU())
nn.add(Dense(3, 2))

# Entraîner le réseau
nn.train(input_data, train_data, epochs=100, learning_rate=0.01)
res = nn.predict(input_data)
print(res.tolist())

nn.save("model.json")

nn = LeafNetwork.load("model.json")

res = nn.predict(input_data)
print(res.tolist())

nn.train(input_data, train_data, epochs=100, learning_rate=0.01)
