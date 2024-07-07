import numpy as np
from LeafNetwork import *
from LeafNetwork import Dense, ReLU

# Convertir en tableaux numpy
input_data = np.array([1, 2])
train_data = np.array([[1, 0]])

# Initialiser le réseau
nn = LeafNetwork(2)
nn.add(Dense(2, 3))
nn.add(ReLU())
nn.add(Dense(3, 2))

# Entraîner le réseau
nn.train(input_data, train_data, epochs=10000, learning_rate=10)
res = nn.predict(input_data)
print(res.tolist())

nn.save("model.json")

nn = LeafNetwork.load("model.json")


nn.train(input_data, train_data, epochs=10000, learning_rate=10)
print(res.tolist())
res = nn.predict(input_data)
print(res.tolist())