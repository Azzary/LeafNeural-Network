import numpy as np
from .LeafLayer import LeafLayer

class Dense(LeafLayer):
    
    def __init__(self, input_size: float, output_size: float):
        super().__init__()
        """
        weights is a matrix of size input_size x output_size
        each row is a neuron's weights
        for a layer x with 3 neurons and layer y with 4 neurons 
        x1, x2, x3
        y1, y2, y3, y4
        y1 = x1*w11 + x2*w12 + x3*w13
        where w11 is the weight between x1 and y1 and x1 is the output of the neuron x1
        
        {
         w11, w12, ..., w1n 
         w21, w22, ..., w2n
         ...
         wm1, wm2, ..., wmn
        }
        """
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        """
        bias is a vector of size output_size
        {
         b1
         b2
         ...
         bn
        }
        """
        self.bias = np.zeros((output_size, 1)) 
        self.input: np.ndarray
    
    def dynamic_init(self):
        self.input: np.ndarray
    
    def forward(self, input: np.ndarray):
        self.input = input
        """
        output = W * X + B
        for a input with 3 features and a layer with 5 neurons
              y1 = x1*w11 + x2*w12 + x3*w13 + b1
        x1    y2 = x1*w21 + x2*w22 + x3*w23 + b2
        x2 *  y3 = x1*w31 + x2*w32 + x3*w33 + b3
        x3    y4 = x1*w41 + x2*w42 + x3*w43 + b4
              y5 = x1*w51 + x2*w52 + x3*w53 + b5
        """
        return np.dot(self.weights, self.input) + self.bias
        
    def backward(self, output_gradient: np.ndarray, learning_rate: float):
        # Calcul du gradient des poids. `output_grad` est le gradient de la perte par rapport à la sortie de la couche suivante.
        # `self.input.T` est la transposée des activations de la couche précédente, permettant un alignement correct pour le produit matriciel.
        weights_gradient = np.dot(output_gradient, self.input.T)

        # Mise à jour des poids: On multiplie le gradient des poids par le taux d'apprentissage et on soustrait le résultat des poids actuels.
        # Cette descente de gradient permet de minimiser la perte en ajustant les poids en fonction de leur impact sur la perte observée dans la couche suivante.
        self.weights -= learning_rate * weights_gradient

        # Mise à jour des biais: Les biais sont ajustés en soustrayant le produit du gradient de sortie de la couche suivante et du taux d'apprentissage.
        # Chaque biais est modifié en fonction de son impact direct sur la perte totale, en supposant une dérivée simple de 1 pour le biais par rapport à la sortie.
        bias_grad = np.sum(output_gradient, axis=1, keepdims=True)
        self.bias -= learning_rate * bias_grad
        
        # Propagation du gradient vers la couche précédente.
        # On utilise le gradient de la perte par rapport à la sortie de cette couche (output_grad) et on le multiplie par les poids transposés de cette couche.
        # Cela permet de calculer le gradient de la perte par rapport aux activations de la couche précédente, essentiel pour la rétropropagation continue à travers le réseau.
  
        
        
        input_gradient = np.dot(self.weights.T, output_gradient)
        return input_gradient


    
if __name__ == "__main__":
    layer = Dense(3, 2)
    input_data = np.random.randn(1, 3)
    output_grad = np.random.randn(1, 2)
    layer.forward(input_data)
    grad_back = layer.backward(output_grad, 0.01)
    print(f"Gradient back-propagated: {grad_back}")