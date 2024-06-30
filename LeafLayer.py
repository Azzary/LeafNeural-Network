from abc import ABC, abstractmethod

class LeafLayer(ABC):
    
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        pass
    
    @abstractmethod
    def backward(self, output_grad, learning_rate):
        pass