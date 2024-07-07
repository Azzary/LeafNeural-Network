from abc import ABC, abstractmethod
import numpy as np

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

    def dynamic_init(self):
        """
        This method is called only if the constructor of the class has parameters
        and can be used for calling the parent constructor or initializing values.

        Error: The class was created without calling the constructor because it has parameters.
        If you need to perform any initialization at the beginning, please implement the `dynamic_init`
        method to initialize your values.
        """
        print(NotImplementedError(
            f"The class {self.__class__.__name__} was created without calling the constructor because it has parameters. "
            "If you need to perform any initialization at the beginning, please implement the `dynamic_init` "
            "method to initialize your values."
        ))
    

    def save(self):
        """
        Return a JSON-serializable dictionary of the layer's parameters.
        format: {
            "type": "ClassName",
            "module": "ModuleName",
            ...
        }
        """
        data = {
            "type": self.__class__.__name__,
            "module": self.__class__.__module__
        }
        for attr, value in self.__dict__.items():
            if not callable(value) and not attr.startswith('_') and attr not in ['input', 'output']:
                if isinstance(value, np.ndarray):
                    data[attr] = {
                        "type": "ndarray",
                        "dtype": str(value.dtype),
                        "shape": value.shape,
                        "data": value.tolist()
                    }
                elif isinstance(value, (int, float, str, bool)):
                    data[attr] = value
                elif isinstance(value, (list, tuple)):
                    data[attr] = {
                        "type": type(value).__name__,
                        "data": value
                    }
                elif isinstance(value, dict):
                    data[attr] = {
                        "type": "dict",
                        "data": {str(k): self._serialize_value(v) for k, v in value.items()}
                    }
        return data

    @staticmethod
    def _serialize_value(value):
        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "dtype": str(value.dtype),
                "shape": value.shape,
                "data": value.tolist()
            }
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return {
                "type": type(value).__name__,
                "data": value
            }
        elif isinstance(value, dict):
            return {
                "type": "dict",
                "data": {str(k): LeafLayer._serialize_value(v) for k, v in value.items()}
            }
        else:
            return str(value)

    @classmethod
    def load(cls, data):
        """Create and return a new instance of the layer from saved data"""
        if len(cls.__init__.__code__.co_varnames) == 1:
            instance = cls()
        else:
            instance = cls.__new__(cls)
            instance.dynamic_init()
        for attr, value in data.items():
            if attr not in ["type", "module"]:
                if isinstance(value, dict) and "type" in value:
                    if value["type"] == "ndarray":
                        setattr(instance, attr, np.array(value["data"], dtype=value["dtype"]))
                    elif value["type"] in ["list", "tuple"]:
                        setattr(instance, attr, eval(value["type"])(value["data"]))
                    elif value["type"] == "dict":
                        setattr(instance, attr, {k: cls._deserialize_value(v) for k, v in value["data"].items()})
                else:
                    setattr(instance, attr, value)
        return instance

    @staticmethod
    def _deserialize_value(value):
        if isinstance(value, dict) and "type" in value:
            if value["type"] == "ndarray":
                return np.array(value["data"], dtype=value["dtype"])
            elif value["type"] in ["list", "tuple"]:
                return eval(value["type"])(value["data"])
            elif value["type"] == "dict":
                return {k: LeafLayer._deserialize_value(v) for k, v in value["data"].items()}
        return value