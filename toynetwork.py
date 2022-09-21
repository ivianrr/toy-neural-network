from typing import Callable
import numpy as np


class Functions:
    @staticmethod
    def ReLU(Z, derivative=False):
        if not derivative:
            return np.maximum(Z, 0)
        else:
            return 1*(Z > 0)

    @staticmethod
    def Sigmoid(Z, derivative=False):
        return 1*(Z > 0)

    @staticmethod
    def softmax(Z, derivative=False):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    @staticmethod
    def cross_ent(Y, A):  # true(onehotencoded), predicted
        return -np.log(A.T[Y.astype(bool).T])


class Layer:
    def __init__(self, size: int, activation_function: Callable) -> None:
        self.act_f: np.ndarray = activation_function
        self.size: int = size
        self.weight_dim: tuple[int, int] = None
        self.weights: np.ndarray = None
        self.biases: np.ndarray = None
        self.Z: np.ndarray = None
        self.A: np.ndarray = None
        self.dW: np.ndarray = None
        self.db: np.ndarray = None
        self.dZ: np.ndarray = None
        self.dA: np.ndarray = None

    def __repr__(self) -> str:
        return f"Layer({self.size}, {self.act_f}, {self.weight_dim})"

    def set_input_size(self, input_size: int) -> None:
        self.weight_dim = (self.size, input_size)

    def init_params(self) -> None:
        assert self.weight_dim != None, "Number of inputs for layer not yet defined."
        self.weights = np.random.rand(*self.weight_dim)-0.5
        self.biases = np.random.rand(self.size, 1)-0.5


class Network:
    def __init__(self, input_size: int, loss_function: Callable = None, layer_list: list[Layer] = None):
        self.layers: list[Layer] = []
        self.input_size: int = input_size
        self.loss_function: Callable = loss_function
        if layer_list != None:
            for l in layer_list:
                self.add_layer(l)

    def __repr__(self) -> str:
        rstr = f"Network({self.input_size}, {self.loss_function}"
        for l in self.layers:
            rstr += ", "
            rstr += str(l)
        rstr += ")"
        return rstr

    def set_loss_function(self, f: Callable):
        self.loss_function = f

    @staticmethod
    def do_something():
        print("hello")

    def add_layer(self, layer: Layer):
        if self.layers:
            prev = self.layers[-1].size
        else:
            prev = self.input_size

        layer.set_input_size(prev)
        self.layers.append(layer)

    def init_params(self):
        for layer in self.layers:
            layer.init_params()

    def disp_layers(self):
        for l in self.layers:
            print(l)

    def forward_propagate(self, X: np.ndarray):
        for i, layer in enumerate(self.layers):
            if i == 0:
                input_value = X
            else:
                input_value = self.layers[i-1].A
            layer.Z = layer.weights @ input_value + layer.biases
            layer.A = layer.act_f(layer.Z)
        return layer.A

    def back_propagate(self, X, Y, alpha, epochs, batchsize):
        pass


if __name__ == "__main__":
    nn = Network(input_size=5)
    nn.add_layer(Layer(2, Functions.ReLU))
    nn.add_layer(Layer(4, Functions.softmax))
    nn.set_loss_function(Functions.cross_ent)
    print(nn)
    nn.init_params()
    X = np.random.rand(5, 100)
    print(nn.forward_propagate(X).shape)
