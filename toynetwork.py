from typing import Callable
import numpy as np

class Functions:
    # https://medium.com/@krishnakalyan3/introduction-to-exponential-linear-unit-d3e2904b366c
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
    def ELU(Z,derivative=False):
        a=1
        A=np.zeros_like(Z)
        if not derivative:
            A[Z>0]=Z[Z>0]
            A[Z<=0]=a*(np.exp(Z[Z<=0])-1)
        else:
            A[Z>0]=1
            A[Z<=0]=Functions.ELU(Z[Z<=0],False)+a
        return A


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

    @staticmethod
    def one_hot(labels: np.ndarray):
        max=labels.max()
        Y=np.zeros((max+1,labels.size))
        Y[labels,np.arange(labels.size)]=1
        return Y

    def get_predictions(self):
        A=self.layers[-1].A
        return np.argmax(A,axis=0)

    def backward_prop(self,X,Y):
        for i, layer in reversed(enumerate(self.layers)):
            
            pass

    def gradient_descent(self, X, Y, alpha, epochs, batch_size):
        def batch_split(x,batch_size,axis):
            b=batch_size
            x=x.swapaxes(0,axis)
            l=x.shape[0]
            n_b=range(l//b)
            for r in n_b:
                yield x[r*b:(r+1)*b].swapaxes(0,axis)
            if l%b>0:
                yield x[l//b*b:].swapaxes(0,axis)

        N=Y.size
        Y_one_hot=self.one_hot(Y)
        for i in range(epochs):
            # Shuffle trainig data first
            perms=np.random.permutation(N)
            X=X[:,perms]
            Y=Y_one_hot[perms]
            # Split in batches and run every batch each epoch. 
            for X_b,Y_b in zip(batch_split(X,batch_size,axis=1),batch_split(Y,batch_size,axis=0)):
                Z1,A1,Z2,A2 = forward_prop(W1,b1,W2,b2,X_b)
                dW1,db1,dW2,db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X_b, Y_b)
                W1, b1, W2, b2 = update_params(W1,b1, W2,b2,dW1,db1,dW2,db2,alpha)

        pass


if __name__ == "__main__":
    model = Network(input_size=5)
    model.add_layer(Layer(2, Functions.ReLU))
    model.add_layer(Layer(2, Functions.ELU))
    model.add_layer(Layer(4, Functions.softmax))
    model.set_loss_function(Functions.cross_ent)
    print(model)
    model.init_params()

    X = np.random.rand(5, 10)
    model.forward_propagate(X)

    print("Predictions")
    print(model.get_predictions())

    print("Probabilities of first prediction.")
    print(model.layers[-1].A[:,0])

