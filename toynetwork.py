import pickle
from typing import Callable
import numpy as np

import struct
import matplotlib.pyplot as plt

def load_data():
    # with open('samples/t10k-images.idx3-ubyte','rb') as f:
    with open('samples/train-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, -1))
        data=data/255
    # with open('samples/t10k-labels.idx1-ubyte','rb') as f:
    with open('samples/train-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        labels = labels.reshape((size,)) # (Optional)
    return labels, data, nrows, ncols
def load_validation_data():
    with open('samples/t10k-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, -1))
        data=data/255
    with open('samples/t10k-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        labels = labels.reshape((size,)) # (Optional)
    return labels, data, nrows, ncols

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
    def ELU(Z,derivative=False,a=1):
        A=np.zeros_like(Z)
        if not derivative:
            A[Z>0]=Z[Z>0]
            A[Z<=0]=a*(np.exp(Z[Z<=0])-1)
        else:
            A[Z>0]=1
            A[Z<=0]=Functions.ELU(Z[Z<=0],False)+a
        return A

    @staticmethod
    def ELU_alpha_generator(alpha):
        def ELU_alpha(*args,**kwargs):
            return Functions.ELU(*args,**kwargs,a=alpha)
        return ELU_alpha

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
        self.history=None
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

    def get_predictions(self, X=None):
        if X is not None: self.forward_propagate(X)            
        A=self.layers[-1].A
        return np.argmax(A,axis=0)

    def backward_prop(self,X:np.ndarray, Y:np.ndarray):
        """
        Args:
        Y:one hot encoded
        """
        batch_size=X.shape[1]
        output_layer=self.layers[-1]
        if self.loss_function.__name__ == "cross_ent" and output_layer.act_f.__name__=="softmax":
            output_layer.dZ=output_layer.A-Y
            output_layer.db=np.sum(output_layer.dZ,axis=1,keepdims=True)/batch_size
            output_layer.dW= (output_layer.dZ @ self.layers[-2].A.T )/batch_size
        else:
            print("Loss:",self.loss_function.__name__,"Act:",)
            raise NotImplementedError("Can't handle other combinations of last layer activation function and loss than softmax, cross_entropy yet.")
        for i, layer in reversed(list(enumerate(self.layers[:-1]))): # TODO: revisar esto
            upper_layer=self.layers[i+1]
            layer.dZ= upper_layer.weights.T @ upper_layer.dZ * layer.act_f(layer.Z,derivative=True)
            layer.db=np.sum(layer.dZ,axis=1,keepdims=True)/batch_size

            lower_A = X if i==0 else self.layers[i-1].A
            layer.dW = (layer.dZ @ lower_A.T )/batch_size

    def update_params(self,alpha):
        for layer in self.layers:
            layer.weights-=alpha*layer.dW
            layer.biases-=alpha*layer.db


    def gradient_descent(self, X, Y,*, alpha, epochs, batch_size,X_V=None,Y_V=None):
        def batch_split(x,batch_size,axis):
            b=batch_size
            x=x.swapaxes(0,axis)
            l=x.shape[0]
            n_b=range(l//b)
            for r in n_b:
                yield x[r*b:(r+1)*b].swapaxes(0,axis)
            if l%b>0:
                yield x[l//b*b:].swapaxes(0,axis)
        self.history=[]
        N=Y.size
        Y_one_hot=self.one_hot(Y)
        for i in range(epochs):
            # Shuffle trainig data first
            perms=np.random.permutation(N)
            X_perm=X[:,perms]
            Y_perm=Y_one_hot[:,perms]
            # Split in batches and run every batch each epoch. 
            nb=0
            for X_b,Y_b in zip(batch_split(X_perm,batch_size,axis=1),batch_split(Y_perm,batch_size,axis=1)):
                self.forward_propagate(X_b)
                self.backward_prop(X_b, Y_b)
                self.update_params(alpha)

                # # Overkill epoch logging, just for fun
                # predictions=self.get_predictions(X)
                # acc=self.get_accuracy(predictions,Y)
                # acc_v=0
                # if X_V is not None and Y_V is not None:
                #     predictions_v=self.get_predictions(X_V)
                #     acc_v=self.get_accuracy(predictions_v,Y_V)
                # self.history.append((i+nb/N*batch_size,acc,acc_v))
                # print(f"Epoch: {i+nb/N*batch_size:.2f}\tAccuracy (T): {acc:.4f}\tAccuracy (V): {acc_v:.4f}")

                nb+=1
            if i%1==0:
                acc=0
                predictions=self.get_predictions(X)
                acc=self.get_accuracy(predictions,Y)
                acc_v=0
                if X_V is not None and Y_V is not None:
                    predictions_v=self.get_predictions(X_V)
                    acc_v=self.get_accuracy(predictions_v,Y_V)
                self.history.append((i,acc,acc_v))
                print(f"Epoch: {i}\tAccuracy (T): {acc:.4f}\tAccuracy (V): {acc_v:.4f}")

    @staticmethod
    def get_accuracy(predictions,Y):
        return np.sum(predictions==Y)/Y.size
    
    def plot_history(self):
        epoch,train_acc,val_acc=zip(*self.history)
        plt.plot(epoch,train_acc,label="Training")
        plt.plot(epoch,val_acc,label="Validation")
        ax=plt.gca()
        ax.set_xlim([0, None])
        ax.set_ylim([0, 1])
        ax.legend()
        plt.show()


if __name__ == "__main__":
    # Load dataset
    Y_T, X_T, nrows,ncols =load_data()
    X_T=X_T.T
    Y_V, X_V, _,_ =load_validation_data()
    X_V=X_V.T    

    # Create and initialise network
    model = Network(input_size=nrows*ncols)
    model.add_layer(Layer(50, Functions.ELU))
    model.add_layer(Layer(10, Functions.softmax))
    model.set_loss_function(Functions.cross_ent)    
    model.init_params()

    # Train the network
    model.gradient_descent(X_T,Y_T,alpha=0.05,epochs=50,batch_size=32,X_V=X_V,Y_V=Y_V)

    # Check accuracy and generate a plot with its history
    predictions_T=model.get_predictions(X_T)
    acc_T=model.get_accuracy(predictions_T,Y_T)
    print("Final accuracy (Training)",acc_T)
    predictions_V=model.get_predictions(X_V)
    acc_V=model.get_accuracy(predictions_V,Y_V)
    print("Final accuracy (Validation)",acc_V)

    with open('model.pkl', 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    model.plot_history()
