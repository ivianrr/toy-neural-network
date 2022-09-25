from toynetwork import Network, Layer,Functions
import numpy as np
import pickle

import matplotlib.pyplot as plt
from util import networkfunctions as Functions
from util import mnistdataset as mnist

if __name__ == "__main__":
    # Load dataset
    Y_T, X_T, nrows, ncols = mnist.load_data()
    X_T = X_T.T
    Y_V, X_V, _, _ = mnist.load_validation_data()
    X_V = X_V.T

    # Create and initialise network
    model = Network(input_size=nrows*ncols)
    model.add_layer(Layer(50, Functions.ReLU))
    # for i in range(5):
    #     model.add_layer(Layer(10, Functions.ELU))
    model.add_layer(Layer(30, Functions.Sigmoid))
    model.add_layer(Layer(10, Functions.softmax))
    model.set_loss_function(Functions.cross_ent)
    model.init_params()

    # Train the network
    model.gradient_descent(X_T, Y_T, alpha=0.1, epochs=10,
                           batch_size=32, X_V=X_V, Y_V=Y_V)


    # Check accuracy and generate a plot with its history
    _, acc_T, acc_V=model.history[-1]
    # predictions_T = model.get_predictions(X_T)
    # acc_T = model.get_accuracy(predictions_T, Y_T)
    print("Final accuracy (Training)", acc_T)
    # predictions_V = model.get_predictions(X_V)
    # acc_V = model.get_accuracy(predictions_V, Y_V)
    print("Final accuracy (Validation)", acc_V)

    with open('model_deep.pkl', 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    model.plot_history()
