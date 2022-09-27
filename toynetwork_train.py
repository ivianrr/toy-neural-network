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
    Y_V, X_V, _, _ = mnist.load_test_data()
    X_V = X_V.T

    # Create and initialise network
    model = Network(input_size=nrows*ncols)
    model.add_layer(Layer(100, Functions.ReLU,lambda_L2=0.005,keep_prob=1))
    model.add_layer(Layer(30, Functions.ELU,lambda_L2=0.005,keep_prob=0.5))
    # for i in range(5):
    #     model.add_layer(Layer(20, Functions.ELU,lambda_L2=0.001,keep_prob=0.8))
    model.add_layer(Layer(30, Functions.Sigmoid,keep_prob=0.5))
    model.add_layer(Layer(10, Functions.softmax))
    model.set_loss_function(Functions.cross_ent)

    model.init_params()

    # Train the network
    acc_T, acc_V = model.gradient_descent(X_T, Y_T, alpha=0.05, epochs=50,
                           batch_size=32, X_V=X_V, Y_V=Y_V,p=0.5,mu=0.9,do_dropout=True)


    # Check accuracy and generate a plot with its history
    # _, acc_T, acc_V=model.history[-1]
    # predictions_T = model.get_predictions(X_T)
    # acc_T = model.get_accuracy(predictions_T, Y_T)
    print("Final accuracy (Training)", f"{acc_T:.3f}")
    # predictions_V = model.get_predictions(X_V)
    # acc_V = model.get_accuracy(predictions_V, Y_V)
    print("Final accuracy (Validation)", f"{acc_V:.3f}")

    with open('model_dropout.pkl', 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    model.plot_history()
