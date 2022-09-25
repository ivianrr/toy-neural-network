# toy-neural-network
Personal project that involves implementing a neural network froms scratch using only numpy following an object oriented programming approach. 

The file `toynetwork.py` contains the `Layer` and `Network` classes and a minimal example. A proper example of usage is shown in the `toynetwork_train.py` file in which a network is trained with the MNIST dataset.
Helper functions for loading the dataset are included in the util module, as well as the node activation functions, which are needed for the network to work and can be modified or extended as needed. 
The only requisite for these activation functions is to have a `derivative` argument in in order for the backpropagator to be able to get the derivative of each functions without hardcoding them elsewhere.

At the moment, cross entropy is the only loss function of the network, which requires a softmax activation function on the last layer. This makes it ideal for classification problems.
The rest of the layers are fully customizable. 

Other available features are:
- Mini-batch, stochastic descent or batch training.
- Momentum.
- L2 regularization.
- Dynamic learning rate.
- Glorot weight initialization.
- Saving the model to pickle and loading it in a different script (see `toynetwork_train.py`).
- Accuracy plot during training.

The jupyter notebook only contains some previous code used while following some tutorials to make sure the math was right.
