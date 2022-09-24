import pickle
from toynetwork import Network, Layer,Functions

with open('model.pkl', 'rb') as inp:
    model: Network = pickle.load(inp)

print(model.layers[-1].A.T[:5])

model.plot_history()
