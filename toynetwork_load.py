import pickle
from toynetwork import Network, Layer,Functions

with open('model.pkl', 'rb') as inp:
    model: Network = pickle.load(inp)

print(model.layers[-1].A.T[:5])
_, acc_T, acc_V=model.history[-1]
print("Final accuracy (Training)", acc_T)
print("Final accuracy (Validation)", acc_V)

model.plot_history()
