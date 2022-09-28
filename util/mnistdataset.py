import struct
import numpy as np


def load_data()->tuple[np.ndarray,np.ndarray,int,int]:
    # with open('samples/t10k-images.idx3-ubyte','rb') as f:
    with open('samples/train-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, -1))
        data = data/255
    # with open('samples/t10k-labels.idx1-ubyte','rb') as f:
    with open('samples/train-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        labels = labels.reshape((size,))  # (Optional)
    return labels, data, nrows, ncols


def load_test_data():
    with open('samples/t10k-images.idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, -1))
        data = data/255
    with open('samples/t10k-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        labels = labels.reshape((size,))  # (Optional)
    return labels, data, nrows, ncols

def load_data_augmented()->tuple[np.ndarray,np.ndarray,int,int]:
    nrows, ncols =28,28
    # with open('samples/t10k-images.idx3-ubyte','rb') as f:
    data=np.load("samples/augmented/data.npy")
    labels=np.load("samples/augmented/labels.npy")
    return labels, data, nrows, ncols