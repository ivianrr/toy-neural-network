import numpy as np

# https://medium.com/@krishnakalyan3/introduction-to-exponential-linear-unit-d3e2904b366c


def ReLU(Z, derivative=False):
    if not derivative:
        return np.maximum(Z, 0)
    else:
        return 1*(Z > 0)


def Sigmoid(Z, derivative=False):
    if not derivative:

        return 1/(1+np.exp(-Z))
    else:
        sz=Sigmoid(Z,derivative=False)
        return sz*(1-sz)


def softmax(Z, derivative=False):
    # return np.exp(Z) / np.sum(np.exp(Z), axis=0)
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / e_Z.sum(axis=0)


def ELU(Z, derivative=False, a=1):
    A = np.zeros_like(Z)
    if not derivative:
        A[Z > 0] = Z[Z > 0]
        A[Z <= 0] = a*(np.exp(Z[Z <= 0])-1)
    else:
        A[Z > 0] = 1
        A[Z <= 0] = ELU(Z[Z <= 0], False)+a
    return A


def ELU_alpha_generator(alpha):
    def ELU_alpha(*args, **kwargs):
        return ELU(*args, **kwargs, a=alpha)
    return ELU_alpha


def cross_ent(Y, A):  # true(onehotencoded), predicted
    return -np.log(A.T[Y.astype(bool).T])
