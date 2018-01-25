import numpy as np 
import scipy.io as spio

def load_data_12():
    TD = np.loadtxt('mocap124dataset/data_per_seq/01_01.dat')

def load_data_64():
    fname = 'mocap6dataset-master/mocapdata_.mat'
    mat = spio.loadmat(fname, squeeze_me=True)

    x = list()
    X = mat['X']
    for i in range(X.shape[0]):
        x.append(X[i]['dat'])
    return x,X[0]['ColNames']

if __name__ == "__main__": 
    x, names = load_data_64()



