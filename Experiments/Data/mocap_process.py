import numpy as np 
import scipy.io as spio

def load_data_12():
    TD = np.loadtxt('mocap124dataset/data_per_seq/01_01.dat')

"""
provides list of mocap datasets for the 64 joints. output is a list
of datasets. 
"""
def load_data_64():
    fname = 'mocap6dataset-master/mocapdata_.mat'
    mat = spio.loadmat(fname, squeeze_me=True)

    x = list()
    X = mat['X']
    for i in range(X.shape[0]):
        x.append(X[i]['dat'])
    return x,X[0]['ColNames']

def determine_equivalent_edges(names):
    name_list = list()
    print(len(names))
    for i in range(len(names)):
        name_temp = str(names[i])[:-3]
        name_list.append(str(names[i])[:-3])

    unique_names = np.unique(name_list)

    index_list = list()
    for i in range(len(unique_names)):
        index_list.append(multiple_indices(name_list,unique_names[i]))

    return(index_list,unique_names)


def multiple_indices(list_h,key):
    str_list = list()
    for i, j in enumerate(list_h):
        if j == key:
            str_list.append(i)
    return(np.array(str_list))


def agg_gamma_graph(Gamma,index_list,unique_names,ag_type="average"):
    p = Gamma.shape[0]
    p_new = len(unique_names)
    Gamma_temp = np.zeros(p,p_new)
    Gamma_new = np.zeros(p_new,p_new)
    for i in range(len(names)):
        if (ag_type == "average"):
                Gamma_temp[:,i] = np.mean(Gamma[:,index_list[i]],axis = 0)
        if (ag_type == "max"):
                Gamma_temp[:,i] = np.max(Gamma[:,index_list[i]],axis = 0)

    for i in range(len(names)):
        if (ag_type == "average"):
            Gamma_new[i,:] = np.mean(Gamma_temp[index_list[i],:],axis=1)
        if (ag_type == "max"):
            Gamma_new[i,:] = np.max(Gamma_temp[index_list[i],:],axis=1)

    return(Gamma_new)


def plot_mocap_results(Gamma,names):
    index_list, unique_names = determine_equivalent_edges(names)
    Gamma_new = make_GC_graph(Gamma,index_list,unique_names)
    thresh = .001
    Gamma_bin = Gamma_new > thresh


#def locations_joints(names):


if __name__ == "__main__": 
    x, names = load_data_64()
    idex_list, unique_names = determine_equivalent_edges(names)



