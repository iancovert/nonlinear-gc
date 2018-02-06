import numpy as np 
import scipy.io as spio
from Data.data_processing import YX_list, reshape_list

def load_data_12():
    TD = np.loadtxt('mocap124dataset/data_per_seq/01_01.dat')

"""
provides list of mocap datasets for the 64 joints. output is a list
of datasets. 
"""
def load_data_64():
    fname = 'Data/mocap6dataset-master/mocapdata_.mat'
    mat = spio.loadmat(fname, squeeze_me=True)

    x = list()
    X = mat['X']
    for i in range(X.shape[0]):
        #get rid of fingers


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


def plot_mocap_results(Gamma,names,thresh,weighted=False):
    index_list, unique_names = determine_equivalent_edges(names)
    Gamma_new = agg_gamma_graph(Gamma,index_list,unique_names)
    thresh = .001
    if not weighted:
        Gamma_bin = Gamma_new > thresh
        locs = locations_joints(unique_names)
        e = np.where(Gamma_bin)
        e_list = zip(*e)
        G = igraph.Graph(e_list,directed = True)
        G.vs["x"] = locs[:,0]
        G.vs["y"] = -locs[:,1]
        igraph.plot(G,inline = False)


def locations_joints(names):
    locs = np.zeros((len(names),2))
    middle = 1
    ###location of the head
    locs[0,:] = [middle,1.1]

    ###location of the upperneck
    yf = 1
    locs[24,:] = [middle, yf]

    ###location of the lowerneck
    yf = .92
    locs[7,:] = [middle,yf]

    ###location of the upperback
    yf = .8
    locs[23,:] = [middle, yf]

    ###location of the thorax
    yf = .7
    locs[22,:] = [middle, yf]


    ###location of the root
    yf = .6
    locs[17,:] = [middle, yf]

    ###location of the lowerback
    yf = .5
    locs[6,:] = [middle,yf]


    ###locations of the clavicle
    yf = .92
    xf = .08
    locs[1,:] = [middle + xf,yf]
    locs[12,:] = [middle - xf,yf]

    ###locations of the humerus
    yf = .85
    xf = .15
    locs[5,:] = [middle + xf,yf]
    locs[16,:] = [middle - xf,yf]

    ###locations of the radius
    yf = .72 
    xf = .2
    locs[8,:] = [middle + xf,yf]
    locs[18,:] = [middle - xf,yf]

    ###locations of the wrist
    yf = .65
    xf = .25
    locs[11,:] = [middle + xf, yf]
    locs[21,:] = [middle - xf, yf]

    ###locations of the hand
    yf = .6 
    xf = .3
    locs[4,:] = [middle + xf,yf]
    locs[15,:] = [middle - xf,yf]


    ###locations of the femur
    yf = .4
    xf = .1
    locs[2,:] = [middle + xf,yf]
    locs[13,:] = [middle - xf,yf]




    ###locations of the tibia
    yf = .2
    xf = .15
    locs[9,:] = [middle + xf, yf]
    locs[19,:] = [middle - xf, yf]

    ###locations of the foot
    yf = .05 - .05
    xf = .15
    locs[3,:] = [middle + xf,yf]
    locs[14,:] = [middle - xf,yf]

    ###locations of the toes
    yf = .02 -.05
    xf = .2
    locs[10,:] = [middle + xf, yf]
    locs[20,:] = [middle - xf, yf]

    return(locs)





if __name__ == "__main__": 
    x, names = load_data_64()
    X = reshape_list(x,20)
    y_train, x_train = YX_list(x)


    idex_list, unique_names = determine_equivalent_edges(names)
    import igraph
    from igraph import *
    import cairo
    p = len(unique_names)
    g = np.zeros((p,p))
    g[1,10] = 1
    g[3,3] = 1
    g[14,3] = 1
    g[24,23] = 1

    locs = locations_joints(unique_names)
    e = np.where(g)
    e_list = zip(*e)
    G = igraph.Graph(e_list,directed = True)
    G.vs["x"] = locs[:,0]
    G.vs["y"] = -locs[:,1]
    igraph.plot(G,inline = False)



