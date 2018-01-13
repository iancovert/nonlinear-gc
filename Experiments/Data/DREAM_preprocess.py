from __future__ import division
import numpy as np 
import pandas as pd 


"""
size:size of network, either 50 or 100
network_type: 'Ecoli' or 'Yeast'
number: 1 or 2 for 'Ecoli' and 1 or 2 or 3 for 'Yeast'

output:
dat - a #time points x # replicates x size (p) of series tensor
G - size (p) x size (p) graph of directed interactions

"""
def import_DREAM(size = 50, network_type = 'Ecoli', number = 1):
	N = 21
	filename = 'Data/DREAMDATA/InSilicoSize' + str(size) + '-' + network_type + str(number) + '-trajectories.tsv'
	dat = pd.read_table(filename)
	dat = dat.dropna()
	datp = dat.as_matrix()
	ax1 = datp.shape[0]/N
	datp = datp.reshape(int(ax1),N,(size+1))
	datp_new = np.swapaxes(datp,0,1)
	datp_new_out = datp_new[:,:,1:(size+1)]

	# Read in the graph
	fname = 'DREAM3GoldStandard_InSilicoSize' + str(size) + '_' + network_type + str(number) + '.txt'
	filename = 'Data/DREAMDATA/ground_truth_graphs/' + fname
	G = make_graph(filename,size)
	G = np.maximum(G, np.eye(size))

	return datp_new[:, :, 1:], G


def make_graph(filename,p):
	G = np.zeros((p,p))
	with open(filename) as f:
		for line in f:
			if (int(line.split()[2]) == 1):
				fr = int(line.split()[0][1:])
				to = int(line.split()[1][1:])
				G[to-1,fr-1] = 1
	return G

if __name__ == '__main__': 
	dat, G = import_DREAM()

