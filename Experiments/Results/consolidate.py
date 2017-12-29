import argparse
import os
import pickle
import six
import pandas as pd
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=None, help='directory containing results to process')
parser.add_argument('--outname', type=str, default=None,help='(optional) name for output file')
args = parser.parse_args()

resdir = args.dir
if resdir is None:
	raise ValueError('must specify directory where results are located')

outname = args.outname
if outname is None:
	split_name = resdir.split('/')
	if split_name[-1] == '':
		base = split_name[-2]
	else:
		base = split_name[-1]	
	outname = ''.join([base.replace(' ', '_'), '.data'])

res_files = glob.glob(os.path.join(resdir, '*.out'))
res_files = [os.path.basename(f) for f in res_files]

if len(res_files) < 1:
	raise ValueError("No result files found in %s" % resdir)

all_par_dict_list = []
for rf in res_files:
	with open(os.path.join(resdir, rf), 'rb') as f:
		res = pickle.load(f)
	experiment_params = res
	if 'model' in experiment_params.keys():
		del experiment_params['model']
	all_par_dict_list.append(experiment_params)

experiment_df = pd.DataFrame(all_par_dict_list)

experiment_df.to_pickle(outname)
