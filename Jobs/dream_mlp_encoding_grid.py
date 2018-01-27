from itertools import product
import numpy as np
import time

dstamp = time.strftime('%Y%m%d')
tstamp = time.strftime('%H%M%S')

jobname = 'dream_mlp_encoding_%s_%s' % (dstamp, tstamp)
jobfile = 'Batches/%s.job' % jobname

lam_grid = np.append(np.geomspace(10.0, 0.001, num = 50), 0)
seed_grid = [0]
hidden_grid = [4, 8]
network_lag_grid = [2]

nepoch_grid = [5000]
lr_grid = [0.001]

size_grid = [100]
type_grid = ['Ecoli', 'Yeast']
number_grid = [1, 2, 3]

BASECMD = 'python dream_mlp_encoding.py'

param_grid = product(lam_grid, seed_grid, hidden_grid, network_lag_grid,
	nepoch_grid, lr_grid,
	size_grid, type_grid, number_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		lam, seed, hidden, network_lag, nepoch, lr, size, typ, number = param

		argstr = BASECMD

		argstr += ' --lam=%e' % lam
		argstr += ' --seed=%d' % seed
		argstr += ' --hidden=%d' % hidden
		argstr += ' --network_lag=%d' % network_lag

		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --lr=%e' % lr
		
		argstr += ' --size=%d' % size
		argstr += ' --type=%s' % typ
		argstr += ' --number=%d' % number

		f.write(argstr + '\n')
