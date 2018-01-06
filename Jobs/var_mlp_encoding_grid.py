from itertools import product
import numpy as np
import time

dstamp = time.strftime('%Y%m%d')
tstamp = time.strftime('%H%M%S')

jobname = 'var_mlp_encoding_%s_%s' % (dstamp, tstamp)
jobfile = 'Batches/%s.job' % jobname

nepoch_grid = [20000]
lr_grid = [0.01]
lam_grid = np.append(np.geomspace(3.0, 0.001, num = 50), 0)
seed_grid = [0]
hidden_grid = [5, 10]
network_lag_grid = [1, 2]

sparsity_grid = [0.3]
p_grid = [10]
T_grid = [500]
lag_grid = [1, 2]

BASECMD = 'python var_mlp_encoding.py'

param_grid = product(nepoch_grid, lr_grid, lam_grid, seed_grid, hidden_grid, network_lag_grid,
	sparsity_grid, p_grid, T_grid, lag_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		nepoch, lr, lam, seed, hidden, network_lag, sparsity, p, T, lag = param

		argstr = BASECMD
		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --lr=%e' % lr
		argstr += ' --lam=%e' % lam
		argstr += ' --seed=%d' % seed
		argstr += ' --hidden=%d' % hidden
		argstr += ' --network_lag=%d' % network_lag

		argstr += ' --sparsity=%e' % sparsity
		argstr += ' --p=%d' % p
		argstr += ' --T=%d' % T
		argstr += ' --lag=%d' % lag

		f.write(argstr + '\n')
