from itertools import product
import numpy as np
import time

dstamp = time.strftime('%Y%m%d')
tstamp = time.strftime('%H%M%S')

jobname = 'long_var_lstm_encoding_%s_%s' % (dstamp, tstamp)
jobfile = 'Batches/%s.job' % jobname

nepoch_grid = [1500]
lr_grid = [0.01]
lam_grid = np.append(np.geomspace(3.0, 0.001, num = 50), 0)
seed_grid = [0]
hidden_grid = [5, 10, 20, 30]

sparsity_grid = [0.3]
p_grid = [10]
T_grid = [500]
lag_grid = [5]

BASECMD = 'python long_var_lstm_encoding.py'

param_grid = product(nepoch_grid, lr_grid, lam_grid, seed_grid, hidden_grid,
	sparsity_grid, p_grid, T_grid, lag_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		nepoch, lr, lam, seed, hidden, sparsity, p, T, lag = param

		argstr = BASECMD
		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --lr=%e' % lr
		argstr += ' --lam=%e' % lam
		argstr += ' --seed=%d' % seed
		argstr += ' --hidden=%d' % hidden

		argstr += ' --sparsity=%e' % sparsity
		argstr += ' --p=%d' % p
		argstr += ' --T=%d' % T
		argstr += ' --lag=%d' % lag

		#f.write(argstr + '\n')
		print(argstr)