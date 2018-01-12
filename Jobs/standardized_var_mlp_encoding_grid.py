from itertools import product
import numpy as np
import time

dstamp = time.strftime('%Y%m%d')
tstamp = time.strftime('%H%M%S')

jobname = 'standardized_var_mlp_encoding_%s_%s' % (dstamp, tstamp)
jobfile = 'Batches/%s.job' % jobname

# Cooldown 

lam_grid = np.append(np.geomspace(3.0, 0.001, num = 50), 0)
seed_grid = [0]
hidden_grid = [10]
network_lag_grid = [1]

nepoch_grid = [5000]
lr_grid = [0.01]
cooldown_grid = ['Y']

sparsity_grid = [0.3]
p_grid = [10]
T_grid = [500]
lag_grid = [1]

BASECMD = 'python standardized_var_mlp_encoding.py'

param_grid = product(lam_grid, seed_grid, hidden_grid, network_lag_grid,
	nepoch_grid, lr_grid, cooldown_grid,
	sparsity_grid, p_grid, T_grid, lag_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		lam, seed, hidden, network_lag, nepoch, lr, cooldown, sparsity, p, T, lag = param

		argstr = BASECMD

		argstr += ' --lam=%e' % lam
		argstr += ' --seed=%d' % seed
		argstr += ' --hidden=%d' % hidden
		argstr += ' --network_lag=%d' % network_lag

		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --lr=%e' % lr
		argstr += ' --cooldown=%s' % cooldown
		
		argstr += ' --sparsity=%e' % sparsity
		argstr += ' --p=%d' % p
		argstr += ' --T=%d' % T
		argstr += ' --lag=%d' % lag

		f.write(argstr + '\n')

# No cooldown

lr_grid = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
cooldown_grid = ['N']

param_grid = product(lam_grid, seed_grid, hidden_grid, network_lag_grid,
	nepoch_grid, lr_grid, cooldown_grid,
	sparsity_grid, p_grid, T_grid, lag_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		lam, seed, hidden, network_lag, nepoch, lr, cooldown, sparsity, p, T, lag = param

		argstr = BASECMD

		argstr += ' --lam=%e' % lam
		argstr += ' --seed=%d' % seed
		argstr += ' --hidden=%d' % hidden
		argstr += ' --network_lag=%d' % network_lag

		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --lr=%e' % lr
		argstr += ' --cooldown=%s' % cooldown
		
		argstr += ' --sparsity=%e' % sparsity
		argstr += ' --p=%d' % p
		argstr += ' --T=%d' % T
		argstr += ' --lag=%d' % lag

		f.write(argstr + '\n')
