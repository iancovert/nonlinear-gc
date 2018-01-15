from itertools import product
import numpy as np
import time

dstamp = time.strftime('%Y%m%d')
tstamp = time.strftime('%H%M%S')

jobname = 'oscillator_mlp_encoding_%s_%s' % (dstamp, tstamp)
jobfile = 'Batches/%s.job' % jobname

lam_grid = np.append(np.geomspace(10.0, 0.001, num = 50), 0)
seed_grid = [0]
hidden_grid = [10]
network_lag_grid = [2]

nepoch_grid = [5000]

sparsity_grid = [0.3]
p_grid = [10]
T_grid = [250]
trials_grid = [10]

BASECMD = 'python oscillator_mlp_encoding.py'

param_grid = product(lam_grid, seed_grid, hidden_grid, network_lag_grid,
	nepoch_grid,
	sparsity_grid, p_grid, T_grid, trials_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		lam, seed, hidden, network_lag, nepoch, sparsity, p, T, trials = param

		argstr = BASECMD

		argstr += ' --lam=%e' % lam
		argstr += ' --seed=%d' % seed
		argstr += ' --hidden=%d' % hidden
		argstr += ' --network_lag=%d' % network_lag

		argstr += ' --nepoch=%d' % nepoch
		
		argstr += ' --sparsity=%e' % sparsity
		argstr += ' --p=%d' % p
		argstr += ' --T=%d' % T
		argstr += ' --trials=%d' % trials

		f.write(argstr + '\n')