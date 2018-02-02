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
network_lag_grid = [5]

nepoch_grid = [30000]
lr_grid = [0.01]

sparsity_grid = [0.3]
sd_grid = [0.1]
dt_grid = [0.1]
p_grid = [20]
T_grid = [500, 750, 1000]

BASECMD = 'python oscillator_mlp_encoding.py'

param_grid = product(lam_grid, seed_grid, hidden_grid, network_lag_grid,
	nepoch_grid, lr_grid,
	sparsity_grid, sd_grid, dt_grid, p_grid, T_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		lam, seed, hidden, network_lag, nepoch, lr, sparsity, sd, dt, p, T = param

		argstr = BASECMD

		argstr += ' --lam=%e' % lam
		argstr += ' --seed=%d' % seed
		argstr += ' --hidden=%d' % hidden
		argstr += ' --network_lag=%d' % network_lag

		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --lr=%e' % lr
		
		argstr += ' --sparsity=%e' % sparsity
		argstr += ' --sd=%e' % sd
		argstr += ' --dt=%e' % dt
		argstr += ' --p=%d' % p
		argstr += ' --T=%d' % T

		f.write(argstr + '\n')
