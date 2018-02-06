from itertools import product
import numpy as np
import time

dstamp = time.strftime('%Y%m%d')
tstamp = time.strftime('%H%M%S')

jobname = 'lorentz_lstm_encoding_%s_%s' % (dstamp, tstamp)
jobfile = 'Batches/%s.job' % jobname

lam_grid = np.append(np.geomspace(1.0, 0.01, num = 50), 0)
seed_grid = [0]
hidden_grid = [10]

nepoch_grid = [15000]
lr_grid = [0.01]
wd_grid = [0.01]

FC_grid = [10, 40]
sd_grid = [2.0]
dt_grid = [0.05]
p_grid = [20]
T_grid = [250, 500, 750, 1000, 1250, 1500]
data_seed_grid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

BASECMD = 'python lorentz_lstm_encoding.py'

param_grid = product(T_grid, data_seed_grid,
	lam_grid, seed_grid, hidden_grid,
	nepoch_grid, lr_grid,
	FC_grid, sd_grid, dt_grid, p_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		T, data_seed, lam, seed, hidden, nepoch, lr, FC, sd, dt, p = param

		argstr = BASECMD

		argstr += ' --lam=%e' % lam
		argstr += ' --seed=%d' % seed
		argstr += ' --hidden=%d' % hidden

		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --lr=%e' % lr
		
		argstr += ' --FC=%e' % FC
		argstr += ' --sd=%e' % sd
		argstr += ' --dt=%e' % dt
		argstr += ' --p=%d' % p
		argstr += ' --T=%d' % T
		argstr += ' --data_seed=%d' % data_seed

		f.write(argstr + '\n')
