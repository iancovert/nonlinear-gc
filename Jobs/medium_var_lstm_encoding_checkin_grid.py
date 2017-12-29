from itertools import product
import numpy as np
import time

dstamp = time.strftime('%Y%m%d')
tstamp = time.strftime('%H%M%S')

jobname = 'medium_var_lstm_encoding_checkin_%s_%s' % (dstamp, tstamp)
jobfile = 'Batches/%s.job' % jobname

lam_grid = [2.5, 1.0, 0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.00025, 0.0001]
nepoch_grid = [1600]
seed_grid = [0]
lr_grid = [0.005, 0.001, 0.0005, 0.0001, 0.00005]
arch_grid = [1, 2, 3]

BASECMD = 'python medium_var_lstm_encoding_checkin.py'

param_grid = product(lam_grid, nepoch_grid, seed_grid, lr_grid, arch_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		lam, nepoch, seed, lr, arch = param
		argstr = BASECMD
		argstr += ' --lam=%e' % lam
		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --seed=%d' % seed
		argstr += ' --lr=%e' % lr
		argstr += ' --arch=%d' % arch

		f.write(argstr + '\n')
