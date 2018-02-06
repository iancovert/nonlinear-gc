from itertools import product
import numpy as np
import time

dstamp = time.strftime('%Y%m%d')
tstamp = time.strftime('%H%M%S')

jobname = 'dream_lstm_encoding_%s_%s' % (dstamp, tstamp)
jobfile = 'Batches/%s.job' % jobname

lam_grid = np.append(np.geomspace(1.0, 0.001, num = 50), 0)
seed_grid = [0]
hidden_grid = [3]

nepoch_grid = [15000]
lr_grid = [0.01]
weight_decay_grid = [0.01]

size_grid = [10]
type_grid = ['Ecoli', 'Yeast']
number_grid = [1, 2, 3]

BASECMD = 'python dream_lstm_encoding.py'

param_grid = product(lam_grid, seed_grid, hidden_grid,
	nepoch_grid, lr_grid, weight_decay_grid,
	size_grid, type_grid, number_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		lam, seed, hidden, nepoch, lr, weight_decay, size, typ, number = param

		argstr = BASECMD

		argstr += ' --lam=%e' % lam
		argstr += ' --seed=%d' % seed
		argstr += ' --hidden=%d' % hidden

		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --lr=%e' % lr
		argstr += ' --weight_decay=%e' % weight_decay
		
		argstr += ' --size=%d' % size
		argstr += ' --type=%s' % typ
		argstr += ' --number=%d' % number

		f.write(argstr + '\n')
