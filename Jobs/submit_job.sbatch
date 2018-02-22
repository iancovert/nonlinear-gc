#!/bin/bash

#SBATCH --mail-user=icovert@uw.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks=1  # each job is a single task
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=512

# we specify job-specific parameters at the command line so that this script
# should be run with:

## sbatch --array=1-NUMLINES --job-name=JOBNAME
##        --output=PATH_TO_STDOUT_FILE_%A_%a.out \
##        --error=PATH_TO_STDERR_FILE_%A_%a.err \
##        --time=11:59 --export=jobfile=JOBFILE.job \
##        submit_job.sbatch
#

# set up python
module load Python/3.5.2-foss-2016b
source ${HOME}/pt-env/bin/activate
command=$(sed -n ${SLURM_ARRAY_TASK_ID}p $jobfile)
cd ${HOME}/nonlinear-gc/Experiments
${command}