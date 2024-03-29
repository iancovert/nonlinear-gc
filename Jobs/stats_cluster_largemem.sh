#!/bin/bash

jobfile=$1
jobnameTemp=${jobfile%.*}
jobname=${jobnameTemp##*/}
njobs=$(wc -l ${jobfile} | cut -d" " -f1)

mkdir -p out/${jobname}
mkdir -p checkpoints/${jobname}

logdir=logs/${jobname}
mkdir -p ${logdir}

outfile=${logdir}/${jobname}_%A_%a.out
errfile=${logdir}/${jobname}_%A_%a.err

if [ $njobs -gt 999 ]
then
    split -d -l 999 $jobfile $jobnameTemp
    
    for jfile in $(ls ${jobnameTemp}[0-9][0-9]*)
    do
        jnjobs=$(wc -l $jfile | cut -d" " -f1)
        sbatch --array=1-${jnjobs} --job-name=${jfile} \
               --output=${outfile} \
               --error=${errfile} \
               --partition=largemem --export=jobfile=${jfile} \
               --time=3-23:59:00 \
               submit_job.sbatch
		echo "Submitted a job file..."
        sleep 60
    done
else
    # just submit the single job file because we can
    sbatch --array=1-${njobs} --job-name=${jobname} \
           --output=${outfile} \
           --error=${errfile} \
           --partition=largemem --export=jobfile=${jobfile} \
           --time=3-23:59:00 \
           submit_job.sbatch
	echo "Submitted job file"
fi

echo "Done submitting"
