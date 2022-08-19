#!/bin/bash
#SBATCH -p <queue>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=4:00:00
#SBATCH --account=<account>
#SBATCH --mem=MaxMemPerNode

module load OpenMPI/4.1.1-GCC-10.3.0

numasync=1000000

EXE="<path_to_executable>"  # build/HybridSBP
DIR="<graph_directory>"

/usr/bin/time -v $EXE --threads 64 -n $size -d $DIR --batches 1 -a $algorithm --transpose -o $overlap \
              -b $blocksizevar --type $graphprops --tag "${graphprops}_${numasync}_${run}" \
              --approximate --asynciterations ${numasync} --csv "./synthetic_results"

if [ $run -lt 5 ] ; then
    run=$(( $run + 1 ))
    sbatch --job-name="${graphprops}_${numasync}_${run}" \
           --export=ALL,overlap=${overlap},blocksizevar=${blocksizevar},size=${size},communities=${communities},graphprops=${graphprops},algorithm=${algorithm},run=${run} \
           ./synthetic_run.sbatch
fi

