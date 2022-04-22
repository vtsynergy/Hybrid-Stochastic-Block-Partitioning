#!/bin/bash
module load OpenMPI/4.1.1-GCC-10.3.0

size=200000

for communities in 74 ; do
  for overlap in 1.0 3.0 5.0 ; do
    for blocksizevar in 1.0 ; do
      for mindegree in 1 10 ; do
        for maxdegree in 0.005 0.05 ; do
          for exponent in -2.5 -2.9 ; do
            graphprops="test_${mindegree}_${maxdegree}_${overlap}_${exponent}"
            for algorithm in "metropolis_hastings" "async_gibbs" "hybrid_mcmc" ; do
              sbatch --job-name="${graphprops}_${numasync}_0" \
                     --export=ALL,overlap=${overlap},blocksizevar=${blocksizevar},size=${size},communities=${communities},graphprops=${graphprops},algorithm=${algorithm},run=0 \
                     ./synthetic_run.sbatch
            done
          done
        done
      done
    done
  done
done
