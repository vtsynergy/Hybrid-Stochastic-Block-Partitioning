#!/bin/bash
module load OpenMPI/4.1.1-GCC-10.3.0

size=-1

for communities in "unk" ; do
  for overlap in "unk" ; do
    for blocksizevar in "unk" ; do
      for mindegree in "unk" ; do
        for maxdegree in "unk" ; do
          for exponent in "unk" ; do
            for run in 0 1 2 3 4 ; do
              for name in "rajat01" "barth5" "cit-HepTh" "p2p-Gnutella31" "wiki-Vote" "amazon-2008" "cnr-2000" "soc-Epinions1" "web-BerkStan" "flickr" "higgs-twitter" "soc-Slashdot0902" "Stanford_Berkeley" "amazon0505" ; do
                graphprops="${name}_${mindegree}_${maxdegree}_${overlap}_${exponent}"
                for algorithm in "metropolis_hastings" "hybrid_mcmc" ; do
                  sbatch --job-name="${graphprops}_${numasync}_${run}" \
                         --export=ALL,overlap=${overlap},blocksizevar=${blocksizevar},size=${size},communities=${communities},graphprops=${graphprops},algorithm=${algorithm},run=${run} \
                         ./real_run.sbatch
                done
              done
            done
          done
        done
      done
    done
  done
done
