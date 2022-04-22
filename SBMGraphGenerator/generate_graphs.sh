#! /bin/bash
directory="--directory=./difficulty_graphs"

vertices=200000

for mindegree in 1 5 10 20 ; do
  for maxdegree in 0.0005 0.005 0.05 ; do  # max degree = maxdegree * vertices * offset to account for island vertices
    for exponent in -2.9 -2.5 -2.1 ; do  # -1.8 -2.1 -2.5 -3.2 ; do
      for overlap in 1.0 3.0 5.0 ; do
        tag="test_${mindegree}_${maxdegree}_${overlap}_${exponent}"
        echo "=== min: ${mindegree} max: ${maxdegree} e: ${exponent} o: ${overlap} ==="
        python generate.py -n ${vertices} -c -1 -m 0.35 -i ${mindegree} -a ${maxdegree} -e ${exponent} \
            -o ${overlap} -s 1.0 -t ${tag} ${directory}
       done
    done
  done
done

