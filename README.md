# Hybrid Stochastic Block Partitioning

Contains code for running the Stochastic Block Partitioning (SBP) algorithm, as well as its 
Asynchronous and Hybrid variants. 

The SBP algorithm is adapted to C++ from the 
[Streaming Graph Challenge: Stochastic Block Partitioning](https://graphchallenge.mit.edu/challenges).

## Notes

- A sample build script is provided in `scripts/biuld.sh`.
- Sample run scripts are provided in `scripts/synthetic_experiment.sh` and `scripts/real_experiment.sh`.
- A sample figure generating script is provided in `scripts/figures.py`
- A script for generating the synthetic datasets is provided in `SBMGraphGenerator/generate_graphs.sh`. 
The graph generator is lightly adapted from [SBMGraphGenerator](https://github.com/vtsynergy/SBMGraphGenerator).
- Synthetic datasets can be downloaded from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/). Download the Matrix Market versions of
the datasets, and delete the first few lines of the file, containing the comments and graph size
information. In other words, the dataset file should only contain an edge list.
- Datasets are expected to be stored in a specific directory structure:
`<parent>/<tag>/<overlap>Overlap_<var>BlockSizeVar/`, where `<parent>` is the contents of 
`--directory`, and `<tag>` is in the form `<min_deg>_<max_deg>_<overlap>_<power_law_exponent>`.
The dataset itself is expected to be named `<name>_<tag>_<overlap>Overlap_<var>BlockSizeVar_<size>_nodes.tsv`.
  - For real-world datasets, everything but `<name>` and `<size>` can be set to `unk`.
- The current version of the code only supports unweighted directed graphs.
- When the number of communities discovered is 1, the evaluation will output NMI as `nan`. This is
a minor bug in the evaluation code; in such situations, the real value of NMI is 0.
- When there is no ground truth available, all evaluation metrics other than Modularity and
MDL-related metrics are invalid.

# License

Please refer to the included [LICENSE](./LICENSE) file.
