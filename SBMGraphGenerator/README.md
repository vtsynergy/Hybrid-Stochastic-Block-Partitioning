Adapted from the SBMGraphGenerator [official repository](https://github.com/vtsynergy/SBMGraphGenerator).

Run the `generate_graphs.sh` script to generate synthetic graphs for the hybrid SBP experiments.

# SBMGraphGenerator

Graph generator with tunable parameters based on the degree-corrected stochastic blockmodel.

Notes:
- The generator is non-deterministic; it generates a different graph each time it is run.
- The graph parameters are controlled indirectly; they will not match the inputs exactly.
- For more extreme graph parameter combinations, the generator may not be able to fit a matching stochastic blockmodel and the generation may fail. Re-running the generator a few times may fix the issue, but for the most part the best solution is to use less extreme parameters.

Requirements:
- The requirements for running this generator are listed in `requirements.txt`.
- All requirements should be installable via pip, but from our experience, the `graph-tool` package can be tricky to install. The following instructions worked for our system:
```
conda config --add channels conda-forge
conda config --add channels pkgw-forge
conda install -c pkgw-forge gtk3 
conda install -c vgauthier graph-tool
```
For more information, refer to the [graph-tool installation instructions](https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions)

Usage:

To generate a graph locally, use `python generate.py <args>`. Note that generating graphs with more than a few million vertices/edges can take several hours.

!WARNING: EXPERIMENTAL! To generate a graph on a remote system, use `python generate.py --remote <remote_url> <args>`. The generator uses the [Paramiko](https://www.paramiko.org/) and [getpass](https://docs.python.org/3/library/getpass.html) packages for authentication. You will need to provide your username and password on the command-line. Systems that do not support a username + password authentication method are not supported.

&copy; Virginia Polytechnic Institute and State University, 2022.

## License

Please refer to the included [LICENSE](./LICENSE) file.
