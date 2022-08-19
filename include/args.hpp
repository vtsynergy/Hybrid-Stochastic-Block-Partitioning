/// A wrapper around tclap to make the use of command-line arguments bearable.
#ifndef TCLAP_WRAPPER_ARGS
#define TCLAP_WRAPPER_ARGS

#include <iostream>
#include <limits.h>
#include <omp.h>
#include <string>

#include "tclap/CmdLine.h"

// TODO: add a verbose argument
class Args {
    public:  // Everything in here is public, because why not?
    /** Define all the arguments here for easy access */
    std::string algorithm;
    bool approximate;
    int asynciterations;
    int batches;
    std::string blocksizevar;
    std::string csv;  // TODO: save results in a csv file
    std::string delimiter;
    std::string distribute;
    std::string directory;
    bool nodelta;  // TODO: if delta is much faster, get rid of this and associated methods.
    int numvertices;
    std::string overlap;
    std::string partition;
    float samplesize;
    std::string samplingalg;
    std::string tag;
    int threads;
    bool transpose;
    std::string type;
    bool undirected;

    /// Default Args constructor.
    Args() {}

    /// Parses the command-line options and stores the results in an easy-to-retrieve manner.
    Args(int argc, char** argv) {
        /** Use tclap to retrieve the arguments here */
        try {
            TCLAP::CmdLine parser("Stochastic block blockmodeling algorithm", ' ', "alpha.0.1");
            TCLAP::ValueArg<std::string> _algorithm("a", "algorithm", "The algorithm to use for the finetuning/MCMC "
                                                    "step of stochastic block blockmodeling. Note: there is currently no "
                                                    "parallel implementation of metropolis hastings", false,
                                                    "async_gibbs", "async_gibbs|metropolis_hastings|hybrid_mcmc", parser);
            TCLAP::SwitchArg _approximate("", "approximate", "If set, an approximate version of the block merge "
                                          "step will be used. It's slightly faster, but less accurate for complex "
                                          "graphs.", parser, false);
            TCLAP::ValueArg<int> _async_iterations("", "asynciterations", "The number of asynchronous iterations to "
                                                   "run before switching to metropolis hastings.", false,
                                                   std::numeric_limits<int>::max(), "[1, infinity]", parser);
            TCLAP::ValueArg<int> _batches("", "batches", "The number of batches to use for the asynchronous_gibbs "
                                          "algorithm. Too many batches will lead to many updates and little parallelism,"
                                          " but too few will lead to poor results or more iterations", false, 10, "int",
                                          parser);
            TCLAP::ValueArg<std::string> _blocksizevar("b", "blocksizevar", "The variation between the sizes of "
                                                       "communities", false, "low", "low|high|unk", parser);
            TCLAP::ValueArg<std::string> _csv("c", "csv",
                                              "The path to the csv file in which the results will be stored, "
                                              "without the suffix, e.g.:\n"
                                              "if --csv=eval/test, results will be stored in eval/test.csv.",
                                              false, "./eval/test", "path", parser);
            TCLAP::ValueArg<std::string> _delimiter("", "delimiter", "The delimiter used in the file storing the graph",
                                                    false, "\t", "string, usually `\\t` or `,`", parser);
            TCLAP::ValueArg<std::string> _distribute("", "distribute", "The distribution scheme to use. Default = "
                                                     "2hop-snowball", false, "2hop-snowball", "none | 2hop-round-robin "
                                                     "| 2hop-size-balanced | 2hop-snowball", parser);
            TCLAP::ValueArg<std::string> _directory("d", "directory",
                                "The directory in which the graph is stored. The following structure is assumed:\n"
                                "filename for graph:"
                                "<type>_<overlap>Overlap_<blocksizevar>BlockSizeVar_<numvertices>_nodes.tsv\n"
                                "filename for truth:"
                                "<type>_<overlap>Overlap_<blocksizevar>BlockSizeVar_<numvertices>_trueBlockmodel.tsv\n"
                                "directory structure:"
                                "<directory>/<type>/<overlap>Overlap_<blocksizevar>BlockSizeVar/<filename>\n",
                                                   false, "./data", "path", parser);
            TCLAP::SwitchArg _nodelta("", "nodelta", "If set, do not use the blockmodel deltas for "
                                      "entropy calculations.", parser, false);
            TCLAP::ValueArg<int> _numvertices("n", "numvertices", "The number of vertices in the graph", false, 1000,
                                              "int", parser);
            TCLAP::ValueArg<std::string> _overlap("o", "overlap", "The degree of overlap between communities", false,
                                                  "low", "low|high|unk", parser);
            TCLAP::ValueArg<std::string> _partition("p", "partition", "The type of partitioning to use to divide the "
                                                    "graph amongst the MPI Processes. Only matters when nprocs > 1",
                                                    false, "round_robin", "round_robin|random|snowball", parser);
            TCLAP::ValueArg<float> _samplesize("", "samplesize", "The percentage of vertices to include in the sample",
                                               false, 1.0, "0 < x <= 1.0", parser);
            TCLAP::ValueArg<std::string> _samplingalg("", "samplingalg", "The sampling algorithm to use, if --samplesize < 1.0",
                                                      false, "random", "random|max_degree|expansion_snowball", parser);
            TCLAP::ValueArg<std::string> _tag("", "tag", "The tag value for this run, for differentiating different "
                                              "runs or adding custom parameters to the save file", false, "default tag",
                                              "string or param1=value1;param2=value2", parser);
            TCLAP::ValueArg<int> _threads("", "threads", "The number of OpenMP threads to use. If less than 1, will set "
                                          "number of threads to number of logical CPU cores", false, 0, "int", parser);
            TCLAP::SwitchArg _transpose("", "transpose", "If set, will also store the matrix transpose for faster column"
                                        "indexing", parser, false);
            TCLAP::ValueArg<std::string> _type("t", "type", "The type of streaming/name of the graph", false, "static",
                                               "string", parser);
            TCLAP::SwitchArg _undirected("", "undirected", "If set, graph will be treated as undirected", parser,
                                         false);
            parser.parse(argc, argv);
            this->algorithm = _algorithm.getValue();
            this->approximate = _approximate.getValue();
            this->asynciterations = _async_iterations.getValue();
            this->batches = _batches.getValue();
            this->blocksizevar = _blocksizevar.getValue();
            this->csv = _csv.getValue();
            this->delimiter = _delimiter.getValue();
            this->distribute = _distribute.getValue();
            this->directory = _directory.getValue();
            this->nodelta = _nodelta.getValue();
            this->numvertices = _numvertices.getValue();
            this->overlap = _overlap.getValue();
            this->partition = _partition.getValue();
            this->samplesize = _samplesize.getValue();
            this->samplingalg = _samplingalg.getValue();
            this->tag = _tag.getValue();
            this->threads = _threads.getValue();
            this->transpose = _transpose.getValue();
            this->type = _type.getValue();
            this->undirected = _undirected.getValue();
        } catch (TCLAP::ArgException &exception) {
            std::cerr << "ERROR: " << exception.error() << " for argument " << exception.argId() << std::endl;
            exit(-1);
        }
    }
};

extern Args args;

#endif // TCLAP_WRAPPER_ARGS
