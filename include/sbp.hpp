/**
 * The stochastic block blockmodeling module.
 */
#ifndef SBP_SBP_HPP
#define SBP_SBP_HPP

#include <omp.h>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "blockmodel/blockmodel_triplet.hpp"
#include "blockmodel/dist_blockmodel.hpp"
#include "graph.hpp"

namespace sbp {

/// Stores intermediate information for later printing.
struct Intermediate {
    float iteration;
    double mdl;
    double normalized_mdl_v1;
    double normalized_mdl_v2;
    double modularity;
    double interblock_edges;
    double block_size_variation;
    int mcmc_iterations;
    double mcmc_time;
};

std::vector<Intermediate> get_intermediates();

/// Performs community detection on the provided graph, using the stochastic block partitioning algorithm
Blockmodel stochastic_block_partition(Graph &graph, Args &args);

/// Returns true if the exit condition is reached based on the provided blockmodels
bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, int min_num_blocks = 0);

namespace dist {

/// Performs community detection on the provided graph using MPI, using the stochastic block partitioning algorithm
Blockmodel stochastic_block_partition(Graph &graph, Args &args);

/// Returns true if the exit condition is reached based on the provided distributed blockmodels
bool done_blockmodeling(TwoHopBlockmodel &blockmodel, DistBlockmodelTriplet &blockmodel_triplet,
                        int min_num_blocks = 0);

} // namespace dist

} // namespace sbp

#endif // SBP_SBP_HPP
