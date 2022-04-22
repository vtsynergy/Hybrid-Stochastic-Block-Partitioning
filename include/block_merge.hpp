/**
 * The block merge phase of the stochastic block blockmodeling module.
 */
#ifndef CPPSBP_BLOCK_MERGE_HPP
#define CPPSBP_BLOCK_MERGE_HPP

#include <limits>
#include <numeric>
#include <random>

// #include <omp.h>
#include "common.hpp"
#include "blockmodel/blockmodel.hpp"
#include "blockmodel/dist_blockmodel.hpp"
// #include "blockmodel/sparse/boost_mapped_matrix.hpp"
#include "blockmodel/sparse/csparse_matrix.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

namespace block_merge {

static const int NUM_AGG_PROPOSALS_PER_BLOCK = 10;  // Proposals per block

typedef struct proposal_evaluation_t {
    int proposed_block;
    double delta_entropy;
} ProposalEvaluation;

/// Performs the block merges with the highest change in entropy/MDL, recalculating change in entropy before each
/// merge to account for dependencies between merges. This function modified the blockmodel.
void carry_out_best_merges_advanced(Blockmodel &blockmodel, const std::vector<double> &delta_entropy_for_each_block,
                                    const std::vector<int> &best_merge_for_each_block, int num_edges);

/// Returns the potential changes to the blockmodel if `current_block` was merged into `proposed_block`.
Delta blockmodel_delta(int current_block, int proposed_block, const Blockmodel &blockmodel);

/// Computes the new edge counts for the affected blocks (communities) under a proposed block merge.
EdgeCountUpdates edge_count_updates(std::shared_ptr<ISparseMatrix> blockmodel, int current_block, int proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks);

/// Fills the new edge counts for the affected blocks (communities) under a proposed block merge.
/// Results are stored as sparse vectors (unordered_maps)
void edge_count_updates_sparse(ISparseMatrix *blockmodel, int current_block, int proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, SparseEdgeCountUpdates &updates);

/// Merges entire blocks (communities) in blockmodel together
Blockmodel &merge_blocks(Blockmodel &blockmodel, const NeighborList &out_neighbors, int num_edges);

/// Proposes a merge for current_block based on the current blockmodel state
ProposalEvaluation propose_merge(int current_block, int num_edges, Blockmodel &blockmodel,
                                 std::vector<int> &block_assignment);

/// Proposes a merge for current_block based on the current blockmodel state, using sparse intermediate structures
ProposalEvaluation propose_merge_sparse(int current_block, int num_edges, Blockmodel &blockmodel,
                                        std::vector<int> &block_assignment,
                                        std::unordered_map<int, bool> &past_proposals);

namespace dist {

/// Merges entire blocks (communities) in blockmodel together in a distributed fashion.
TwoHopBlockmodel &merge_blocks(TwoHopBlockmodel &blockmodel, const NeighborList &out_neighbors, int num_edges);

}  // namespace dist

} // namespace block_merge

#endif // CPPSBP_BLOCK_MERGE_HPP
