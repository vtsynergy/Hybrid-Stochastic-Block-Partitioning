/**
 * The finetuning phase of the stochastic block blockmodeling algorithm.
 */
#ifndef SBP_FINETUNE_HPP
#define SBP_FINETUNE_HPP

#include <cmath>
#include <vector>

#include <omp.h>

#include "common.hpp"
#include "graph.hpp"
#include "blockmodel/blockmodel.hpp"
#include "blockmodel/dist_blockmodel.hpp"
#include "blockmodel/blockmodel_triplet.hpp"
#include "blockmodel/sparse/delta.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

/*******************
 * FINE-TUNE
 ******************/
namespace finetune {

extern int MCMC_iterations;

typedef struct vertex_move_t {
    double delta_entropy;
    bool did_move;
    int vertex;
    int proposed_block;
} VertexMove;

//static const int MOVING_AVG_WINDOW = 3;      // Window for calculating change in entropy
//static const double SEARCH_THRESHOLD = 5e-4; // Threshold before golden ratio is established
//static const double GOLDEN_THRESHOLD = 1e-4; // Threshold after golden ratio is established
static const int MAX_NUM_ITERATIONS = 100;   // Maximum number of finetuning iterations

bool accept(double delta_entropy, double hastings_correction);

Blockmodel &asynchronous_gibbs(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels);

EdgeWeights block_edge_weights(const std::vector<int> &block_assignment, EdgeWeights &neighbor_weights);

/// Returns the potential changes to the blockmodel if the vertex with `out_edges` and `in_edges` moves from
/// `current_block` into `proposed_block`.
/// NOTE: assumes that any self edges are present in exactly one of `out_edges` and `in_edges`.
Delta blockmodel_delta(int vertex, int current_block, int proposed_block, const EdgeWeights &out_edges,
                       const EdgeWeights &in_edges, const Blockmodel &blockmodel);

bool early_stop(int iteration, BlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies);

bool early_stop(int iteration, double initial_entropy, std::vector<double> &delta_entropies);

[[maybe_unused]] EdgeCountUpdates edge_count_updates(ISparseMatrix *blockmodel, int current_block, int proposed_block,
                                                     EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                                     int self_edge_weight);

void edge_count_updates_sparse(const Blockmodel &blockmodel, int vertex, int current_block, int proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, SparseEdgeCountUpdates &updates);

/// Returns the edge weights in `neighbors[vertex]` as an `EdgeWeights` struct. If `ignore_self` is `true`, then
/// self-edges will not be added to EdgeWeights.
EdgeWeights edge_weights(const NeighborList &neighbors, int vertex, bool ignore_self = false);

/// Evaluates a potential move of `vertex` from `current_block` to `proposal.proposal` using MCMC logic.
VertexMove eval_vertex_move(int vertex, int current_block, utils::ProposalAndEdgeCounts proposal,
                            const Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
                            EdgeWeights &in_edges);

/// Evaluates a potential move of `vertex` from `current_block` to `proposal.proposal` using MCMC logic without using
/// blockmodel deltas.
VertexMove eval_vertex_move_nodelta(int vertex, int current_block, utils::ProposalAndEdgeCounts proposal,
                                    const Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
                                    EdgeWeights &in_edges);

/// Runs the synchronous Metropolis Hastings algorithm on the high-degree vertices of `blockmodel`, and
/// Asynchronous Gibbs on the rest.
Blockmodel &hybrid_mcmc(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels);

[[maybe_unused]] Blockmodel &finetune_assignment(Blockmodel &blockmodel, Graph &graph);

/// Runs the synchronous Metropolis Hastings algorithm on `blockmodel`.
Blockmodel &metropolis_hastings(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels);

/// Moves `vertex` from `current_block` to `proposal.proposal` using MCMC logic.
VertexMove move_vertex(int vertex, int current_block, utils::ProposalAndEdgeCounts proposal, Blockmodel &blockmodel,
                       const Graph &graph, EdgeWeights &out_edges, EdgeWeights &in_edges);

/// Moves `vertex` from `current_block` to `proposal.proposal` using MCMC logic without using blockmodel deltas.
VertexMove move_vertex_nodelta(int vertex, int current_block, utils::ProposalAndEdgeCounts proposal,
                               Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
                               EdgeWeights &in_edges);

/// Computes the overall entropy of the given blockmodel.
//double mdl(const Blockmodel &blockmodel, int num_vertices, int num_edges);

/// Proposes a new Metropolis-Hastings vertex move.
VertexMove propose_move(Blockmodel &blockmodel, int vertex, const Graph &graph);

/// Proposes a new Asynchronous Gibbs vertex move.
VertexMove propose_gibbs_move(const Blockmodel &blockmodel, int vertex, const Graph &graph);

//namespace directed {
//
///// Computes the overall entropy of the given blockmodel for a directed graph.
//double overall_entropy(const Blockmodel &blockmodel, int num_vertices, int num_edges);
//
//}  // namespace directed
//
//namespace undirected {
//
///// Computes the overall entropy of the given blockmodel for an undirected graph.
//double overall_entropy(const Blockmodel &blockmodel, int num_vertices, int num_edges);
//
//}  // namespace undirected

namespace dist {

/// Runs the Asynchronous Gibbs algorithm in a distributed fashion using MPI.
TwoHopBlockmodel &asynchronous_gibbs(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels);

/// If the average of the last 3 delta entropies is < threshold * initial_entropy, stop the algorithm.
bool early_stop(int iteration, DistBlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies);

/// Runs the Metropolis Hastings algorithm in a distributed fashion using MPI.
TwoHopBlockmodel &metropolis_hastings(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels);

/// Proposes an asynchronous Gibbs move in a distributed setting.
VertexMove propose_gibbs_move(const TwoHopBlockmodel &blockmodel, int vertex, const Graph &graph);

/// Proposes a metropolis hastings move in a distributed setting.
VertexMove propose_mh_move(TwoHopBlockmodel &blockmodel, int vertex, const Graph &graph);

}  // namespace dist

} // namespace finetune

#endif // SBP_FINETUNE_HPP
