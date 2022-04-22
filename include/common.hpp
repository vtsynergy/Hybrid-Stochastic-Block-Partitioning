/**
 * Structs and functions common to both the block merge and finetune phases.
 */

#ifndef SBP_COMMON_HPP
#define SBP_COMMON_HPP

#include <random>
#include <vector>

// #include <Eigen/Core>

#include "blockmodel/blockmodel.hpp"
#include "blockmodel/dist_blockmodel.hpp"
#include "blockmodel/sparse/csparse_matrix.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

// TODO: move everything that uses `blockmodel` to one of the Blockmodel classes

namespace common {

// TODO: explore making these static thread_local variables? Or create an array of these, with one per thread
static std::random_device seeder;
static std::default_random_engine generator(seeder());

typedef struct new_block_degrees_t {
    std::vector<int> block_degrees_out;
    std::vector<int> block_degrees_in;
    std::vector<int> block_degrees;
} NewBlockDegrees;

/// Calculates the entropy of a single blockmodel cell.
inline float cell_entropy(float value, float degree_in, float degree_out) {
    if (value == 0.0) return 0.0;
    float entropy = value * logf(value / (degree_in * degree_out));
    /* if (std::isnan(entropy) || std::isinf(entropy)) {
        std::cerr << "value: " << value << " dIn: " << degree_in << " dOut: " << degree_out << std::endl;
        throw std::invalid_argument("something is wrong");
    } */
    return entropy;
    // return value * std::log(value / degree_in / degree_out);
}

/// TODO
int choose_neighbor(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights);

/// Chooses a neighboring block using a multinomial distribution based on the number of edges connecting the current
/// block to the neighboring blocks.
int choose_neighbor(const SparseVector<double> &multinomial_distribution);

/// TODO: computing current_block_self_edges is annoying af. Maybe use Updates and Deltas instead.
/// Computes the block degrees under a proposed move
NewBlockDegrees compute_new_block_degrees(int current_block, const Blockmodel &blockmodel, int current_block_self_edges,
                                          int proposed_block_self_edges, utils::ProposalAndEdgeCounts &proposal);

/// Computes the entropy of one row or column of data.
double delta_entropy_temp(std::vector<int> &row_or_col, std::vector<int> &block_degrees, int degree, int num_edges);

// /// Computes the entropy of one row or column of sparse data
// double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &_block_degrees, int degree);

/// Computes the entropy of one row or column of sparse data.
double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int num_edges);

// /// Computes the entropy of one row or column of sparse data, ignoring indices `current_block` and `proposal`
// double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &_block_degrees, int degree,
//                           int current_block, int proposal);

/// Computes the entropy of one row or column of sparse data, ignoring indices `current_block` and `proposal`.
double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int current_block, int proposal, int num_edges);

/// Removes entries from in whose index is index1 or index
std::vector<int> exclude_indices(const std::vector<int> &in, int index1, int index2);

/// Removes entries from in whose index is index1 or index
MapVector<int>& exclude_indices(MapVector<int> &in, int index1, int index2);

/// Returns a subset of <values> corresponding to the indices where the value of <indices> != 0
std::vector<int> index_nonzero(const std::vector<int> &values, std::vector<int> &indices);

/// Returns a subset of <values> corresponding to the indices where the value of <indices> != 0
std::vector<int> index_nonzero(const std::vector<int> &values, MapVector<int> &indices);

/// Returns the non-zero values in <in>
std::vector<int> nonzeros(std::vector<int> &in);

/// Returns the non-zero values in <in>
std::vector<int> nonzeros(MapVector<int> &in);

/// Proposes a new block for either the block merge or finetune step based on `bool block_merge`.
utils::ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                               const std::vector<int> &block_assignment, const Blockmodel &blockmodel,
                                               bool block_merge = false);
/// TODO
int propose_random_block(int current_block, int num_blocks);

/// Returns a random integer between low and high
int random_integer(int low, int high);

namespace directed {

/// Computes the entropy of one row or column of data for a directed graph.
double delta_entropy_temp(std::vector<int> &row_or_col, std::vector<int> &block_degrees, int degree);

/// Computes the entropy of one row or column of sparse data for a directed graph.
double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree);

/// Computes the entropy of one row or column of sparse data, ignoring indices `current_block` and `proposal`, for a
/// directed graph.
double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int current_block, int proposal);
}

namespace undirected {

/// Computes the entropy of one row or column of data for an undirected graph.
double delta_entropy_temp(std::vector<int> &row_or_col, std::vector<int> &block_degrees, int degree, int num_edges);

/// Computes the entropy of one row or column of sparse data for an undirected graph.
double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int num_edges);

/// Computes the entropy of one row or column of sparse data, ignoring indices `current_block` and `proposal`, for an
/// undirected graph.
double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int current_block, int proposal, int num_edges);
}

namespace dist {

// TODO: get rid of block_assignment, just use blockmodel?
utils::ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                               const std::vector<int> &block_assignment, const TwoHopBlockmodel &blockmodel,
                                               bool block_merge);

} // namespace dist

} // namespace common

#endif // SBP_COMMON_HPP
