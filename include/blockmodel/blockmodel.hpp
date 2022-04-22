/***
 * Stores the current graph blockmodeling results.
 */
#ifndef SBP_DELTA_HPP
#define SBP_DELTA_HPP

#include <iostream>
#include <limits>
#include <numeric>
#include <map>
#include <memory>
#include <queue>

// #include <Eigen/Core>
// #include "sparse/boost_mapped_matrix.hpp"
#include "../args.hpp"
#include "delta.hpp"
#include "sparse/dict_matrix.hpp"
#include "sparse/dict_transpose_matrix.hpp"
#include "sparse/typedefs.hpp"
#include "../utils.hpp"
#include "typedefs.hpp"
#include "utils.hpp"

static const double BLOCK_REDUCTION_RATE = 0.5;

// typedef py::EigenDRef<Eigen::Matrix<int, Eigen::Dynamic, 2>> Matrix2Column;

typedef struct edge_count_updates_t {
    std::vector<int> block_row;
    std::vector<int> proposal_row;
    std::vector<int> block_col;
    std::vector<int> proposal_col;
} EdgeCountUpdates;

typedef struct sparse_edge_count_updates_t {
    MapVector<int> block_row;
    MapVector<int> proposal_row;
    MapVector<int> block_col;
    MapVector<int> proposal_col;
} SparseEdgeCountUpdates;

// TODO: make a Blockmodel interface (?) Or keep Blockmodel pointers in memory
class Blockmodel {
  public:
    Blockmodel() : empty(true) {}
    Blockmodel(int num_blocks, float block_reduction_rate) : empty(false) {
        this->num_blocks = num_blocks;
        this->block_reduction_rate = block_reduction_rate;
        this->overall_entropy = std::numeric_limits<float>::max();
        if (args.transpose) {
            this->_blockmatrix = std::make_shared<DictTransposeMatrix>(this->num_blocks, this->num_blocks);
        } else {
            this->_blockmatrix = std::make_shared<DictMatrix>(this->num_blocks, this->num_blocks);
        }
        // Set the block assignment to be the range [0, this->num_blocks)
        this->_block_assignment = utils::range<int>(0, this->num_blocks);

        // Number of blocks to merge
        this->num_blocks_to_merge = (int)(this->num_blocks * this->block_reduction_rate);
    }
    Blockmodel(int num_blocks, const NeighborList &out_neighbors, float block_reduction_rate)
        : Blockmodel(num_blocks, block_reduction_rate) {
        this->initialize_edge_counts(out_neighbors);
    }
    Blockmodel(int num_blocks, const NeighborList &out_neighbors, float block_reduction_rate,
               std::vector<int> &block_assignment) : Blockmodel(num_blocks, block_reduction_rate) {
        // Set the block assignment
        this->_block_assignment = block_assignment;
        // Number of blocks to merge
        this->initialize_edge_counts(out_neighbors);
    }
    /// TODO
    static std::vector<int> build_mapping(const std::vector<int> &values) ;
    /// Performs the block merges with the highest change in entropy/MDL
    void carry_out_best_merges(const std::vector<double> &delta_entropy_for_each_block,
                               const std::vector<int> &best_merge_for_each_block);
    /// TODO
    Blockmodel clone_with_true_block_membership(NeighborList &neighbors, std::vector<int> &true_block_membership);
    /// Returns a copy of the current Blockmodel
    Blockmodel copy();
    /// TODO documentation
    // TODO: move block_reduction_rate to some constants file
    static Blockmodel from_sample(int num_blocks, NeighborList &neighbors, std::vector<int> &sample_block_membership,
                                 std::map<int, int> &mapping, float block_reduction_rate);
    /// Returns the normalized difference in block sizes.
    double block_size_variation() const;
    /// Difficulty score, being the geometric mean between block_size_variation() and interblock_edges().
    double difficulty_score() const;
    /// TODO
    void initialize_edge_counts(const NeighborList &neighbors);
    /// TODO
    double log_posterior_probability() const;
    /// TODO
    double log_posterior_probability(int num_edges) const;
    /// Moves `vertex` from `current_block` to `new_block`. Updates the blockmodel using the new rows and columns from
    /// `updates`, and updates the block degrees.
    /// TODO: update block degrees on the fly.
    void move_vertex(int vertex, int current_block, int new_block, EdgeCountUpdates &updates,
                     std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                     std::vector<int> &new_block_degrees);
    /// Moves `vertex` from `current_block` to `new_block`. Updates the blockmodel using the new rows and columns from
    /// `updates`, and updates the block degrees.
    /// TODO: update block degrees on the fly.
    void move_vertex(int vertex, int current_block, int new_block, SparseEdgeCountUpdates &updates,
                     std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                     std::vector<int> &new_block_degrees);
    /// Moves `vertex` from `current_block` to `new_block`. Updates the blockmodel using the new blockmodel values from
    /// `delta`, and updates the block degrees.
    /// TODO: update block degrees on the fly.
    void move_vertex(int vertex, int new_block, const Delta &delta, std::vector<int> &new_block_degrees_out,
                     std::vector<int> &new_block_degrees_in, std::vector<int> &new_block_degrees);
    /// Moves `vertex` from one block to another. Updates the blockmodel using the new blockmodel values from `delta`,
    /// and updates the block degrees, which are calculated on-the-fly.
    void move_vertex(int vertex, const Delta &delta, utils::ProposalAndEdgeCounts &proposal);
    /// TODO
    void set_block_membership(int vertex, int block);
    /// TODO: Get rid of getters and setters?
    std::shared_ptr<ISparseMatrix> blockmatrix() const { return this->_blockmatrix; }
//    ISparseMatrix *blockmatrix() const { return this->_blockmatrix; }
    /// Returns an immutable copy of the vertex-to-block assignment vector.
    const std::vector<int> &block_assignment() const { return this->_block_assignment; }
    /// Returns the block assignment for `vertex`.
    int block_assignment(int vertex) const { return this->_block_assignment[vertex]; }
    /// Returns true if `block1` is a neighbor of `block2`.
    bool is_neighbor_of(int block1, int block2) const;
    /// Returns the percentage of edges occurring between blocks.
    double interblock_edges() const;
    /// Prints blockmatrix to file (should not be used for large blockmatrices)
    void print_blockmatrix() const;
    /// Prints the blockmodel with some additional information.
    void print_blockmodel() const;
    /// Returns true if the blockmodel owns the current block (always returns true for non-distributed blockmodel).
    bool stores(int block) const { return true; }
    /// TODO
    void update_block_assignment(int from_block, int to_block);
    /// Updates the blockmodel values for `current_block` and `proposed_block` using the rows and columns in `updates`.
    void update_edge_counts(int current_block, int proposed_block, EdgeCountUpdates &updates);
    /// Updates the blockmodel values for `current_block` and `proposed_block` using the rows and columns in `updates`.
    void update_edge_counts(int current_block, int proposed_block, SparseEdgeCountUpdates &updates);
    /// Validates the blockmatrix entries given the current block assignment.
    bool validate(const NeighborList &out_neighbors);
    /// Sets the block assignment for this `vertex` to `block`.
    void set_block_assignment(int vertex, int block) { this->_block_assignment[vertex] = block; }
    void set_block_assignment(std::vector<int> block_assignment) { this->_block_assignment = block_assignment; }
    const std::vector<int> &degrees() const { return this->_block_degrees; }
    int degrees(int block) const { return this->_block_degrees[block]; }
    void degrees(int block, int value) { this->_block_degrees[block] = value; }
    void degrees(std::vector<int> block_degrees) { this->_block_degrees = block_degrees; }
    const std::vector<int> &degrees_in() const { return this->_block_degrees_in; }
    int degrees_in(int block) const { return this->_block_degrees_in[block]; }
    void degrees_in(int block, int value) { this->_block_degrees_in[block] = value; }
    void degrees_in(std::vector<int> block_degrees_in) { this->_block_degrees_in = block_degrees_in; }
    const std::vector<int> &degrees_out() const { return this->_block_degrees_out; }
    int degrees_out(int block) const { return this->_block_degrees_out[block]; }
    void degrees_out(int block, int value) { this->_block_degrees_out[block] = value; }
    void degrees_out(std::vector<int> block_degrees_out) { this->_block_degrees_out = block_degrees_out; }
    float &getBlock_reduction_rate() { return this->block_reduction_rate; }
    void setBlock_reduction_rate(float block_reduction_rate) { this->block_reduction_rate = block_reduction_rate; }
    float &getOverall_entropy() { return this->overall_entropy; }
    void setOverall_entropy(double overall_entropy) { this->overall_entropy = overall_entropy; }
    int &getNum_blocks_to_merge() { return this->num_blocks_to_merge; }
    void setNum_blocks_to_merge(int num_blocks_to_merge) { this->num_blocks_to_merge = num_blocks_to_merge; }
    int getNum_blocks() const { return this->num_blocks; }
    void setNum_blocks(int num_blocks) { this->num_blocks = num_blocks; }
    // Other
    bool empty;

  protected:
    // Structure
    int num_blocks;
    std::shared_ptr<ISparseMatrix> _blockmatrix;
//    ISparseMatrix *_blockmatrix;
    // Known info
    std::vector<int> _block_assignment;
    std::vector<int> _block_degrees;
    std::vector<int> _block_degrees_in;
    std::vector<int> _block_degrees_out;
    float block_reduction_rate;
    // Computed info
    float overall_entropy;
    int num_blocks_to_merge;
    /// Sorts the indices of an array in descending order according to the values of the array
    std::vector<int> sort_indices(const std::vector<double> &unsorted);
};

#endif // SBP_DELTA_HPP
