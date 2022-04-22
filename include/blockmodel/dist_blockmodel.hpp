/***
 * Stores the current distributed graph blockmodeling results.
 */
#ifndef SBP_DIST_BLOCKMODEL_HPP
#define SBP_DIST_BLOCKMODEL_HPP

#include <iostream>
#include <limits>
#include <numeric>
#include <map>
#include <memory>
#include <mpi.h>
#include <random>
#include <queue>

// #include <Eigen/Core>
// #include "sparse/boost_mapped_matrix.hpp"
#include "../args.hpp"
#include "blockmodel.hpp"
#include "../graph.hpp"
#include "mpi_data.hpp"
#include "sparse/dict_matrix.hpp"
#include "sparse/dict_transpose_matrix.hpp"
//#include "sparse/dist_dict_matrix.hpp"
#include "sparse/typedefs.hpp"
#include "../utils.hpp"

//class DistBlockmodel {
//  public:
//    DistBlockmodel() : empty(true) {}
//    DistBlockmodel(const Graph &graph, const Args &args) {
//        /// ================= TODO: move to partition() methid =====================
//        // NeighborList in_neighbors(graph.num_vertices());
//        NeighborList out_neighbors(graph.num_vertices());
//        int num_vertices = 0, num_edges = 0;
//        std::unordered_map<int, int> translator;
//        for (int i = mpi.rank; i < (int) graph.out_neighbors().size(); i += mpi.num_processes) {
//            if (utils::insert(translator, i, num_vertices))
//                num_vertices++;
//            int from = i;
//            // int from = translator[i];  // TODO: can avoid additional lookups by returning the inserted element in insert
//            for (int neighbor : graph.out_neighbors(i)) {
//                if ((neighbor % mpi.num_processes) - mpi.rank == 0) {
//                    if (utils::insert(translator, neighbor, num_vertices))
//                        num_vertices++;
//                    int to = neighbor;
//                    // int to = translator[neighbor];
//                    utils::insert(out_neighbors, from, to);
//                    // utils::insert(in_neighbors, to, from);
//                    num_edges++;
//                }
//            }
//        }
//        std::vector<int> assignment(graph.num_vertices(), -1);
//        std::vector<int> myblocks;
//        for (const std::pair<const int, int> &element : translator) {
//            myblocks.push_back(element.first);
//            assignment[element.first] = graph.assignment(element.first);
//        }
//        std::cout << "NOTE: rank " << mpi.rank << "/" << mpi.num_processes - 1 << " has N = " << num_vertices << " E = ";
//        utils::print<int>(myblocks);
//        std::cout << num_edges << std::endl;
//        this->_num_blocks = num_vertices;
//        this->_global_num_blocks = graph.num_vertices();
//        this->_assignment = assignment;
//        this->_overall_entropy = std::numeric_limits<float>::max();
//        // Number of blocks to merge
//        this->_num_blocks_to_merge = (int)(this->_num_blocks * BLOCK_REDUCTION_RATE);
//        this->initialize_edge_counts(out_neighbors, myblocks);
//    }
//    /// TODO
//    std::vector<int> build_mapping(std::vector<int> &values);
//    /// Performs the block merges with the highest change in entropy/MDL
//    void carry_out_best_merges(const std::vector<double> &delta_entropy_for_each_block,
//                               const std::vector<int> &best_merge_for_each_block);
//    /// TODO implement this if needed
//    // DistBlockmodel clone_with_true_block_membership(NeighborList &neighbors, std::vector<int> &true_block_membership);
//    /// Returns a copy of the current Blockmodel
//    DistBlockmodel copy();
//    /// TODO implement this if needed
//    // static DistBlockmodel from_sample(int num_blocks, NeighborList &neighbors, std::vector<int> &sample_block_membership,
//    //                               std::map<int, int> &mapping, float block_reduction_rate);
//    /// Uses the graph to fill out the blockmatrix and the block degree vectors.
//    // void initialize_edge_counts(const NeighborList &neighbors, const MPI &mpi, const std::vector<int> &myblocks);
//    void initialize_edge_counts(const NeighborList &neighbors, const std::vector<int> &myblocks);
//    /// TODO
//    double log_posterior_probability();
//    /// TODO
//    void merge_blocks(int from_block, int to_block);
//    /// TODO
//    void move_vertex(int vertex, int current_block, int new_block, EdgeCountUpdates &updates,
//                     std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
//                     std::vector<int> &new_block_degrees);
//    /// TODO
//    void set_block_membership(int vertex, int block);
//    /// TODO
//    void update_edge_counts(int current_block, int proposed_block, EdgeCountUpdates &updates);
//    /// TODO: Get rid of getters and setters?
//    std::vector<int> &assignment() { return this->_assignment; }
//    int &assignment(int block) { return this->_assignment[block]; }
//    void assignment(std::vector<int> block_assignment) { this->_assignment = block_assignment; }
//    ISparseMatrix *blockmatrix() { return this->_blockmatrix; }
//    std::vector<int> &degrees() { return this->_degrees; }
//    int &degrees(int block) { return this->_degrees[block]; }
//    void degrees(std::vector<int> _block_degrees) { this->_degrees = _block_degrees; }
//    std::vector<int> &degrees_in() { return this->_degrees_in; }
//    int &degrees_in(int block) { return this->_degrees_in[block]; }
//    void degrees_in(std::vector<int> _block_degrees_in) { this->_degrees_in = _block_degrees_in; }
//    std::vector<int> &degrees_out() { return this->_degrees_out; }
//    int &degrees_out(int block) { return this->_degrees_out[block]; }
//    void degrees_out(std::vector<int> _block_degrees_out) { this->_degrees_out = _block_degrees_out; }
//    // float &getBlock_reduction_rate() { return this->block_reduction_rate; }
//    // void setBlock_reduction_rate(float block_reduction_rate) { this->block_reduction_rate = block_reduction_rate; }
//    int &num_blocks_to_merge() { return this->_num_blocks_to_merge; }
//    void num_blocks_to_merge(int num) { this->_num_blocks_to_merge = num; }
//    int &num_blocks() { return this->_num_blocks; }
//    void num_blocks(int num) { this->_num_blocks = num; }
//    float &mdl() { return this->_overall_entropy; }
//    void mdl(float entropy) { this->_overall_entropy = entropy; }
//    // std::vector<int> &ownership() { return this->_ownership; }
//    // int &ownership(int block) { return this->_ownership[block]; }
//    // void ownership(std::vector<int> new_ownership) { this->_ownership = new_ownership; }
//    // Other
//    bool empty;
//    // TODO: method to sync after an iteration (rebalance blocks, etc)
//
//  protected:
//    // Structure
//    int _num_blocks;
//    int _global_num_blocks;
//    IDistSparseMatrix *_blockmatrix;
//    // Known info
//    std::vector<int> _assignment;
//    std::vector<int> _degrees;
//    std::vector<int> _degrees_in;
//    std::vector<int> _degrees_out;
//    // float block_reduction_rate;
//    // Computed info
//    float _overall_entropy;
//    int _num_blocks_to_merge;
//    /// Sorts the indices of an array in descending order according to the values of the array
//    std::vector<int> sort_indices(const std::vector<double> &unsorted);
//    // Distributed info
//    /// The MPI_rank of the process that owns each block
//    // std::vector<int> _ownership;
//    /// Syncs the ownership between all MPI processes. NOTE: assumes _num_blocks and _global_num_blocks are correctly
//    /// set already
//    // void sync_ownership(const std::vector<int> &myblocks, const MPI &mpi) {
//    //     int numblocks[mpi.num_processes];
//    //     MPI_Allgather(&(this->_num_blocks), 1, MPI_INT, &numblocks, 1, MPI_INT, MPI_COMM_WORLD);
//    //     int offsets[mpi.num_processes];
//    //     offsets[0] = 0;
//    //     for (int i = 1; i < mpi.num_processes; ++i) {
//    //         offsets[i] = offsets[i-1] + numblocks[i-1];
//    //     }
//    //     std::vector<int> allblocks(this->_global_num_blocks, -1);
//    //     MPI_Allgatherv(myblocks.data(), this->_num_blocks, MPI_INT, allblocks.data(), &(numblocks[0]), &(offsets[0]), MPI_INT, MPI_COMM_WORLD);
//    //     if (mpi.rank == 0) {
//    //         utils::print<int>(allblocks);
//    //     }
//    //     int owner = 0;
//    //     for (int i = 0; i < this->_global_num_blocks; ++i) {
//    //         if (owner < mpi.num_processes - 1 && i >= offsets[owner+1]) {
//    //             owner++;
//    //         }
//    //         this->_ownership[allblocks[i]] = owner;
//    //     }
//    //     if (mpi.rank == 0) {
//    //         utils::print<int>(this->_ownership);
//    //     }
//    // }
//};

class TwoHopBlockmodel : public Blockmodel {
  public:
    // using Blockmodel::Blockmodel;
    // Constructors are not derived from base class
    TwoHopBlockmodel() : Blockmodel() {}
    TwoHopBlockmodel(int num_blocks, float block_reduction_rate) : Blockmodel(num_blocks, block_reduction_rate) {}
    TwoHopBlockmodel(int num_blocks, const NeighborList &out_neighbors, float block_reduction_rate)
        : TwoHopBlockmodel(num_blocks, block_reduction_rate) {
        // If the block assignment is not provided, use round-robin assignment
        this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
        for (int i = mpi.rank; i < this->num_blocks; i += mpi.num_processes) {  // round-robin work mapping
            this->_my_blocks[i] = true;
        }
        this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);  // no distribution
        this->initialize_edge_counts(out_neighbors);
    }
    TwoHopBlockmodel(int num_blocks, const NeighborList &out_neighbors, float block_reduction_rate,
                     std::vector<int> &block_assignment) : TwoHopBlockmodel(num_blocks, block_reduction_rate) {
        // Set the block assignment
        this->_block_assignment = block_assignment;
        this->distribute(out_neighbors);
        this->initialize_edge_counts(out_neighbors);
    }
    /// Sets the _in_two_hop_radius for a 2-hop blockmodel.
    void build_two_hop_blockmodel(const NeighborList &neighbors);
    TwoHopBlockmodel copy();
    /// Distributes the blockmodel amongst MPI ranks. Needs to be called before the first call to
    /// initialize_edge_counts, since it sets the _in_two_hop_radius and _my_blocks vectors. After that, it only needs
    /// to be called to re-distribute the blockmodel (followed by initialize_edge_counts).
    void distribute(const NeighborList &neighbors);
    /// Returns the _in_two_hop_radius vector.
    const std::vector<bool>& in_two_hop_radius() const { return this->_in_two_hop_radius; }
    void initialize_edge_counts(const NeighborList &neighbors);
    double log_posterior_probability() const;
    /// Returns true if this blockmodel owns the compute for the requested block.
    bool owns_block(int block) const;
    /// Returns true if this blockmodel owns the compute for the requested vertex.
    bool owns_vertex(int vertex) const;
    /// Returns true if this blockmodel owns storage for the requested block.
    bool stores(int block) const;
  private:
    // ===== Functions
    /// Returns a sorted vector of <block, block size> pairs, in descending order of block size.
    std::vector<std::pair<int,int>> sorted_block_sizes() const;
    /// No data distribution, work is mapped using round-robin strategy.
    void distribute_none();
    /// 2-Hop data distribution using round-robin assignment, each MPI rank responsible for the vertices in the blocks
    /// mapped to it.
    void distribute_2hop_round_robin(const NeighborList &neighbors);
    /// 2-Hop data distribution, balanced by block size, each MPI rank responsible for the vertices in the blocks
    /// mapped to it.
    void distribute_2hop_size_balanced(const NeighborList &neighbors);
    /// 2-Hop data distribution, based on snowball sampling over vertices, each MPI rank responsible for the vertices
    /// in the blocks mapped to it.
    void distribute_2hop_snowball(const NeighborList &neighbors);
    // ===== Variables
    /// Stores true for in_two_hop_radius[block] if block is stored in this blockmodel.
    std::vector<bool> _in_two_hop_radius;
    /// Stores true for my_blocks[block] if this blockmodel owns the compute for this block.
    std::vector<bool> _my_blocks;
    /// Stores 1 for any vertex that this blockmodel owns.
    std::vector<int> _my_vertices;
};

#endif // SBP_DIST_BLOCKMODEL_HPP
