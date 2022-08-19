#include "blockmodel.hpp"

#include "assert.h"

#include "../args.hpp"
#include "typedefs.hpp"
#include "utils.hpp"

double Blockmodel::block_size_variation() const {
    // Normalized using variance / max_variance, where max_variance = range^2 / 4
    // See: https://link.springer.com/content/pdf/10.1007/BF00143817.pdf
    std::vector<int> block_sizes(this->num_blocks, 0);
    for (int block : this->_block_assignment) {
        block_sizes[block]++;
    }
    double total = utils::sum<int>(block_sizes);
    double mean = total / double(this->num_blocks);
    double min = std::numeric_limits<double>::max(), max = std::numeric_limits<double>::min(), variance = 0;
    for (int block_size : block_sizes) {
        if (block_size < min) min = block_size;
        if (block_size > max) max = block_size;
        variance += double(block_size - mean) * double(block_size - mean);
    }
    variance /= double(this->num_blocks);
    double max_variance = (double(max - min) * double(max - mean)) / 4.0;
    return float(variance / max_variance);
}

std::vector<int> Blockmodel::build_mapping(const std::vector<int> &values) {
    std::map<int, bool> unique_map;
    for (size_t i = 0; i < values.size(); ++i) {
        unique_map[values[i]] = true;
    }
    std::vector<int> mapping = utils::constant<int>((int) values.size(), -1);
    int counter = 0;
    for (std::pair<int, bool> element : unique_map) {
        mapping[element.first] = counter;
        counter++;
    }
    return mapping;
}

double Blockmodel::difficulty_score() const {
    double norm_variance = this->block_size_variation();
    double interblock_edges = this->interblock_edges();
    return (2.0 * norm_variance * interblock_edges) / (norm_variance + interblock_edges);
}

// TODO: move to block_merge.cpp
void Blockmodel::carry_out_best_merges(const std::vector<double> &delta_entropy_for_each_block,
                                       const std::vector<int> &best_merge_for_each_block) {
    std::vector<int> best_merges = utils::sort_indices(delta_entropy_for_each_block);
    std::vector<int> block_map = utils::range<int>(0, this->num_blocks);
    int num_merged = 0;
    int counter = 0;
    while (num_merged < this->num_blocks_to_merge) {
        int merge_from = best_merges[counter];
        int merge_to = block_map[best_merge_for_each_block[merge_from]];
        counter++;
        if (merge_to != merge_from) {
            for (size_t i = 0; i < block_map.size(); ++i) {
                int block = block_map[i];
                if (block == merge_from) {
                    block_map[i] = merge_to;
                }
            }
            this->update_block_assignment(merge_from, merge_to);
            num_merged++;
        }
    }
    std::vector<int> mapping = build_mapping(this->_block_assignment);
    for (size_t i = 0; i < this->_block_assignment.size(); ++i) {
        int block = this->_block_assignment[i];
        int new_block = mapping[block];
        this->_block_assignment[i] = new_block;
    }
    this->num_blocks -= this->num_blocks_to_merge;
}

Blockmodel Blockmodel::clone_with_true_block_membership(const Graph &graph, std::vector<int> &true_block_membership) {
    int num_blocks = 0;
    std::vector<int> uniques = utils::constant<int>(true_block_membership.size(), 0);
    for (uint i = 0; i < true_block_membership.size(); ++i) {
        int membership = true_block_membership[i];
        uniques[membership] = 1; // mark as used
    }
    for (uint block = 0; block < uniques.size(); ++block) {
        if (uniques[block] == 1) {
            num_blocks++;
        }
    }
    return Blockmodel(num_blocks, graph, this->block_reduction_rate, true_block_membership);
}

Blockmodel Blockmodel::copy() {
    Blockmodel blockmodel_copy = Blockmodel(this->num_blocks, this->block_reduction_rate);
    blockmodel_copy._block_assignment = std::vector<int>(this->_block_assignment);
    blockmodel_copy.overall_entropy = this->overall_entropy;
    blockmodel_copy._blockmatrix = std::shared_ptr<ISparseMatrix>(this->_blockmatrix->copy());
    blockmodel_copy._block_degrees = std::vector<int>(this->_block_degrees);
    blockmodel_copy._block_degrees_out = std::vector<int>(this->_block_degrees_out);
    blockmodel_copy._block_degrees_in = std::vector<int>(this->_block_degrees_in);
    blockmodel_copy.num_blocks_to_merge = 0;
    return blockmodel_copy;
}

Blockmodel Blockmodel::from_sample(int num_blocks, const Graph &graph, std::vector<int> &sample_block_membership,
                                 std::map<int, int> &mapping, float block_reduction_rate) {
    // Fill in initial block assignment
    std::vector<int> _block_assignment = utils::constant<int>(graph.num_vertices(), -1);  // neighbors.size(), -1);
    for (const auto &item : mapping) {
        _block_assignment[item.first] = sample_block_membership[item.second];
    }
    // Every unassigned block gets assigned to the next block number
    int next_block = num_blocks;
    for (uint vertex = 0; vertex < graph.num_vertices(); ++vertex) {  // neighbors.size(); ++vertex) {
        if (_block_assignment[vertex] >= 0) {
            continue;
        }
        _block_assignment[vertex] = next_block;
        next_block++;
    }
    // Every previously unassigned block gets assigned to the block it's most connected to
    for (uint vertex = 0; vertex < graph.num_vertices(); ++vertex) {  // neighbors.size(); ++vertex) {
        if (_block_assignment[vertex] < num_blocks) {
            continue;
        }
        std::vector<int> block_counts = utils::constant<int>(num_blocks, 0);
        // TODO: this can only handle unweighted graphs
        std::vector<int> vertex_neighbors = graph.out_neighbors(vertex);  // [vertex];
        for (uint i = 0; i < vertex_neighbors.size(); ++i) {
            int neighbor = vertex_neighbors[i];
            int neighbor_block = _block_assignment[neighbor];
            if (neighbor_block < num_blocks) {
                block_counts[neighbor_block]++;
            }
        }
        int new_block = utils::argmax<int>(block_counts);
        // block_counts.maxCoeff(&new_block);
        _block_assignment[vertex] = new_block;
    }
    return Blockmodel(num_blocks, graph, block_reduction_rate, _block_assignment);
}

//void Blockmodel::initialize_edge_counts(const NeighborList &neighbors) {
//    double start = omp_get_wtime();
////    std::cout << "OLD BLOCKMODEL BOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO" << std::endl;
//    /// TODO: this recreates the matrix (possibly unnecessary)
//    if (args.transpose) {
//        this->_blockmatrix = std::make_shared<DictTransposeMatrix>(this->num_blocks, this->num_blocks);
//    } else {
//        this->_blockmatrix = std::make_shared<DictMatrix>(this->num_blocks, this->num_blocks);
//    }
//    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
//    this->_block_degrees_in = utils::constant<int>(this->num_blocks, 0);
//    this->_block_degrees_out = utils::constant<int>(this->num_blocks, 0);
//    this->_block_degrees = utils::constant<int>(this->num_blocks, 0);
//    // Initialize the blockmodel
//    // TODO: find a way to parallelize the matrix filling step
//    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
//        std::vector<int> vertex_neighbors = neighbors[vertex];
//        if (vertex_neighbors.empty()) {
//            continue;
//        }
//        int block = this->_block_assignment[vertex];
//        for (size_t i = 0; i < vertex_neighbors.size(); ++i) {
//            // Get count
//            int neighbor = vertex_neighbors[i];
//            int neighbor_block = this->_block_assignment[neighbor];
//            // TODO: change this once code is updated to support weighted graphs
//            int weight = 1;
//            // int weight = vertex_neighbors[i];
//            // Update blockmodel
//            this->_blockmatrix->add(block, neighbor_block, weight);
//            // Update degrees
//            this->_block_degrees_out[block] += weight;
//            this->_block_degrees_in[neighbor_block] += weight;
//            this->_block_degrees[block] += weight;
//            if (block != neighbor_block)
//                this->_block_degrees[neighbor_block] += weight;
//        }
//    }
//    double end = omp_get_wtime();
//    std::cout << omp_get_thread_num() << "Matrix creation walltime = " << end - start << std::endl;
//}

void Blockmodel::initialize_edge_counts(const Graph &graph) {  // Parallel version!
//    double start = omp_get_wtime();
//    std::cout << "OLD BLOCKMODEL BOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO" << std::endl;
    /// TODO: this recreates the matrix (possibly unnecessary)
    std::shared_ptr<ISparseMatrix> blockmatrix;
    if (args.transpose) {
        blockmatrix = std::make_shared<DictTransposeMatrix>(this->num_blocks, this->num_blocks);
    } else {
        blockmatrix = std::make_shared<DictMatrix>(this->num_blocks, this->num_blocks);
    }
    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
    std::vector<int> block_degrees_in = utils::constant<int>(this->num_blocks, 0);
    std::vector<int> block_degrees_out = utils::constant<int>(this->num_blocks, 0);
    std::vector<int> block_degrees = utils::constant<int>(this->num_blocks, 0);
    // Initialize the blockmodel
    // TODO: find a way to parallelize the matrix filling step
    #pragma omp parallel default(none) \
    shared(blockmatrix, block_degrees_in, block_degrees_out, block_degrees, graph, std::cout, args)
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int my_num_blocks = ceil(double(this->num_blocks) / double(num_threads));
        int start = my_num_blocks * tid;
        int end = start + my_num_blocks;
        for (uint vertex = 0; vertex < graph.num_vertices(); ++vertex) {
//            std::vector<int> vertex_neighbors = graph.out_neighbors(vertex);  // neighbors[vertex];
//            if (vertex_neighbors.empty()) {
//                continue;
//            }
            int block = this->_block_assignment[vertex];
            if (block < start || block >= end)  // only modify blocks this thread is responsible for
                continue;
            for (int neighbor : graph.out_neighbors(int(vertex))) {  // vertex_neighbors) {
//                size_t i = 0; i < vertex_neighbors.size(); ++i) {
                // Get count
//                int neighbor = vertex_neighbors[i];
                int neighbor_block = this->_block_assignment[neighbor];
                // TODO: change this once code is updated to support weighted graphs
                int weight = 1;
                // int weight = vertex_neighbors[i];
                // Update blockmodel
                blockmatrix->add(block, neighbor_block, weight);
                // Update degrees
                block_degrees_out[block] += weight;
                block_degrees[block] += weight;
//            }
            }
            for (int neighbor : graph.in_neighbors(int(vertex))) {
                int neighbor_block = this->_block_assignment[neighbor];
                int weight = 1;
                if (args.transpose) {
                    std::shared_ptr<DictTransposeMatrix> blockmatrix_dtm = std::dynamic_pointer_cast<DictTransposeMatrix>(blockmatrix);
                    blockmatrix_dtm->add_transpose(neighbor_block, block, weight);
                }
                // Update degrees
                block_degrees_in[block] += weight;
                if (block != neighbor_block) {
                    block_degrees[block] += weight;
                }
            }
        }
    }  // OMP_PARALLEL
    this->_blockmatrix = std::move(blockmatrix);
    this->_block_degrees_out = std::move(block_degrees_out);
    this->_block_degrees_in = std::move(block_degrees_in);
    this->_block_degrees = std::move(block_degrees);
//    double end = omp_get_wtime();
//    std::cout << omp_get_thread_num() << "Matrix creation walltime = " << end - start << std::endl;
}

double Blockmodel::interblock_edges() const {
    double num_edges = utils::sum<int>(this->_block_degrees_in);
    double interblock_edges = num_edges - double(this->_blockmatrix->trace());
    return interblock_edges / num_edges;
}

bool Blockmodel::is_neighbor_of(int block1, int block2) const {
    return this->blockmatrix()->get(block1, block2) + this->blockmatrix()->get(block2, block1) > 0;
}

double Blockmodel::log_posterior_probability() const {
    Indices nonzero_indices = this->_blockmatrix->nonzero();
    std::vector<double> values = utils::to_double<int>(this->_blockmatrix->values());
    std::vector<double> degrees_in;
    std::vector<double> degrees_out;
    for (uint i = 0; i < nonzero_indices.rows.size(); ++i) {
        degrees_in.push_back(this->_block_degrees_in[nonzero_indices.cols[i]]);
        degrees_out.push_back(this->_block_degrees_out[nonzero_indices.rows[i]]);
    }
    for (uint i = 0; i < values.size(); ++i) {
        if (degrees_in[i] == 0.0 || degrees_out[i] == 0.0) {
            std::cout << "value: " << values[i] << " degree_in: " << degrees_in[i] << " degree_out: " << degrees_out[i] << std::endl;
            exit(-1000);
        }
    }
    std::vector<double> temp = values * utils::nat_log<double>(
        values / (degrees_out * degrees_in));
    return utils::sum<double>(temp);
}

double Blockmodel::log_posterior_probability(int num_edges) const {
    if (args.undirected) {
        Indices nonzero_indices = this->_blockmatrix->nonzero();
        std::vector<double> values = utils::to_double<int>(this->_blockmatrix->values());
        std::vector<double> degrees_in;
        std::vector<double> degrees_out;
        for (uint i = 0; i < nonzero_indices.rows.size(); ++i) {
            // This is OK bcause _block_degrees_in == _block_degrees_out == _block_degrees
            degrees_in.push_back(this->_block_degrees_in[nonzero_indices.cols[i]] / (2.0));
            degrees_out.push_back(this->_block_degrees_out[nonzero_indices.rows[i]] / (2.0));
        }
        std::vector<double> temp = values * utils::nat_log<double>(
            (values / (2.0)) / (degrees_out * degrees_in));
        double result = 0.5 * utils::sum<double>(temp);
        return result;
    }
    return log_posterior_probability();
}

void Blockmodel::update_block_assignment(int from_block, int to_block) {
    for (size_t index = 0; index < this->_block_assignment.size(); ++index) {
        if (this->_block_assignment[index] == from_block) {
            this->_block_assignment[index] = to_block;
        }
    }
}

void Blockmodel::move_vertex(int vertex, int current_block, int new_block, EdgeCountUpdates &updates,
                             std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                             std::vector<int> &new_block_degrees) {
    this->_block_assignment[vertex] = new_block;
    this->update_edge_counts(current_block, new_block, updates);
    this->_block_degrees_out = new_block_degrees_out;
    this->_block_degrees_in = new_block_degrees_in;
    this->_block_degrees = new_block_degrees;
}

void Blockmodel::move_vertex(int vertex, int current_block, int new_block, SparseEdgeCountUpdates &updates,
                             std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                             std::vector<int> &new_block_degrees) {
    this->_block_assignment[vertex] = new_block;
    this->update_edge_counts(current_block, new_block, updates);
    this->_block_degrees_out = new_block_degrees_out;
    this->_block_degrees_in = new_block_degrees_in;
    this->_block_degrees = new_block_degrees;
}

void Blockmodel::move_vertex(int vertex, int new_block, const Delta &delta,
                             std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
                             std::vector<int> &new_block_degrees) {
    this->_block_assignment[vertex] = new_block;
    this->_blockmatrix->update_edge_counts(delta);
    this->_block_degrees_out = new_block_degrees_out;
    this->_block_degrees_in = new_block_degrees_in;
    this->_block_degrees = new_block_degrees;
}

void Blockmodel::move_vertex(int vertex, const Delta &delta, utils::ProposalAndEdgeCounts &proposal) {
    this->_block_assignment[vertex] = proposal.proposal;
    this->_blockmatrix->update_edge_counts(delta);
    int current_block = delta.current_block();
    int current_block_self_edges = this->_blockmatrix->get(current_block, current_block)
                                   + delta.get(current_block, current_block);
    int proposed_block_self_edges = this->_blockmatrix->get(proposal.proposal, proposal.proposal)
                                    + delta.get(proposal.proposal, proposal.proposal);
    this->_block_degrees_out[current_block] -= proposal.num_out_neighbor_edges;
    this->_block_degrees_out[proposal.proposal] += proposal.num_out_neighbor_edges;
    this->_block_degrees_in[current_block] -= (proposal.num_in_neighbor_edges + delta.self_edge_weight());
    this->_block_degrees_in[proposal.proposal] += (proposal.num_in_neighbor_edges + delta.self_edge_weight());
    this->_block_degrees[current_block] = this->_block_degrees_out[current_block] +
            this->_block_degrees_in[current_block] - current_block_self_edges;
    this->_block_degrees[proposal.proposal] = this->_block_degrees_out[proposal.proposal] +
            this->_block_degrees_in[proposal.proposal] - proposed_block_self_edges;
}

void Blockmodel::print_blockmatrix() const {
    this->_blockmatrix->print();
}

void Blockmodel::print_blockmodel() const {
    std::cout << "Blockmodel: " << std::endl;
    this->print_blockmatrix();
    std::cout << "Block degrees out: ";
    utils::print<int>(this->_block_degrees_out);
    std::cout << "Block degrees in: ";
    utils::print<int>(this->_block_degrees_in);
    std::cout << "Block degrees: ";
    utils::print<int>(this->_block_degrees);
    std::cout << "Assignment: ";
    utils::print<int>(this->_block_assignment);
}

void Blockmodel::set_block_membership(int vertex, int block) { this->_block_assignment[vertex] = block; }

void Blockmodel::update_edge_counts(int current_block, int proposed_block, EdgeCountUpdates &updates) {
    this->_blockmatrix->update_edge_counts(current_block, proposed_block, updates.block_row, updates.proposal_row,
                                           updates.block_col, updates.proposal_col);
}

void Blockmodel::update_edge_counts(int current_block, int proposed_block, SparseEdgeCountUpdates &updates) {
    this->_blockmatrix->update_edge_counts(current_block, proposed_block, updates.block_row, updates.proposal_row,
                                           updates.block_col, updates.proposal_col);
}

bool Blockmodel::validate(const Graph &graph) {
    Blockmodel correct(this->num_blocks, graph, this->block_reduction_rate, this->_block_assignment);

    for (int row = 0; row < this->num_blocks; ++row) {
        for (int col = 0; col < this->num_blocks; ++col) {
//            int this_val = this->blockmatrix()->get(row, col);
            int correct_val = correct.blockmatrix()->get(row, col);
            if (!this->blockmatrix()->validate(row, col, correct_val)) return false;
//            if (this_val != correct_val) return false;
        }
    }
    return true;
}
