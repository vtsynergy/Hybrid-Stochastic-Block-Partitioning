#include "dist_blockmodel.hpp"

#include <unordered_set>

//std::vector<int> DistBlockmodel::build_mapping(std::vector<int> &values) {
//    std::map<int, bool> unique_map;
//    for (int i = 0; i < values.size(); ++i) {
//        unique_map[values[i]] = true;
//    }
//    std::vector<int> mapping = utils::constant<int>(values.size(), -1);
//    int counter = 0;
//    for (std::pair<int, bool> element : unique_map) {
//        mapping[element.first] = counter;
//        counter++;
//    }
//    return mapping;
//}
//
//// TODO: move to block_merge.cpp
//void DistBlockmodel::carry_out_best_merges(const std::vector<double> &delta_entropy_for_each_block,
//                                       const std::vector<int> &best_merge_for_each_block) {
//    std::vector<int> best_merges = sort_indices(delta_entropy_for_each_block);
//    std::vector<int> block_map = utils::range<int>(0, this->_num_blocks);
//    int num_merged = 0;
//    int counter = 0;
//    while (num_merged < this->_num_blocks_to_merge) {
//        int merge_from = best_merges[counter];
//        int merge_to = block_map[best_merge_for_each_block[merge_from]];
//        counter++;
//        if (merge_to != merge_from) {
//            for (int i = 0; i < block_map.size(); ++i) {
//                int block = block_map[i];
//                if (block == merge_from) {
//                    block_map[i] = merge_to;
//                }
//            }
//            this->merge_blocks(merge_from, merge_to);
//            num_merged++;
//        }
//    }
//    std::vector<int> mapping = build_mapping(this->_assignment);
//    for (int i = 0; i < this->_assignment.size(); ++i) {
//        int block = this->_assignment[i];
//        int new_block = mapping[block];
//        this->_assignment[i] = new_block;
//    }
//    this->_num_blocks -= this->_num_blocks_to_merge;
//}
//
//// DistBlockmodel DistBlockmodel::clone_with_true_block_membership(NeighborList &neighbors,
////                                                                 std::vector<int> &true_block_membership) {
////     int num_blocks = 0;
////     std::vector<int> uniques = utils::constant<int>(true_block_membership.size(), 0);
////     for (uint i = 0; i < true_block_membership.size(); ++i) {
////         int membership = true_block_membership[i];
////         uniques[membership] = 1; // mark as used
////     }
////     for (uint block = 0; block < uniques.size(); ++block) {
////         if (uniques[block] == 1) {
////             num_blocks++;
////         }
////     }
////     return DistBlockmodel(num_blocks, neighbors, true_block_membership);
//// }
//
//DistBlockmodel DistBlockmodel::copy() {
//    DistBlockmodel blockmodel_copy = DistBlockmodel();
//    blockmodel_copy._num_blocks = this->_num_blocks;
//    blockmodel_copy._global_num_blocks = this->_global_num_blocks;
//    blockmodel_copy._assignment = std::vector<int>(this->_assignment);
//    blockmodel_copy._overall_entropy = this->_overall_entropy;
//    blockmodel_copy._blockmatrix = this->_blockmatrix->copyDistSparseMatrix();
//    blockmodel_copy._degrees = std::vector<int>(this->_degrees);
//    blockmodel_copy._degrees_out = std::vector<int>(this->_degrees_out);
//    blockmodel_copy._degrees_in = std::vector<int>(this->_degrees_in);
//    blockmodel_copy._num_blocks_to_merge = 0;
//    blockmodel_copy.empty = false;
//    return blockmodel_copy;
//}
//
//// DistBlockmodel DistBlockmodel::from_sample(int num_blocks, NeighborList &neighbors, std::vector<int> &sample_block_membership,
////                                  std::map<int, int> &mapping, float block_reduction_rate) {
////     // Fill in initial block assignment
////     std::vector<int> block_assignment = utils::constant<int>(neighbors.size(), -1);
////     for (const auto &item : mapping) {
////         block_assignment[item.first] = sample_block_membership[item.second];
////     }
////     // Every unassigned block gets assigned to the next block number
////     int next_block = num_blocks;
////     for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
////         if (block_assignment[vertex] >= 0) {
////             continue;
////         }
////         block_assignment[vertex] = next_block;
////         next_block++;
////     }
////     // Every previously unassigned block gets assigned to the block it's most connected to
////     for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
////         if (block_assignment[vertex] < num_blocks) {
////             continue;
////         }
////         std::vector<int> block_counts = utils::constant<int>(num_blocks, 0);
////         // TODO: this can only handle unweighted graphs
////         std::vector<int> vertex_neighbors = neighbors[vertex];
////         for (uint i = 0; i < vertex_neighbors.size(); ++i) {
////             int neighbor = vertex_neighbors[i];
////             int neighbor_block = block_assignment[neighbor];
////             if (neighbor_block < num_blocks) {
////                 block_counts[neighbor_block]++;
////             }
////         }
////         int new_block = utils::argmax<int>(block_counts);
////         // block_counts.maxCoeff(&new_block);
////         block_assignment[vertex] = new_block;
////     }
////     return DistBlockmodel(num_blocks, neighbors, block_reduction_rate, block_assignment);
//// }
//
//void DistBlockmodel::initialize_edge_counts(const NeighborList &neighbors, const std::vector<int> &myblocks) {
//    /// TODO: this recreates the matrix (possibly unnecessary)
//    // this->_blockmatrix = new DistDictMatrix(this->_global_num_blocks, this->_global_num_blocks, mpi, myblocks);
//    this->_blockmatrix = new DistDictMatrix(this->_global_num_blocks, this->_global_num_blocks, myblocks);
//    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
//    this->_degrees_in = utils::constant<int>(this->_num_blocks, 0);
//    this->_degrees_out = utils::constant<int>(this->_num_blocks, 0);
//    // Initialize the blockmodel
//    // TODO: find a way to parallelize the matrix filling step
//    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
//        std::vector<int> vertex_neighbors = neighbors[vertex];
//        if (vertex_neighbors.size() == 0) {
//            continue;
//        }
//        int block = this->_assignment[vertex];
//        for (int i = 0; i < vertex_neighbors.size(); ++i) {
//            // Get count
//            int neighbor = vertex_neighbors[i];
//            int neighbor_block = this->_assignment[neighbor];
//            // TODO: change this once code is updated to support weighted graphs
//            int weight = 1;
//            // int weight = vertex_neighbors[i];
//            // Update blockmodel
//            if (this->_blockmatrix->stores(block))
//                this->_blockmatrix->add(block, neighbor_block, weight);
//            // Update degrees
//            this->_degrees_out[block] += weight;
//            this->_degrees_in[neighbor_block] += weight;
//        }
//    }
//    // Count block degrees
//    this->_degrees = this->_degrees_out + this->_degrees_in;
//    exit(-10);
//}
//
//double DistBlockmodel::log_posterior_probability() {
//    Indices nonzero_indices = this->_blockmatrix->nonzero();
//    std::vector<double> values = utils::to_double<int>(this->_blockmatrix->values());
//    std::vector<double> degrees_in;
//    std::vector<double> degrees_out;
//    for (uint i = 0; i < nonzero_indices.rows.size(); ++i) {
//        degrees_in.push_back(this->_degrees_in[nonzero_indices.cols[i]]);
//        degrees_out.push_back(this->_degrees_out[nonzero_indices.rows[i]]);
//    }
//    std::vector<double> temp = values * utils::nat_log<double>(
//        values / (degrees_out * degrees_in));
//    return utils::sum<double>(temp);
//}
//
//void DistBlockmodel::merge_blocks(int from_block, int to_block) {
//    for (int index = 0; index < this->_assignment.size(); ++index) {
//        if (this->_assignment[index] == from_block) {
//            this->_assignment[index] = to_block;
//        }
//    }
//};
//
//void DistBlockmodel::move_vertex(int vertex, int current_block, int new_block, EdgeCountUpdates &updates,
//                            std::vector<int> &new_block_degrees_out, std::vector<int> &new_block_degrees_in,
//                            std::vector<int> &new_block_degrees) {
//    this->_assignment[vertex] = new_block;
//    this->update_edge_counts(current_block, new_block, updates);
//    this->_degrees_out = new_block_degrees_out;
//    this->_degrees_in = new_block_degrees_in;
//    this->_degrees = new_block_degrees;
//};
//
//void DistBlockmodel::set_block_membership(int vertex, int block) { this->_assignment[vertex] = block; }
//
//std::vector<int> DistBlockmodel::sort_indices(const std::vector<double> &unsorted) {
//    // initialize original index locations
//    std::vector<int> indices = utils::range<int>(0, unsorted.size());
//
//    // sort indexes based on comparing values in unsorted
//    std::sort(indices.data(), indices.data() + indices.size(),
//              [unsorted](size_t i1, size_t i2) { return unsorted[i1] < unsorted[i2]; });
//
//    return indices;
//}
//
//void DistBlockmodel::update_edge_counts(int current_block, int proposed_block, EdgeCountUpdates &updates) {
//    this->_blockmatrix->update_edge_counts(current_block, proposed_block, updates.block_row, updates.proposal_row,
//                                         updates.block_col, updates.proposal_col);
//}

void TwoHopBlockmodel::build_two_hop_blockmodel(const NeighborList &neighbors) {
    if (args.distribute == "none") {
        this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);
        return;
    }
    if (args.distribute == "2hop-snowball") {
        this->_my_blocks = std::vector<bool>(this->num_blocks, false);
        for (int v = 0; v < (int) neighbors.size(); ++v) {
            if (this->owns_vertex(v)) {
                int b = this->block_assignment(v);
                this->_my_blocks[b] = true;
            }
        }
    }
    // I think there will be a missing block in mcmc phase vertex->neighbor->block->neighbor_block
    this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, false);
    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        std::vector<int> vertex_neighbors = neighbors[vertex];
        if (vertex_neighbors.empty()) {
            continue;
        }
        int block = this->_block_assignment[vertex];
        for (size_t i = 0; i < vertex_neighbors.size(); ++i) {
            int neighbor = vertex_neighbors[i];
            int neighbor_block = this->_block_assignment[neighbor];
            if (this->_my_blocks[block] || this->_my_blocks[neighbor_block]) {
            // if ((block % mpi.num_processes == mpi.rank) || (neighbor_block % mpi.num_processes == mpi.rank)) {
                this->_in_two_hop_radius[block] = true;
                this->_in_two_hop_radius[neighbor_block] = true;
            }
        }
    }
    int two_hop_radius_size = 0;
    for (const bool val : this->_in_two_hop_radius) {
        if (val) two_hop_radius_size++;
    }
    if (mpi.rank == 0) std::cout << "rank 0: num blocks in 2-hop radius == " << two_hop_radius_size << " / " << this->num_blocks << std::endl;
}

TwoHopBlockmodel TwoHopBlockmodel::copy() {
    TwoHopBlockmodel blockmodel_copy = TwoHopBlockmodel(this->num_blocks, this->block_reduction_rate);
    blockmodel_copy._block_assignment = std::vector<int>(this->_block_assignment);
    blockmodel_copy.overall_entropy = this->overall_entropy;
    blockmodel_copy._blockmatrix = std::shared_ptr<ISparseMatrix>(this->_blockmatrix->copy());
    blockmodel_copy._block_degrees = std::vector<int>(this->_block_degrees);
    blockmodel_copy._block_degrees_out = std::vector<int>(this->_block_degrees_out);
    blockmodel_copy._block_degrees_in = std::vector<int>(this->_block_degrees_in);
    blockmodel_copy._in_two_hop_radius = std::vector<bool>(this->_in_two_hop_radius);
    blockmodel_copy.num_blocks_to_merge = 0;
    blockmodel_copy._my_blocks = std::vector<bool>(this->_my_blocks);
    blockmodel_copy.empty = this->empty;
    return blockmodel_copy;
}

void TwoHopBlockmodel::distribute(const NeighborList &neighbors) {
    if (args.distribute == "none")
        distribute_none();
    else if (args.distribute == "2hop-round-robin")
        distribute_2hop_round_robin(neighbors);
    else if (args.distribute == "2hop-size-balanced")
        distribute_2hop_size_balanced(neighbors);
    else if (args.distribute == "2hop-snowball")
        distribute_2hop_snowball(neighbors);
    else
        distribute_2hop_snowball(neighbors);
    // // std::cout << "rank " << mpi.rank << " is load balancing!" << std::endl;
    // this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    // std::vector<std::pair<int,int>> block_sizes = this->sorted_block_sizes();
    // for (int i = mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
    //     int block = block_sizes[i].first;
    //     this->_my_blocks[block] = true;
    // }
    // for (int i = 2 * mpi.num_processes - 1 - mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
    //     int block = block_sizes[i].first;
    //     this->_my_blocks[block] = true;
    // }
    // // First pass: find out which blocks are in the 2-hop radius of my blocks
    // // I think there will be a missing block in mcmc phase vertex->neighbor->block->neighbor_block
    // this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, false);
    // for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
    //     std::vector<int> vertex_neighbors = neighbors[vertex];
    //     if (vertex_neighbors.size() == 0) {
    //         continue;
    //     }
    //     int block = this->_block_assignment[vertex];
    //     for (int i = 0; i < vertex_neighbors.size(); ++i) {
    //         int neighbor = vertex_neighbors[i];
    //         int neighbor_block = this->_block_assignment[neighbor];
    //         if ((this->_my_blocks[block] == true) || (this->_my_blocks[neighbor_block] == true)) {
    //         // if ((block % mpi.num_processes == mpi.rank) || (neighbor_block % mpi.num_processes == mpi.rank)) {
    //             this->_in_two_hop_radius[block] = true;
    //             this->_in_two_hop_radius[neighbor_block] = true;
    //         }
    //     }
    // }
    // int two_hop_radius_size = 0;
    // for (const bool val : this->_in_two_hop_radius) {
    //     if (val == true) two_hop_radius_size++;
    // }
    // if (mpi.rank == 0) std::cout << "rank 0: num blocks in 2-hop radius == " << two_hop_radius_size << " / " << this->num_blocks << std::endl;
}

void TwoHopBlockmodel::distribute_none() {
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    for (int i = mpi.rank; i < this->num_blocks; i += mpi.num_processes)
        this->_my_blocks[i] = true;
    this->_in_two_hop_radius = utils::constant<bool>(this->num_blocks, true);
}

void TwoHopBlockmodel::distribute_2hop_round_robin(const NeighborList &neighbors) {
    // Step 1: decide which blocks to own
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    for (int i = mpi.rank; i < this->num_blocks; i += mpi.num_processes)
        this->_my_blocks[i] = true;
    // Step 2: find out which blocks are in the 2-hop radius of my blocks
    build_two_hop_blockmodel(neighbors);
}

void TwoHopBlockmodel::distribute_2hop_size_balanced(const NeighborList &neighbors) {
    // Step 1: decide which blocks to own
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    std::vector<std::pair<int,int>> block_sizes = this->sorted_block_sizes();
    for (int i = mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        int block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    for (int i = 2 * mpi.num_processes - 1 - mpi.rank; i < this->num_blocks; i += 2 * mpi.num_processes) {
        int block = block_sizes[i].first;
        this->_my_blocks[block] = true;
    }
    // Step 2: find out which blocks are in the 2-hop radius of my blocks
    build_two_hop_blockmodel(neighbors);
}

void TwoHopBlockmodel::distribute_2hop_snowball(const NeighborList &neighbors) {
    // Step 1: decide which blocks to own
    this->_my_blocks = utils::constant<bool>(this->num_blocks, false);
    // std::cout << "my vertices size: " << this->_my_vertices.size() << " neighbors size: " << neighbors.size() << std::endl;
    if (this->_my_vertices.size() == neighbors.size()) {  // if already done sampling, no need to do it again
        std::cout << "already done sampling, now just re-assigning blocks based on sampled vertices" << std::endl;
        for (size_t vertex = 0; vertex < neighbors.size(); ++vertex) {
            if (this->_my_vertices[vertex] == 0) continue;
            int block = this->_block_assignment[vertex];
            this->_my_blocks[block] = true;
        }
    } else {
        int target = ceil((double) neighbors.size() / (double) mpi.num_processes);
        this->_my_vertices = utils::constant<int>(neighbors.size(), 0);  // cannot send vector<bool>.data() over MPI
        std::unordered_set<int> frontier;
        // Snowball Sampling
        srand(mpi.num_processes + mpi.rank);
        int start = rand() % neighbors.size();  // replace this with a proper int distribution
        std::cout << "rank: " << mpi.rank << " with start = " << start << std::endl;
        this->_my_vertices[start] = 1;
        for (int neighbor : neighbors[start]) {
            frontier.insert(neighbor);
        }
        int block = this->_block_assignment[start];
        this->_my_blocks[block] = true;
        int num_vertices = 1;
        while (num_vertices < target) {
            std::unordered_set<int> new_frontier;
            for (int vertex : frontier) {
                if (this->_my_vertices[vertex] == 1) continue;
                this->_my_vertices[vertex] = 1;
                for (int neighbor : neighbors[vertex]) {
                    new_frontier.insert(neighbor);
                }
                block = this->_block_assignment[vertex];
                this->_my_blocks[block] = true;
                num_vertices++;
                if (num_vertices == target) break;
            }
            if (num_vertices < target && frontier.size() == 0) {  // restart with a new vertex that isn't already selected
                std::unordered_set<int> candidates;
                for (int i = 0; i < (int) neighbors.size(); ++i) {
                    if (this->_my_vertices[i] == 0) candidates.insert(i);
                }
                int index = rand() % candidates.size();
                auto it = candidates.begin();
                std::advance(it, index);
                start = *it;
                this->_my_vertices[start] = 1;
                for (int neighbor : neighbors[start]) {
                    new_frontier.insert(neighbor);
                }
                block = this->_block_assignment[start];
                this->_my_blocks[block] = true;
                num_vertices++;
            }
            frontier = std::unordered_set<int>(new_frontier);
        }
        // Some vertices may be unassigned across all ranks. Find out what they are, and assign 1/num_processes of them
        // to this process.
        std::vector<int> global_selected(neighbors.size(), 0);
        MPI_Allreduce(this->_my_vertices.data(), global_selected.data(), neighbors.size(), MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        // if (mpi.rank == 0) {
            // std::cout << "my selected: " << std::boolalpha;
            // utils::print<int>(this->_my_vertices);
            // std::cout << "globally selected: ";
            // utils::print<int>(global_selected);
        // }
        std::vector<int> vertices_left;
        for (int vertex = 0; vertex < (int) global_selected.size(); ++vertex) {
            if (global_selected[vertex] == 0) {
                vertices_left.push_back(vertex);
            }
        }
        // assign remaining vertices in round-robin fashion
        for (size_t i = mpi.rank; i < vertices_left.size(); i += mpi.num_processes) {
            int vertex = vertices_left[i];
            this->_my_vertices[vertex] = 1;
            block = this->_block_assignment[vertex];
            this->_my_blocks[block] = true;
        }
    }
    // Step 2: find out which blocks are in the 2-hop radius of my blocks
    this->build_two_hop_blockmodel(neighbors);
}

void TwoHopBlockmodel::initialize_edge_counts(const NeighborList &neighbors) {
    /// TODO: this recreates the matrix (possibly unnecessary)
    if (args.transpose) {
        this->_blockmatrix = std::make_shared<DictTransposeMatrix>(this->num_blocks, this->num_blocks);
    } else {
        this->_blockmatrix = std::make_shared<DictMatrix>(this->num_blocks, this->num_blocks);
    }
    // This may or may not be faster with push_backs. TODO: test init & fill vs push_back
    this->_block_degrees_in = utils::constant<int>(this->num_blocks, 0);
    this->_block_degrees_out = utils::constant<int>(this->num_blocks, 0);

    for (uint vertex = 0; vertex < neighbors.size(); ++vertex) {
        std::vector<int> vertex_neighbors = neighbors[vertex];
        if (vertex_neighbors.empty()) {
            continue;
        }
        int block = this->_block_assignment[vertex];
        if (!this->_in_two_hop_radius[block]) {
            continue;
        }
        for (size_t i = 0; i < vertex_neighbors.size(); ++i) {
            // Get count
            int neighbor = vertex_neighbors[i];
            int neighbor_block = this->_block_assignment[neighbor];
            if (!this->_in_two_hop_radius[neighbor_block]) {
                continue;
            }
            // TODO: change this once code is updated to support weighted graphs
            int weight = 1;
            // Update blockmodel
            this->_blockmatrix->add(block, neighbor_block, weight);
            // Update degrees
            this->_block_degrees_out[block] += weight;
            this->_block_degrees_in[neighbor_block] += weight;
        }
    }
    // Count block degrees
    if (args.undirected) {
        this->_block_degrees = std::vector<int>(this->_block_degrees_out);
    } else {
        this->_block_degrees = this->_block_degrees_out + this->_block_degrees_in;
    }
}

double TwoHopBlockmodel::log_posterior_probability() const {
    std::vector<int> my_blocks;
    if (args.distribute == "2hop-snowball") {
        my_blocks = utils::constant<int>(this->num_blocks, -1);
        for (int block = 0; block < this->num_blocks; ++block) {
            if (this->_my_blocks[block])
                my_blocks[block] = mpi.rank;
        }
        MPI_Allreduce(MPI_IN_PLACE, my_blocks.data(), this->num_blocks, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        // utils::print<int>(my_blocks);
    }
    Indices nonzero_indices = this->_blockmatrix->nonzero();
    std::vector<double> all_values = utils::to_double<int>(this->_blockmatrix->values());
    std::vector<double> degrees_in;
    std::vector<double> degrees_out;
    std::vector<double> values;
    for (uint i = 0; i < nonzero_indices.rows.size(); ++i) {
        int row = nonzero_indices.rows[i];
        if (args.distribute == "2hop-snowball") {
            if (my_blocks[row] != mpi.rank) continue;
        } else {
            if (this->_my_blocks[row] == false) continue;
        }
        // if (row % mpi.num_processes != mpi.rank) continue;
        values.push_back(all_values[i]);
        degrees_in.push_back(this->_block_degrees_in[nonzero_indices.cols[i]]);
        degrees_out.push_back(this->_block_degrees_out[nonzero_indices.rows[i]]);
    }
    std::vector<double> temp = values * utils::nat_log<double>(
        values / (degrees_out * degrees_in));
    double partial_sum = utils::sum<double>(temp);
    // MPI COMMUNICATION START
    double final_sum = 0.0;
    MPI_Allreduce(&partial_sum, &final_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // MPI COMMUNICATION END
    // Alternative Plan for sampled 2-hop blockmodel:
    // 1. Break all_values, degrees_in, degrees_out into row-like statuses
    // 2. Compute temp across the rows that you own
    // 3. Perform an AllReduce MAX to find missing values
    // 4. Sum across the rows to find the final_sum
    return final_sum;
}

bool TwoHopBlockmodel::owns_block(int block) const {
    return this->_my_blocks[block];
}

bool TwoHopBlockmodel::owns_vertex(int vertex) const {
    if (args.distribute == "2hop-snowball") {
        return this->_my_vertices[vertex];
    }
    int block = this->_block_assignment[vertex];
    return this->owns_block(block);
}

std::vector<std::pair<int,int>> TwoHopBlockmodel::sorted_block_sizes() const {
    std::vector<std::pair<int,int>> block_sizes;
    for (int i = 0; i < this->num_blocks; ++i) {
        block_sizes.push_back(std::make_pair(i, 0));
    }
    for (const int &block : this->_block_assignment) {
        block_sizes[block].second++;
    }
    std::sort(block_sizes.begin(), block_sizes.end(),
              [](const std::pair<int, int> &a, const std::pair<int, int> &b) { return a.second > b.second; });
    return block_sizes;
}

bool TwoHopBlockmodel::stores(int block) const {
    return this->_in_two_hop_radius[block];
}
