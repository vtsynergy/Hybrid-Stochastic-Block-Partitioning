#include "finetune.hpp"

#include "args.hpp"
#include "entropy.hpp"
#include "mpi_data.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

#include <cassert>
#include <fenv.h>
#include <iostream>

namespace finetune {

int MCMC_iterations = 0;
std::ofstream my_file;

bool accept(double delta_entropy, double hastings_correction) {
    // fedisableexcept(FE_INVALID | FE_OVERFLOW);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double random_probability = distribution(common::generator);
    // std::cout << "dE: " << delta_entropy << " hastings correction: " << hastings_correction << std::endl;
    // TODO: if delta_entropy > X (or less than?) return true (accept the move)
    double accept_probability = exp(-3.0 * delta_entropy) * hastings_correction;
    accept_probability = (accept_probability >= 1.0) ? 1.0 : accept_probability;
    // feenableexcept(FE_INVALID | FE_OVERFLOW);
    return random_probability <= accept_probability;
}

Blockmodel &asynchronous_gibbs(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels) {
    std::cout << "Asynchronous Gibbs iteration" << std::endl;
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
    double initial_entropy = blockmodel.getOverall_entropy();

    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double delta_entropy = 0.0;
        double num_batches = args.batches;
        int batch_size = int(ceil(graph.num_vertices() / num_batches));
        for (int batch = 0; batch < graph.num_vertices() / batch_size; ++batch) {
            int start = batch * batch_size;
            int end = std::min(graph.num_vertices(), (batch + 1) * batch_size);
            // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
            // asynchronous Gibbs sampling
            std::vector<int> block_assignment(blockmodel.block_assignment());
            #pragma omp parallel for schedule(dynamic) default(none) \
            shared(start, end, blockmodel, graph, vertex_moves, delta_entropy, block_assignment)
            for (int vertex = start; vertex < end; ++vertex) {
                VertexMove proposal = propose_gibbs_move(blockmodel, vertex, graph);
                if (proposal.did_move) {
                    vertex_moves++;
                    delta_entropy += proposal.delta_entropy;
                    block_assignment[vertex] = proposal.proposed_block;
                }
            }
            blockmodel = Blockmodel(blockmodel.getNum_blocks(), graph.out_neighbors(),
                                    blockmodel.getBlock_reduction_rate(), block_assignment);
        }
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy / initial_entropy << std::endl;
        total_vertex_moves += vertex_moves;
        MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, blockmodels, initial_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

EdgeWeights block_edge_weights(const std::vector<int> &block_assignment, EdgeWeights &neighbor_weights) {
    std::map<int, int> block_counts;
    for (uint i = 0; i < neighbor_weights.indices.size(); ++i) {
        int neighbor = neighbor_weights.indices[i];
        int block = block_assignment[neighbor];
        int weight = neighbor_weights.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    std::vector<int> blocks;
    std::vector<int> weights;
    for (auto const &entry: block_counts) {
        blocks.push_back(entry.first);
        weights.push_back(entry.second);
    }
    return EdgeWeights{blocks, weights};
}

Delta blockmodel_delta(int vertex, int current_block, int proposed_block, const EdgeWeights &out_edges,
                       const EdgeWeights &in_edges, const Blockmodel &blockmodel) {
    Delta delta(current_block, proposed_block, int(std::max(out_edges.indices.size(), in_edges.indices.size())));

    // current_block -> current_block == proposed_block --> proposed_block  (this includes self edges)
    // current_block --> other_block == proposed_block --> other_block
    // other_block --> current_block == other_block --> proposed_block
    // current_block --> proposed_block == proposed_block --> proposed_block
    // proposed_block --> current_block == proposed_block --> proposed_block
    for (size_t i = 0; i < out_edges.indices.size(); ++i) {
        int out_vertex = out_edges.indices[i];
        int out_block = blockmodel.block_assignment(out_vertex);
        int edge_weight = out_edges.values[i];
        if (vertex == out_vertex) {
            delta.add(proposed_block, proposed_block, edge_weight);
            delta.self_edge_weight(1);
        } else {
            delta.add(proposed_block, out_block, edge_weight);
        }
        delta.sub(current_block, out_block, edge_weight);
    }
    for (size_t i = 0; i < in_edges.indices.size(); ++i) {
        int in_vertex = in_edges.indices[i];
        int in_block = blockmodel.block_assignment(in_vertex);
        int edge_weight = in_edges.values[i];
        if (vertex == in_vertex) {
            delta.add(proposed_block, proposed_block, edge_weight);
            delta.self_edge_weight(1);
        } else {
            delta.add(in_block, proposed_block, edge_weight);
        }
        delta.sub(in_block, current_block, edge_weight);
    }
    return delta;
}

bool early_stop(int iteration, BlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies) {
    size_t last_index = delta_entropies.size() - 1;
    if (delta_entropies[last_index] == 0.0) {
        return true;
    }
    if (iteration < 3) {
        return false;
    }
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold;
    if (blockmodels.get(2).empty) { // Golden ratio bracket not yet established
        threshold = 5e-4 * initial_entropy;
    } else {
        threshold = 1e-4 * initial_entropy;
    }
    return average < threshold;
}

bool early_stop(int iteration, double initial_entropy, std::vector<double> &delta_entropies) {
    if (iteration < 3) {
        return false;
    }
    size_t last_index = delta_entropies.size() - 1;
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold = 1e-4 * initial_entropy;
    return average < threshold;
}

[[maybe_unused]] EdgeCountUpdates edge_count_updates(ISparseMatrix *blockmodel, int current_block, int proposed_block,
                                                     EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                                     int self_edge_weight) {
    std::vector<int> block_row = blockmodel->getrow(current_block);
    std::vector<int> block_col = blockmodel->getcol(current_block);
    std::vector<int> proposal_row = blockmodel->getrow(proposed_block);
    std::vector<int> proposal_col = blockmodel->getcol(proposed_block);

    int count_in_block = 0, count_out_block = 0;
    int count_in_proposal = self_edge_weight, count_out_proposal = self_edge_weight;

    for (uint i = 0; i < in_blocks.indices.size(); ++i) {
        int index = in_blocks.indices[i];
        int value = in_blocks.values[i];
        if (index == current_block) {
            count_in_block += value;
        }
        if (index == proposed_block) {
            count_in_proposal += value;
        }
        block_col[index] -= value;
        proposal_col[index] += value;
    }
    for (uint i = 0; i < out_blocks.indices.size(); ++i) {
        int index = out_blocks.indices[i];
        int value = out_blocks.values[i];
        if (index == current_block) {
            count_out_block += value;
        }
        if (index == proposed_block) {
            count_out_proposal += value;
        }
        block_row[index] -= value;
        proposal_row[index] += value;
    }

    proposal_row[current_block] -= count_in_proposal;
    proposal_row[proposed_block] += count_in_proposal;
    proposal_col[current_block] -= count_out_proposal;
    proposal_col[proposed_block] += count_out_proposal;

    block_row[current_block] -= count_in_block;
    block_row[proposed_block] += count_in_block;
    block_col[current_block] -= count_out_block;
    block_col[proposed_block] += count_out_block;

    return EdgeCountUpdates{block_row, proposal_row, block_col, proposal_col};
}

// TODO: remove double counts from the edge count updates? But then we'll have to figure out how to correctly update
// the blockmodel since we won't be able to do bm.column[block/proposal] = updates.block/proposal_col
void edge_count_updates_sparse(const Blockmodel &blockmodel, int vertex, int current_block, int proposed_block,
                               EdgeWeights &out_edges, EdgeWeights &in_edges, SparseEdgeCountUpdates &updates) {
    updates.block_row = blockmodel.blockmatrix()->getrow_sparse(current_block);
    updates.block_col = blockmodel.blockmatrix()->getcol_sparse(current_block);
    updates.proposal_row = blockmodel.blockmatrix()->getrow_sparse(proposed_block);
    updates.proposal_col = blockmodel.blockmatrix()->getcol_sparse(proposed_block);

    for (size_t i = 0; i < out_edges.indices.size(); ++i) {
        int out_vertex = out_edges.indices[i];
        int out_block = blockmodel.block_assignment(out_vertex);
        int edge_weight = out_edges.values[i];
        if (vertex == out_vertex) {
            updates.proposal_row[proposed_block] += edge_weight;
            updates.proposal_col[proposed_block] += edge_weight;
        } else {
            updates.proposal_row[out_block] += edge_weight;
            if (out_block == proposed_block)
                updates.proposal_col[proposed_block] += edge_weight;
            if (out_block == current_block)
                updates.block_col[proposed_block] += edge_weight;
        }
        updates.block_row[out_block] -= edge_weight;
        if (out_block == current_block)
            updates.block_col[current_block] -= edge_weight;
        if (out_block == proposed_block)
            updates.proposal_col[current_block] -= edge_weight;
    }
    for (size_t i = 0; i < in_edges.indices.size(); ++i) {
        int in_vertex = in_edges.indices[i];
        int in_block = blockmodel.block_assignment(in_vertex);
        int edge_weight = in_edges.values[i];
        if (vertex == in_vertex) {
            updates.proposal_col[proposed_block] += edge_weight;
            updates.proposal_row[proposed_block] += edge_weight;
        } else {
            updates.proposal_col[in_block] += edge_weight;
            if (in_block == proposed_block)
                updates.proposal_row[proposed_block] += edge_weight;
            if (in_block == current_block)
                updates.block_row[proposed_block] += edge_weight;
        }
        updates.block_col[in_block] -= edge_weight;
        if (in_block == current_block)
            updates.block_row[current_block] -= edge_weight;
        if (in_block == proposed_block)
            updates.proposal_row[current_block] -= edge_weight;
    }
}

EdgeWeights edge_weights(const NeighborList &neighbors, int vertex, bool ignore_self) {
    std::vector<int> indices;
    std::vector<int> values;
    // Assumes graph is unweighted
    const std::vector<int> &neighbor_vector = neighbors[vertex];
    for (const int neighbor: neighbor_vector) {
        if (ignore_self && neighbor == vertex) continue;
        indices.push_back(neighbor);
        values.push_back(1);
    }
//    for (int row = 0; row < neighbor_vector.size(); ++row) {
//        indices.push_back(neighbor_vector[row]);
//        values.push_back(1);
//    }
    return EdgeWeights{indices, values};
}

VertexMove eval_vertex_move(int vertex, int current_block, utils::ProposalAndEdgeCounts proposal,
                            const Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
                            EdgeWeights &in_edges) {
    if (args.nodelta)
        return eval_vertex_move_nodelta(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
    const Delta delta = blockmodel_delta(vertex, current_block, proposal.proposal, out_edges, in_edges, blockmodel);
    double hastings = entropy::hastings_correction(vertex, graph, blockmodel, delta, current_block, proposal);
    double delta_entropy = entropy::delta_mdl(blockmodel, delta, proposal);

    if (accept(delta_entropy, hastings))
        return VertexMove{delta_entropy, true, vertex, proposal.proposal};
    return VertexMove{delta_entropy, false, -1, -1};
}

VertexMove eval_vertex_move_nodelta(int vertex, int current_block, utils::ProposalAndEdgeCounts proposal,
                                    const Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
                                    EdgeWeights &in_edges) {
    EdgeWeights blocks_out_neighbors = block_edge_weights(blockmodel.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = block_edge_weights(blockmodel.block_assignment(), in_edges);
    SparseEdgeCountUpdates updates;
    edge_count_updates_sparse(blockmodel, vertex, current_block, proposal.proposal, out_edges, in_edges, updates);
    int current_block_self_edges = updates.block_row[current_block];
    int proposed_block_self_edges = updates.proposal_row[proposal.proposal];
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, blockmodel, current_block_self_edges, proposed_block_self_edges, proposal);
    double hastings =
            entropy::hastings_correction(blockmodel, blocks_out_neighbors, blocks_in_neighbors, proposal, updates,
                                         new_block_degrees);
    double delta_entropy =
            entropy::delta_mdl(current_block, proposal.proposal, blockmodel, graph.num_edges(), updates,
                               new_block_degrees);
    if (accept(delta_entropy, hastings))
        return VertexMove{delta_entropy, true, vertex, proposal.proposal};
    return VertexMove{delta_entropy, false, -1, -1};
}

Blockmodel &hybrid_mcmc(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels) {
    std::cout << "Hybrid MCMC iteration" << std::endl;
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
    double initial_entropy = blockmodel.getOverall_entropy();
    double num_batches = args.batches;
//        int batch_size = int(ceil(graph.num_vertices() / num_batches));
    int num_low_degree_vertices = int(graph.low_degree_vertices().size());
    int batch_size = int(ceil(num_low_degree_vertices / num_batches));

    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double delta_entropy = 0.0;
        for (int vertex : graph.high_degree_vertices()) {
            VertexMove proposal = propose_move(blockmodel, vertex, graph);
            if (proposal.did_move) {
                vertex_moves++;
                delta_entropy += proposal.delta_entropy;
            }
        }
        for (int batch = 0; batch < num_low_degree_vertices / batch_size; ++batch) {
            int start = batch * batch_size;
            int end = std::min(num_low_degree_vertices, (batch + 1) * batch_size);
            // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
            // asynchronous Gibbs sampling
            std::vector<int> block_assignment(blockmodel.block_assignment());
            #pragma omp parallel for schedule(dynamic) default(none) \
            shared(start, end, blockmodel, graph, vertex_moves, delta_entropy, block_assignment)
            for (int index = start; index < end; ++index) {
                int vertex = graph.low_degree_vertices()[index];
                VertexMove proposal = propose_gibbs_move(blockmodel, vertex, graph);
                if (proposal.did_move) {
                    #pragma omp atomic
                    vertex_moves++;
                    delta_entropy += proposal.delta_entropy;
                    block_assignment[vertex] = proposal.proposed_block;
                }
            }
            blockmodel = Blockmodel(blockmodel.getNum_blocks(), graph.out_neighbors(),
                                    blockmodel.getBlock_reduction_rate(), block_assignment);
        }
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy / initial_entropy << std::endl;
        total_vertex_moves += vertex_moves;
        MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, blockmodels, initial_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

VertexMove move_vertex(int vertex, int current_block, utils::ProposalAndEdgeCounts proposal, Blockmodel &blockmodel,
                       const Graph &graph, EdgeWeights &out_edges, EdgeWeights &in_edges) {
    if (args.nodelta)
        return move_vertex_nodelta(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
    Delta delta = blockmodel_delta(vertex, current_block, proposal.proposal, out_edges, in_edges,
                                   blockmodel);

    double hastings = entropy::hastings_correction(vertex, graph, blockmodel, delta, current_block, proposal);
    double delta_entropy = entropy::delta_mdl(blockmodel, delta, proposal);

    if (accept(delta_entropy, hastings)) {
        blockmodel.move_vertex(vertex, delta, proposal);
        return VertexMove{delta_entropy, true, vertex, proposal.proposal};
    }
    return VertexMove{delta_entropy, false, vertex, proposal.proposal};
}

VertexMove move_vertex_nodelta(int vertex, int current_block, utils::ProposalAndEdgeCounts proposal,
                               Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
                               EdgeWeights &in_edges) {
    EdgeWeights blocks_out_neighbors = block_edge_weights(blockmodel.block_assignment(), out_edges);
    EdgeWeights blocks_in_neighbors = block_edge_weights(blockmodel.block_assignment(), in_edges);
    SparseEdgeCountUpdates updates;
    edge_count_updates_sparse(blockmodel, vertex, current_block, proposal.proposal, out_edges, in_edges, updates);
    int current_block_self_edges = updates.block_row[current_block];
    int proposed_block_self_edges = updates.proposal_row[proposal.proposal];
    common::NewBlockDegrees new_block_degrees = common::compute_new_block_degrees(
            current_block, blockmodel, current_block_self_edges, proposed_block_self_edges, proposal);
    double hastings =
            entropy::hastings_correction(blockmodel, blocks_out_neighbors, blocks_in_neighbors, proposal, updates,
                                         new_block_degrees);
    double delta_entropy =
            entropy::delta_mdl(current_block, proposal.proposal, blockmodel, graph.num_edges(), updates,
                               new_block_degrees);
    if (accept(delta_entropy, hastings)) {
        blockmodel.move_vertex(vertex, current_block, proposal.proposal, updates, new_block_degrees.block_degrees_out,
                               new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
        return VertexMove{delta_entropy, true, vertex, proposal.proposal};
    }
    return VertexMove{delta_entropy, false, -1, -1};
}

VertexMove propose_move(Blockmodel &blockmodel, int vertex, const Graph &graph) {
    bool did_move = false;
    int current_block = blockmodel.block_assignment(vertex);
    EdgeWeights out_edges = edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(), vertex, true);

    utils::ProposalAndEdgeCounts proposal = common::propose_new_block(
            current_block, out_edges, in_edges, blockmodel.block_assignment(), blockmodel, false);
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1};
    }

    return move_vertex(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

VertexMove propose_gibbs_move(const Blockmodel &blockmodel, int vertex, const Graph &graph) {
    bool did_move = false;
    int current_block = blockmodel.block_assignment(vertex);

    EdgeWeights out_edges = edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(), vertex, true);

    utils::ProposalAndEdgeCounts proposal = common::propose_new_block(current_block, out_edges, in_edges,
                                                                      blockmodel.block_assignment(), blockmodel,
                                                                      false);
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1};
    }
    /**if (blockmodel.degrees_in(proposal.proposal) == 0 && proposal.num_in_neighbor_edges == 0) {
        std::cout << "out neighbors for " << vertex << " --> " << proposal.proposal << ": ";
        for (int n : graph.out_neighbors(vertex)) {
            std::cout << n << " (" << blockmodel.block_assignment(n) << "), ";
        }
        std::cout << std::endl;
        std::cout << "in neighbors for " << vertex << " --> " << proposal.proposal << ": ";
        for (int n : graph.in_neighbors(vertex)) {
            std::cout << n << " (" << blockmodel.block_assignment(n) << "), ";
        }
        std::cout << std::endl;
    }*/
    return eval_vertex_move(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

Blockmodel &metropolis_hastings(Blockmodel &blockmodel, Graph &graph, BlockmodelTriplet &blockmodels) {
    std::cout << "Metropolis hastings iteration" << std::endl;
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double delta_entropy = 0.0;
        for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            VertexMove proposal = propose_move(blockmodel, vertex, graph);
            if (proposal.did_move) {
                vertex_moves++;
                delta_entropy += proposal.delta_entropy;
            }
        }
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy << std::endl;
        total_vertex_moves += vertex_moves;
        MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, blockmodels, blockmodel.getOverall_entropy(), delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

[[maybe_unused]] Blockmodel &finetune_assignment(Blockmodel &blockmodel, Graph &graph) {
    std::vector<double> delta_entropies;
    // TODO: Add number of finetuning iterations to evaluation
    int total_vertex_moves = 0;
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double delta_entropy = 0.0;
        for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            VertexMove proposal = propose_move(blockmodel, vertex, graph);
            if (proposal.did_move) {
                vertex_moves++;
                delta_entropy += proposal.delta_entropy;
            }
        }
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of finetuning moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy / blockmodel.getOverall_entropy() << std::endl;
        total_vertex_moves += vertex_moves;
        // Early stopping
        if (early_stop(iteration, blockmodel.getOverall_entropy(), delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

//namespace undirected {
//
//double mdl(const Blockmodel &blockmodel, int num_vertices, int num_edges) {
//    std::cout << "undirected!" << std::endl;
//    double log_posterior_p = blockmodel.log_posterior_probability(num_edges);
//    if (std::isnan(log_posterior_p)) {
//        std::cout << "nan in log posterior" << std::endl;
//        exit(-5000);
//    }
//    double x = blockmodel.getNum_blocks() * (blockmodel.getNum_blocks() + 1.0) / (2.0 * num_edges);
//    if (std::isnan(x)) {
//        std::cout << "nan in X" << std::endl;
//        exit(-5000);
//    }
//    double h = ((1 + x) * log(1 + x)) - (x * log(x));
//    if (std::isnan(h)) {
//        std::cout << "nan in h()" << std::endl;
//    }
//        // std::cout << "X: " << x << std::endl;
//        // std::cout << "log(X): " << log(x) << std::endl;
//    if (std::isnan(h)) {
//        exit(-5000);
//    }
//    double first = (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks()));
//    std::cout << "first: " << first << " log_posterior: " << log_posterior_p << std::endl;
//    double result = (num_edges * h) + (num_vertices * log(blockmodel.getNum_blocks())) - log_posterior_p;
//    if (std::isnan(result)) {
//        std::cout << "nan in result" << std::endl;
//        exit(-5000);
//    }
//    return result;
//}
//
//}  // namespace undirected

namespace dist {

TwoHopBlockmodel &asynchronous_gibbs(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels) {
    my_file.open(args.csv, std::ios::out | std::ios::app);
    MPI_Datatype Membership_t;
    int membership_blocklengths[2] = {1, 1};
    MPI_Aint membership_displacements[2] = {0, sizeof(int)};
    MPI_Datatype membership_types[2] = {MPI_INT, MPI_INT};
    MPI_Type_create_struct(2, membership_blocklengths, membership_displacements, membership_types, &Membership_t);
    MPI_Type_commit(&Membership_t);
    // MPI Datatype init
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    double old_entropy = entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    blockmodel.setOverall_entropy(old_entropy);
//    double initial_entropy = blockmodel.getOverall_entropy();
    double new_entropy = 0;
    double t0;
    double t1;
    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        int vertex_moves = 0;
        double num_batches = args.batches;
        int batch_size = int(ceil(graph.num_vertices() / num_batches));
        // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
        // asynchronous Gibbs sampling
        std::vector<int> block_assignment(blockmodel.block_assignment());
//        int my_vertices = 0;
        for (int batch = 0; batch < graph.num_vertices() / batch_size; ++batch) {
            t0 = MPI_Wtime();
            int start = batch * batch_size;
            int end = std::min(graph.num_vertices(), (batch + 1) * batch_size);
            std::vector<Membership> membership_updates;
#pragma omp parallel for schedule(dynamic) default(none) \
            shared(start, end, blockmodel, graph, vertex_moves, membership_updates, block_assignment)
            for (int vertex = start; vertex < end; ++vertex) {
                // TODO: separate "new" code so can be switched on/off
                // TODO: batch by % of my vertices? Can be calculated same time as load balancing
                if (!blockmodel.owns_vertex(vertex)) continue;
                VertexMove proposal = dist::propose_gibbs_move(blockmodel, vertex, graph);
                if (proposal.did_move) {
//                    assert(blockmodel.stores(proposal.proposed_block));  // assert no work with default(none) until gcc 9.3.0
#pragma omp critical (updates)
                    {
                        membership_updates.push_back(Membership{vertex, proposal.proposed_block});
                    }
                }
            }
            int num_moves = (int) membership_updates.size();
            // MPI COMMUNICATION
            int rank_moves[mpi.num_processes];
            t1 = MPI_Wtime();
            my_file << mpi.rank << "," << MCMC_iterations << "," << t1 - t0 << std::endl;
            MPI_Allgather(&num_moves, 1, MPI_INT, &rank_moves, 1, MPI_INT, MPI_COMM_WORLD);
            int offsets[mpi.num_processes];
            offsets[0] = 0;
            for (int i = 1; i < mpi.num_processes; ++i) {
                offsets[i] = offsets[i - 1] + rank_moves[i - 1];
            }
            int batch_vertex_moves = offsets[mpi.num_processes - 1] + rank_moves[mpi.num_processes - 1];
            std::vector<Membership> collected_membership_updates(batch_vertex_moves);
            MPI_Allgatherv(membership_updates.data(), num_moves, Membership_t, collected_membership_updates.data(),
                           rank_moves, offsets, Membership_t, MPI_COMM_WORLD);
            // END MPI COMMUNICATION
            for (const Membership &membership: collected_membership_updates) {
                block_assignment[membership.vertex] = membership.block;
            }
            blockmodel.set_block_assignment(block_assignment);
            blockmodel.build_two_hop_blockmodel(graph.out_neighbors());
            blockmodel.initialize_edge_counts(graph.out_neighbors());
            vertex_moves += batch_vertex_moves;
        }
        new_entropy = entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        old_entropy = new_entropy;
        delta_entropies.push_back(delta_entropy);
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << vertex_moves << " delta S: "
                      << delta_entropy / new_entropy << std::endl;
        }
        total_vertex_moves += vertex_moves;
        MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, blockmodels, new_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(new_entropy);
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    MPI_Type_free(&Membership_t);
    my_file.close();
    // are there more iterations with the 2-hop blockmodel due to restricted vertex moves?
    return blockmodel;
}

TwoHopBlockmodel &metropolis_hastings(TwoHopBlockmodel &blockmodel, Graph &graph, DistBlockmodelTriplet &blockmodels) {
    // std::cout << "running distributed metropolis hastings yo!" << std::endl;
    my_file.open(args.csv, std::ios::out | std::ios::app);
    // MPI Datatype init
    MPI_Datatype Membership_t;
    int membership_blocklengths[2] = {1, 1};  // Number of items in each field of Membership_t
    MPI_Aint membership_displacements[2] = {0, sizeof(int)};
    MPI_Datatype membership_types[2] = {MPI_INT, MPI_INT};
    MPI_Type_create_struct(2, membership_blocklengths, membership_displacements, membership_types, &Membership_t);
    MPI_Type_commit(&Membership_t);
    // MPI Datatype init
    if (blockmodel.getNum_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    int total_vertex_moves = 0;
    double old_entropy = entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    blockmodel.setOverall_entropy(old_entropy);
//    double initial_entropy = blockmodel.getOverall_entropy();
    double new_entropy = 0;
    double t0, t1;
    for (int iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        t0 = MPI_Wtime();
        // Block assignment used to re-create the Blockmodel after each iteration to incorporate moves from other ranks
        std::vector<int> block_assignment(blockmodel.block_assignment());
        std::vector<Membership> membership_updates;
        int vertex_moves = 0;
        for (int vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            if (!blockmodel.owns_vertex(vertex)) continue;
            VertexMove proposal = dist::propose_mh_move(blockmodel, vertex, graph);
            if (proposal.did_move) {
                vertex_moves++;
                assert(blockmodel.stores(proposal.proposed_block));
                membership_updates.push_back(Membership{vertex, proposal.proposed_block});
            }
        }
        int num_moves = (int) membership_updates.size();
        // MPI COMMUNICATION
        int rank_moves[mpi.num_processes];
        t1 = MPI_Wtime();
        my_file << mpi.rank << "," << MCMC_iterations << "," << t1 - t0 << std::endl;
        MPI_Allgather(&num_moves, 1, MPI_INT, &rank_moves, 1, MPI_INT, MPI_COMM_WORLD);
        int offsets[mpi.num_processes];
        offsets[0] = 0;
        for (int i = 1; i < mpi.num_processes; ++i) {
            offsets[i] = offsets[i - 1] + rank_moves[i - 1];
        }
        int batch_vertex_moves = offsets[mpi.num_processes - 1] + rank_moves[mpi.num_processes - 1];
        std::vector<Membership> collected_membership_updates(batch_vertex_moves);
        MPI_Allgatherv(membership_updates.data(), num_moves, Membership_t, collected_membership_updates.data(),
                       rank_moves, offsets, Membership_t, MPI_COMM_WORLD);
        // END MPI COMMUNICATION
        for (const Membership &membership: collected_membership_updates) {
            block_assignment[membership.vertex] = membership.block;
        }
        blockmodel.set_block_assignment(block_assignment);
        blockmodel.build_two_hop_blockmodel(graph.out_neighbors());
        blockmodel.initialize_edge_counts(graph.out_neighbors());
        vertex_moves += batch_vertex_moves;
        new_entropy = entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        old_entropy = new_entropy;
        delta_entropies.push_back(delta_entropy);
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << vertex_moves << " delta S: "
                      << delta_entropy / new_entropy << std::endl;
        }
        total_vertex_moves += vertex_moves;
        MCMC_iterations++;
        if (early_stop(iteration, blockmodels, new_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(new_entropy);
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    MPI_Type_free(&Membership_t);
    my_file.close();
    return blockmodel;
}

bool early_stop(int iteration, DistBlockmodelTriplet &blockmodels, double initial_entropy,
                std::vector<double> &delta_entropies) {
    size_t last_index = delta_entropies.size() - 1;
    if (delta_entropies[last_index] == 0.0) {
        return true;
    }
    if (iteration < 3) {
        return false;
    }
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold;
    if (blockmodels.get(2).empty) { // Golden ratio bracket not yet established
        threshold = 5e-4 * initial_entropy;
    } else {
        threshold = 1e-4 * initial_entropy;
    }
    return average < threshold;
}

VertexMove propose_gibbs_move(const TwoHopBlockmodel &blockmodel, int vertex, const Graph &graph) {
    bool did_move = false;
    int current_block = blockmodel.block_assignment(vertex);

    EdgeWeights out_edges = edge_weights(graph.out_neighbors(), vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(), vertex, true);

    utils::ProposalAndEdgeCounts proposal = common::dist::propose_new_block(
            current_block, out_edges, in_edges, blockmodel.block_assignment(), blockmodel, false);
    if (!blockmodel.stores(proposal.proposal)) {
        std::cerr << "blockmodel doesn't own proposed block!!!!!" << std::endl;
        exit(-1000000000);
    }
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1};
    }

    return eval_vertex_move(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

VertexMove propose_mh_move(TwoHopBlockmodel &blockmodel, int vertex, const Graph &graph) {
    bool did_move = false;
    int current_block = blockmodel.block_assignment(vertex);  // getBlock_assignment()[vertex];
    EdgeWeights out_edges = edge_weights(graph.out_neighbors(), vertex);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(), vertex);

    utils::ProposalAndEdgeCounts proposal = common::dist::propose_new_block(
            current_block, out_edges, in_edges, blockmodel.block_assignment(), blockmodel, false);
    if (!blockmodel.stores(proposal.proposal)) {
        std::cerr << "blockmodel doesn't own proposed block!!!!!" << std::endl;
        exit(-1000000000);
    }
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1};
    }

    return move_vertex(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

}  // namespace dist

}  // namespace finetune
