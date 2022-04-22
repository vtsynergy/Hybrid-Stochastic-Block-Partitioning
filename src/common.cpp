#include "common.hpp"

#include "args.hpp"

#include "assert.h"
#include "utils.hpp"
#include "typedefs.hpp"

namespace common {

int choose_neighbor(std::vector<int> &neighbor_indices, std::vector<int> &neighbor_weights) {
    std::discrete_distribution<int> distribution(neighbor_weights.begin(), neighbor_weights.end());
    int index = distribution(generator);
    return neighbor_indices[index];
}

int choose_neighbor(const SparseVector<double> &multinomial_distribution) {
    // std::cout << "in choose_neighbor" << std::endl;
    // std::cout << "data.size = " << multinomial_distribution.data.size() << " idx.size = ";
    // std::cout << multinomial_distribution.idx.size() << std::endl;
    std::discrete_distribution<int> distribution(
        multinomial_distribution.data.begin(), multinomial_distribution.data.end());
    int index = distribution(generator);
    // std::cout << "index = " << index << std::endl;
    return multinomial_distribution.idx[index];
}

NewBlockDegrees compute_new_block_degrees(int current_block, const Blockmodel &blockmodel, int current_block_self_edges,
                                          int proposed_block_self_edges, utils::ProposalAndEdgeCounts &proposal) {
    std::vector<int> degrees_out(blockmodel.degrees_out());
    std::vector<int> degrees_in(blockmodel.degrees_in());
    std::vector<int> degrees_total(blockmodel.degrees());
    degrees_out[current_block] -= proposal.num_out_neighbor_edges;
    degrees_out[proposal.proposal] += proposal.num_out_neighbor_edges;
    degrees_in[current_block] -= proposal.num_in_neighbor_edges;
    degrees_in[proposal.proposal] += proposal.num_in_neighbor_edges;
    degrees_total[current_block] = degrees_out[current_block] + degrees_in[current_block] - current_block_self_edges;
    degrees_total[proposal.proposal] = degrees_out[proposal.proposal] + degrees_in[proposal.proposal]
            - proposed_block_self_edges;
    return NewBlockDegrees{degrees_out, degrees_in, degrees_total};
}

double delta_entropy_temp(std::vector<int> &row_or_col, std::vector<int> &block_degrees, int degree, int num_edges) {
    if (args.undirected)
        return undirected::delta_entropy_temp(row_or_col, block_degrees, degree, num_edges);
    return directed::delta_entropy_temp(row_or_col, block_degrees, degree);
}

double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int num_edges) {
    if (args.undirected)
        return undirected::delta_entropy_temp(row_or_col, block_degrees, degree, num_edges);
    return directed::delta_entropy_temp(row_or_col, block_degrees, degree);
}

double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int current_block, int proposal, int num_edges) {
    if (args.undirected)
        return undirected::delta_entropy_temp(row_or_col, block_degrees, degree, current_block, proposal, num_edges);
    return directed::delta_entropy_temp(row_or_col, block_degrees, degree, current_block, proposal);
}

std::vector<int> exclude_indices(const std::vector<int> &in, int index1, int index2) {
    std::vector<int> out = utils::constant<int>((int) in.size() - 1, 0);
    int count = 0;
    for (int i = 0; i < (int) in.size(); ++i) {
        if (i == index1 || i == index2) {
            continue;
        }
        out[count] = in[i];
        count++;
    }
    return out;
}

MapVector<int>& exclude_indices(MapVector<int> &in, int index1, int index2) {
    // MapVector<int> out(in);
    in.erase(index1);
    in.erase(index2);
    return in;
}

std::vector<int> index_nonzero(const std::vector<int> &values, std::vector<int> &indices) {
    std::vector<int> results;
    for (size_t i = 0; i < indices.size(); ++i) {
        int index = indices[i];
        if (index != 0) {
            int value = values[i];
            results.push_back(value);
        }
    }
    return results;
}

std::vector<int> index_nonzero(const std::vector<int> &values, MapVector<int> &indices_map) {
    // if (omp_get_thread_num() == 0)
    //     std::cout << "sparse version" << std::endl;
    std::vector<int> results;
    for (const std::pair<int, int> &element : indices_map) {
        if (element.second != 0) {
            // if (omp_get_thread_num() == 0)
            //     std::cout << "pushing " << values[element.first] << " because " << element.first << " = " << element.second << std::endl;
            results.push_back(values[element.first]);
        }
    }
    return results;
}

std::vector<int> nonzeros(std::vector<int> &in) {
    std::vector<int> values;
    for (size_t i = 0; i < in.size(); ++i) {
        int value = in[i];
        if (value != 0) {
            values.push_back(value);
        }
    }
    return values;
}

std::vector<int> nonzeros(MapVector<int> &in) {
    std::vector<int> values;
    for (const std::pair<const int, int> &element : in) {
        if (element.second != 0) {
            values.push_back(element.second);
        }
    }
    return values;
}

// TODO: get rid of block_assignment, just use blockmodel?
utils::ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                               const std::vector<int> &block_assignment, const Blockmodel &blockmodel,
                                               bool block_merge) {
    std::vector<int> neighbor_indices = utils::concatenate<int>(out_blocks.indices, in_blocks.indices);
    std::vector<int> neighbor_weights = utils::concatenate<int>(out_blocks.values, in_blocks.values);
    int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
    int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
    int k = k_out + k_in;
    int num_blocks = blockmodel.getNum_blocks();

    if (k == 0) { // If the current block has no neighbors, propose merge with random block
        int proposal = propose_random_block(current_block, num_blocks);
        return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    int neighbor_block;
    if (block_merge)
        neighbor_block = choose_neighbor(neighbor_indices, neighbor_weights);
    else {
        int neighbor = choose_neighbor(neighbor_indices, neighbor_weights);
        neighbor_block = block_assignment[neighbor];
    }

    // With a probability inversely proportional to block degree, propose a random block merge
    if (std::rand() <= (num_blocks / ((float) blockmodel.degrees(neighbor_block) + num_blocks))) {
        int proposal = propose_random_block(current_block, num_blocks);
        return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }

    // Build multinomial distribution
    double total_edges = 0.0;
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    const MapVector<int> &col = matrix->getcol_sparse(neighbor_block);
    MapVector<int> edges = blockmodel.blockmatrix()->getrow_sparse(neighbor_block);
    for (const std::pair<const int, int> &pair : col) {
        edges[pair.first] += pair.second;
    }
    if (block_merge) {  // Make sure proposal != current_block
        edges[current_block] = 0;
        total_edges = utils::sum<double, int>(edges);
        if (total_edges == 0.0) { // Neighbor block has no neighbors, so propose a random block
            int proposal = propose_random_block(current_block, num_blocks);
            return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
        }
    } else {
        total_edges = utils::sum<double, int>(edges);
    }
    // Propose a block based on the multinomial distribution of block neighbor edges
    SparseVector<double> multinomial_distribution;
    utils::div(edges, total_edges, multinomial_distribution);
    int proposal = choose_neighbor(multinomial_distribution);
    return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
}

int propose_random_block(int current_block, int num_blocks) {
    // Generate numbers 0..num_blocks-2 in order to exclude the current block
    std::uniform_int_distribution<int> distribution(0, num_blocks - 2);
    int proposed = distribution(generator);
    if (proposed >= current_block) {
        proposed++;
    }
    return proposed;
}

int random_integer(int low, int high) {
    std::uniform_int_distribution<int> distribution(low, high);
    return distribution(generator);
}

namespace directed {

double delta_entropy_temp(std::vector<int> &row_or_col, std::vector<int> &block_degrees, int degree) {
    // std::cout << "dE_temp_directed_dense!" << std::endl;
    std::vector<float> row_or_col_double = utils::to_float<int>(row_or_col);
    std::vector<float> block_degrees_double = utils::to_float<int>(block_degrees);
    std::vector<float> result = row_or_col_double / block_degrees_double / float(degree);
    result = row_or_col_double * utils::nat_log<float>(result);
    return (double)utils::sum<float>(result);
}

double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree) {
    // std::cout << "dE_temp_directed_sparse!" << std::endl;
    // throw std::runtime_error("SHOULD BE UNDIRECTED");
    double result = 0.0;
    for (const std::pair<const int, int> &pair : row_or_col) {
        if (pair.second == 0)  // 0s sometimes get inserted into the sparse matrix
            continue;
        double temp = (double) pair.second / (double) block_degrees[pair.first] / degree;
        temp = (double) pair.second * std::log(temp);
        result += temp;
    }
    return result;
}

double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int current_block, int proposal) {
    // std::cout << "dE_temp_directed_sparse_ignore!" << std::endl;
    // throw std::runtime_error("SHOULD BE UNDIRECTED");
    double result = 0.0;
    for (const std::pair<const int, int> &pair : row_or_col) {
        // 0s sometimes get inserted into the sparse matrix
        if (pair.second == 0 || pair.first == current_block || pair.first == proposal)
            continue;
        double temp = (double) pair.second / (double) block_degrees[pair.first] / degree;
        temp = (double) pair.second * std::log(temp);
        result += temp;
    }
    return result;
}
}  // namespace directed

namespace undirected {

double delta_entropy_temp(std::vector<int> &row_or_col, std::vector<int> &block_degrees, int degree, int num_edges) {
    // std::cout << "dE_temp_undirected_dense!" << std::endl;
    std::vector<double> row_or_col_double = utils::to_double<int>(row_or_col) / 2.0;
    std::vector<double> block_degrees_double = utils::to_double<int>(block_degrees) / 2.0;
    std::vector<double> result = (row_or_col_double / 2.0) / (block_degrees_double * degree);
    // std::vector<double> result = row_or_col_double / (block_degrees_double * degree * 2.0 * num_edges);
    result = row_or_col_double * utils::nat_log<double>(result);
    double dE = 0.5 * utils::sum<double>(result);
    assert(!std::isnan(dE));
    return dE;
    // std::vector<double> row_or_col_double = utils::to_double<int>(row_or_col) / (2.0 * num_edges);
    // std::vector<double> block_degrees_double = utils::to_double<int>(_block_degrees) / (2.0 * num_edges);
    // std::vector<double> result = (row_or_col_double * 2.0 * num_edges) / (block_degrees_double * degree);
    // // std::vector<double> result = row_or_col_double / (block_degrees_double * degree * 2.0 * num_edges);
    // result = row_or_col_double * utils::nat_log<double>(result);
    // double dE = 0.5 * utils::sum<double>(result);
    // assert(!std::isnan(dE));
    // return dE;
    // return 0.5 * utils::sum<double>(result);
}

double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int num_edges) {
    double result = 0.0;
    double deg = degree / 2.0;
    for (const std::pair<int, int> &pair : row_or_col) {
        if (pair.second == 0)  // 0s sometimes get inserted into the sparse matrix
            continue;
        double block_deg = (double) block_degrees[pair.first] / 2.0;
        // double temp = (double) pair.second / (2.0 * num_edges * (double) _block_degrees[pair.first] * degree);
        double temp = ((double) pair.second / 2.0) / (block_deg * deg);
        temp = (double) pair.second * std::log(temp);
        result += temp;
    }
    result *= 0.5;
    assert(!std::isnan(result));
    return result;
    // // std::cout << "dE_temp_undirected_sparse!" << std::endl;
    // double result = 0.0;
    // double deg = degree / (2.0 * num_edges);
    // for (const std::pair<int, int> &pair : row_or_col) {
    //     if (pair.second == 0)  // 0s sometimes get inserted into the sparse matrix
    //         continue;
    //     double block_deg = (double) _block_degrees[pair.first] / (2.0 * num_edges);
    //     // double temp = (double) pair.second / (2.0 * num_edges * (double) _block_degrees[pair.first] * degree);
    //     double temp = ((double) pair.second * 2.0 * num_edges) / (block_deg * deg);
    //     temp = (double) pair.second * std::log(temp);
    //     result += temp;
    // }
    // result *= 0.5;
    // assert(!std::isnan(result));
    // return result;
}

double delta_entropy_temp(const MapVector<int> &row_or_col, const std::vector<int> &block_degrees, int degree,
                          int current_block, int proposal, int num_edges) {
    double result = 0.0;
    double deg = degree / 2.0;
    for (const std::pair<int, int> &pair : row_or_col) {
        // 0s sometimes get inserted into the sparse matrix
        if (pair.second == 0 || pair.first == current_block || pair.first == proposal)
            continue;
        double block_deg = (double) block_degrees[pair.first] / 2.0;
        double temp = ((double) pair.second / 2.0) / (block_deg * deg);
        // double temp = (double) pair.second / (2.0 * num_edges * (double) _block_degrees[pair.first] * degree);
        temp = (double) pair.second * std::log(temp);
        result += temp;
    }
    result *= 0.5;
    assert(!std::isnan(result));
    return result;
    // double result = 0.0;
    // double deg = degree / (2.0 * num_edges);
    // for (const std::pair<int, int> &pair : row_or_col) {
    //     // 0s sometimes get inserted into the sparse matrix
    //     if (pair.second == 0 || pair.first == current_block || pair.first == proposal)
    //         continue;
    //     double block_deg = (double) _block_degrees[pair.first] / (2.0 * num_edges);
    //     double temp = ((double) pair.second * 2.0 * num_edges) / (block_deg * deg);
    //     // double temp = (double) pair.second / (2.0 * num_edges * (double) _block_degrees[pair.first] * degree);
    //     temp = (double) pair.second * std::log(temp);
    //     result += temp;
    // }
    // result *= 0.5;
    // assert(!std::isnan(result));
    // return result;
}
}  // namespace undirected

namespace dist {

// TODO: get rid of block_assignment, just use blockmodel?
utils::ProposalAndEdgeCounts propose_new_block(int current_block, EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                               const std::vector<int> &block_assignment, const TwoHopBlockmodel &blockmodel,
                                               bool block_merge) {
    std::vector<int> neighbor_indices = utils::concatenate<int>(out_blocks.indices, in_blocks.indices);
    std::vector<int> neighbor_weights = utils::concatenate<int>(out_blocks.values, in_blocks.values);
    int k_out = std::accumulate(out_blocks.values.begin(), out_blocks.values.end(), 0);
    int k_in = std::accumulate(in_blocks.values.begin(), in_blocks.values.end(), 0);
    int k = k_out + k_in;
    int num_blocks = blockmodel.getNum_blocks();

    if (k == 0) { // If the current block has no neighbors, propose merge with random block
        std::vector<int> blocks = utils::range<int>(0, blockmodel.getNum_blocks());
        std::vector<int> weights = utils::to_int<bool>(blockmodel.in_two_hop_radius());
        int proposal = choose_neighbor(blocks, weights);
        // int proposal = propose_random_block(current_block, num_blocks);  // TODO: only propose blocks in 2 hop radius
        assert(blockmodel.stores(proposal));
        return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }
    int neighbor_block;
    if (block_merge)
        neighbor_block = choose_neighbor(neighbor_indices, neighbor_weights);
    else {
        int neighbor = choose_neighbor(neighbor_indices, neighbor_weights);
        neighbor_block = block_assignment[neighbor];
    }
    assert(blockmodel.stores(neighbor_block));

    // With a probability inversely proportional to block degree, propose a random block merge
    if (std::rand() <= (num_blocks / ((float) blockmodel.degrees(neighbor_block) + num_blocks))) {
        std::vector<int> blocks = utils::range<int>(0, blockmodel.getNum_blocks());
        std::vector<int> weights = utils::to_int<bool>(blockmodel.in_two_hop_radius());
        int proposal = choose_neighbor(blocks, weights);
        // int proposal = propose_random_block(current_block, num_blocks);
        assert(blockmodel.stores(proposal));
        return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
    }

    // Build multinomial distribution
    double total_edges = 0.0;
    const std::shared_ptr<ISparseMatrix> matrix = blockmodel.blockmatrix();
    const MapVector<int> &col = matrix->getcol_sparse(neighbor_block);
    MapVector<int> edges = blockmodel.blockmatrix()->getrow_sparse(neighbor_block);
    for (const std::pair<int, int> &pair : col) {
        edges[pair.first] += pair.second;
    }
    if (block_merge) {  // Make sure proposal != current_block
        edges[current_block] = 0;
        total_edges = utils::sum<double, int>(edges);
        if (total_edges == 0.0) { // Neighbor block has no neighbors, so propose a random block
            int proposal = propose_random_block(current_block, num_blocks);
            assert(blockmodel.stores(proposal));
            return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
        }
    } else {
        total_edges = utils::sum<double, int>(edges);
    }
    // Propose a block based on the multinomial distribution of block neighbor edges
    SparseVector<double> multinomial_distribution;
    utils::div(edges, total_edges, multinomial_distribution);
    int proposal = choose_neighbor(multinomial_distribution);
    assert(blockmodel.stores(proposal));
    return utils::ProposalAndEdgeCounts{proposal, k_out, k_in, k};
}

}  // namespace dist

}  // namespace common
