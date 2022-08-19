#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
//#include <limits.h>
#include <omp.h>
#include <random>
#include <vector>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "block_merge.hpp"
#include "entropy.hpp"
#include "evaluate.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "sbp.hpp"
#include "utils.hpp"

MPI_t mpi;
Args args;

struct Result {
    std::vector<int> membership;
    std::vector<std::vector<double>> matrix;
    double max_influence;
    double avg_influence;
};

double stop = false;

std::vector<std::vector<int>> Graph1 { {0, 0}, {0, 1}, {0, 2}, {1, 2}, {2, 3}, {3, 1}, {3, 2}, {3, 5}, {4, 1}, {4, 6},
                                       {5, 4}, {5, 5}, {5, 6}, {5, 7}, {6, 4}, {7, 3}, {7, 9}, {8, 5}, {8, 7}, {9, 10},
                                       {10, 7}, {10, 8}, {10, 10}
};

std::vector<std::vector<int>> Graph2 { {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4},
                                       {5, 6}, {5, 7}, {5, 8}, {5, 9}, {6, 7}, {6, 8}, {6, 9}, {7, 8}, {7, 9}, {8, 9},
                                       {2, 6}
};

std::vector<std::vector<int>> Graph3 { {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4},
                                       {5, 6}, {5, 7}, {5, 8}, {5, 9}, {6, 7}, {6, 8}, {6, 9}, {7, 8}, {7, 9}, {8, 9},
                                       {2, 6}, {2, 5}, {4, 8}
};

std::vector<std::vector<int>> Graph4 { {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4},
                                       {5, 6}, {5, 7}, {5, 8}, {5, 9}, {6, 7}, {6, 8}, {6, 9}, {7, 8}, {7, 9}, {8, 9},
                                       {2, 6}, {2, 5}, {4, 8}, {0, 5}, {4, 6}
};

std::vector<std::vector<int>> Graph5 { {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4},
                                       {5, 6}, {5, 7}, {5, 8}, {5, 9}, {6, 7}, {6, 8}, {6, 9}, {7, 8}, {7, 9}, {8, 9},
                                       {2, 6}, {2, 5}, {4, 8}, {0, 5}, {4, 6}, {6, 0}, {5, 4}
};

std::vector<std::vector<int>> Graph6 { {0, 1}, {2, 0}, {0, 3}, {4, 0}, {5, 0}, {0, 6}, {2, 1}, {2, 3}, {5, 4}, {6, 5},
                                       {6, 1}, {3, 8}, {7, 9}, {8, 7}, {9, 8}
};

std::vector<std::vector<int>> Graph7 { {0, 1}, {2, 0}, {0, 3}, {4, 0}, {5, 0}, {0, 6}, {2, 1}, {2, 3}, {5, 4}, {6, 5},
                                       {6, 1}, {3, 8}, {7, 9}, {8, 7}, {9, 8}, {7, 3}
};

std::vector<double> conditional_distribution(const Graph &graph, const std::vector<int> &assignment, int vertex1,
                                             int num_blocks = -1, bool mdl = false) {
    if (num_blocks == -1)
        num_blocks = graph.num_vertices();
    std::vector<double> distribution(num_blocks, 0.0);
    for (int block = 0; block < num_blocks; ++block) {
        std::vector<int> modified_assignment(assignment);
        modified_assignment[vertex1] = block;
        std::vector<int> block_map(num_blocks, -1);
        int block_number = 0;
        for (int i = 0; i < graph.num_vertices(); ++i) {
            int _block = modified_assignment[i];
            if (block_map[_block] == -1) {
                block_map[_block] = block_number;
                block_number++;
            }
            modified_assignment[i] = block_map[_block];
        }
//        std::cout << "modified assignment: ";
//        utils::print<int>(modified_assignment);
        Blockmodel blockmodel(block_number, graph, 0.5, modified_assignment);
//        std::cout << "log_posterior_prob: " << blockmodel.log_posterior_probability() << " exp(log_p) = " << std::exp(blockmodel.log_posterior_probability()) << std::endl;
        if (mdl)
            distribution[block] = 0.0 - entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        else  // use log posterior probability
            distribution[block] = blockmodel.log_posterior_probability();
    }
//    if (!mdl) {
    double min_log_posterior_probability = std::numeric_limits<double>::max();
    for (double val : distribution) {
        if (val < min_log_posterior_probability)
            min_log_posterior_probability = val;
    }
    min_log_posterior_probability = std::abs(min_log_posterior_probability);
    for (int block = 0; block < num_blocks; ++block) {
        distribution[block] = std::exp(min_log_posterior_probability + distribution[block]);
    }
//    }
//    if (mdl)
//        utils::print<double>(distribution);
//    std::cout << "distribution: ";
//    utils::print<double>(distribution);
    auto sum = utils::sum<double>(distribution);
    if (sum == 0)
        return distribution;
    distribution = distribution / sum;
//    if (mdl)
//        utils::print<double>(distribution);
    return distribution;
}

std::vector<double> neighbor_conditional_distribution(const Graph &graph, const std::vector<int> &assignment,
                                                      int vertex1, std::set<int> &neighbors, int num_blocks = -1,
                                                      bool mdl = false) {
    if (num_blocks == -1)
        num_blocks = graph.num_vertices();
    std::vector<double> distribution;
    for (const int &block : neighbors) {
        std::vector<int> modified_assignment(assignment);
        modified_assignment[vertex1] = block;
        std::vector<int> block_map(num_blocks, -1);
        int block_number = 0;
        for (int i = 0; i < graph.num_vertices(); ++i) {
            int _block = modified_assignment[i];
            if (block_map[_block] == -1) {
                block_map[_block] = block_number;
                block_number++;
            }
            modified_assignment[i] = block_map[_block];
        }
        Blockmodel blockmodel(block_number, graph, 0.5, modified_assignment);
        if (mdl)
            distribution.push_back(0.0 - entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges()));
        else  // use log posterior probability
            distribution.push_back(blockmodel.log_posterior_probability());
    }
    double min_log_posterior_probability = std::numeric_limits<double>::max();
    for (double val : distribution) {
        if (val < min_log_posterior_probability)
            min_log_posterior_probability = val;
    }
    min_log_posterior_probability = std::abs(min_log_posterior_probability);
    for (int block = 0; block < distribution.size(); ++block) {
        distribution[block] = std::exp(min_log_posterior_probability + distribution[block]);
    }
    auto sum = utils::sum<double>(distribution);
    if (sum == 0)
        return distribution;
    distribution = distribution / sum;
    return distribution;
}

double total_variation_distance(const Blockmodel &B, const Graph &graph, int vertex1, int vertex2, int block1, int block2, bool mdl = false) {
    std::vector<int> X(B.block_assignment());  // graph.assignment());
    X[vertex2] = block1;
    std::vector<double> Xcd = conditional_distribution(graph, X, vertex1, B.getNum_blocks(), mdl);
    std::vector<int> Y(B.block_assignment());  // graph.assignment());
    Y[vertex2] = block2;
    std::vector<double> Ycd = conditional_distribution(graph, Y, vertex1, B.getNum_blocks(), mdl);
    double tvd = 0.0;
    for (int block = 0; block < B.getNum_blocks(); ++block) {
//        if (mdl && vertex1 == 0 && vertex2 == 1) {
//            std::cout << "block = " << block << "|" << Xcd[block] << " - " << Ycd[block] << "| = " << Xcd[block] - Ycd[block] << std::endl;
//        }
        tvd += std::abs(Xcd[block] - Ycd[block]);
    }
//    if (vertex1 == 0 && vertex2 == 7) exit(0);
    tvd *= 0.5;
    return tvd;
}

double neighbor_total_variation_distance(const Blockmodel &B, const Graph &graph, int vertex1, int vertex2, int block1,
                                         int block2, bool mdl = false) {
    std::set<int> v1_neighbors = B.blockmatrix()->neighbors(B.block_assignment(vertex1));
    // We're assuming that only the membership to neighboring nodes changes significantly. Thus, adding block1 and
    // block2 to v1_neighbors shouldn't affect the result much, even if they're not neighbors of vertex1.
    v1_neighbors.insert(block1);
    v1_neighbors.insert(block2);
    std::vector<int> X(B.block_assignment());  // graph.assignment());
    X[vertex2] = block1;
    std::vector<double> Xcd = neighbor_conditional_distribution(graph, X, vertex1, v1_neighbors, B.getNum_blocks(), mdl);
    std::vector<int> Y(B.block_assignment());  // graph.assignment());
    Y[vertex2] = block2;
    std::vector<double> Ycd = neighbor_conditional_distribution(graph, Y, vertex1, v1_neighbors, B.getNum_blocks(), mdl);
    double tvd = 0.0;
    for (int block = 0; block < v1_neighbors.size(); ++block) {
        tvd += std::abs(Xcd[block] - Ycd[block]);
    }
    tvd *= 0.5;
    return tvd;
}

Result compute_influence(const Graph &graph, const Blockmodel &B, bool do_merge = false, bool mdl = false) {
    std::cout << "computing influence" << std::endl;
//    std::vector<int> initial_assignment = utils::range<int>(0, graph.num_vertices());
//    Blockmodel B(graph.num_vertices(), graph.out_neighbors(), 0.5, initial_assignment);
//    if (do_merge)
//        B = block_merge::merge_blocks(B, graph.out_neighbors(), graph.num_edges());
//    std::cout << "Block assignment = ";
//    utils::print<int>(B.block_assignment());
    std::vector<double> influence(graph.num_vertices(), std::numeric_limits<double>::min());
    std::vector<std::vector<double>> influence_matrix(graph.num_vertices(), std::vector<double>(graph.num_vertices(), 0.0));
    std::cout << "Influence matrix, where M[x,y] = influence of y on x" << std::endl;
    for (int vertex1 = 0; vertex1 < graph.num_vertices(); ++vertex1) {
        double sum_influence_on_vertex1 = 0.0;
#pragma omp parallel for default(shared)
        for (int vertex2 = 0; vertex2 < graph.num_vertices(); ++vertex2) {
            if (vertex1 == vertex2) continue;
            double max_influence_on_vertex1 = std::numeric_limits<double>::min();
            // Keep block1 as the current block for vertex2
            // Only loop over b1lock2
//            int b1 = -1;
//            int b2 = -1;
            std::vector<double> tvds;
            for (int block1 = 0; block1 < B.getNum_blocks(); ++block1) {
                for (int block2 = block1 + 1; block2 < B.getNum_blocks(); ++block2) {
                    if (block1 == block2) continue;
//                    double tvd = total_variation_distance(B, graph, vertex1, vertex2, block1, block2, mdl);
                    double tvd = total_variation_distance(B, graph, vertex1, vertex2, block1, block2, mdl);
                    tvds.push_back(tvd);
//                    if (tvd > max_influence_on_vertex1) {
//                        b1 = block1;
//                        b2 = block2;
//                    }
                    max_influence_on_vertex1 = std::max(tvd, max_influence_on_vertex1);
                }
            }
            double min_tvd = std::numeric_limits<double>::max(), max_tvd = std::numeric_limits<double>::min(), avg_tvd = 0, std_tvd = 0.0;
            for (double &tvd : tvds) {
                if (tvd < min_tvd) min_tvd = tvd;
                if (tvd > max_tvd) max_tvd = tvd;
                avg_tvd += tvd;
            }
            avg_tvd /= double(tvds.size());
            for (double &tvd : tvds) {
                std_tvd += (tvd - avg_tvd) * (tvd - avg_tvd);
            }
            std_tvd = std::sqrt(std_tvd / double(tvds.size()));
//            std::cout << "min: " << min_tvd << " max: " << max_tvd << " avg: " << avg_tvd << " std: " << std_tvd << std::endl;
            #pragma omp atomic
            sum_influence_on_vertex1 += max_influence_on_vertex1;
            influence_matrix[vertex1][vertex2] = max_influence_on_vertex1;
//            std::cout << "highest tvd on " << vertex2 << " --> " << vertex1 << " when b1 = " << b1 << " b2 = " << b2 << std::endl;
        }
        influence[vertex1] = sum_influence_on_vertex1;
//        max_influence = std::max(max_influence, sum_influence_on_vertex1);
//        int max_influence_vertex = utils::argmax<double>(influence_matrix[vertex1]);
//        std::cout << vertex1 << " (" << sum_influence_on_vertex1 << "," << max_influence_vertex << "): ";
//        utils::print<double>(influence_matrix[vertex1]);
    }
    double max_influence = std::numeric_limits<double>::min();
    double avg_influence = 0.0;
    int v = 0;
    for (double alpha : influence) {
        max_influence = std::max(alpha, max_influence);
        avg_influence += alpha;
        std::cout << alpha << " (" << graph.out_neighbors(v).size() + graph.in_neighbors(v).size() << "), ";
        v++;
    }
    std::cout << std::endl;
    avg_influence /= double(graph.num_vertices());
    std::cout << "total (max) influence = " << max_influence << " avg influence = " << avg_influence << std::endl;
    return { B.block_assignment(), influence_matrix, max_influence, avg_influence };
//    return std::make_tuple(max_influence, avg_influence, influence_matrix);
}

/// Adapted from: https://dev.to/babak/an-algorithm-for-picking-random-numbers-in-a-range-without-repetition-4cp6
std::vector<int> pseudoshuffle_range(int num, int range_max, int num_vertices) {
    MapVector<int> mapping;
    std::vector<int> result;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 generator(seed);
//    std::cout << "seed: " << seed << std::endl;
    std::uniform_int_distribution<int> distribution;
    int index = range_max - 1;
    while (result.size() < num) {
        // below allows re-using the same distribution object with different start, end values.
        int random_index = distribution(generator, decltype(distribution)::param_type{0, index});
        const auto it = mapping.find(random_index);
        if (it != mapping.end())
            random_index = it->second;
        int pointer_index = index;
        const auto it2 = mapping.find(pointer_index);
        if (it2 != mapping.end())
            pointer_index = it2->second;
        mapping[random_index] = pointer_index;
        const auto it3 = mapping.find(pointer_index);
        if (it3 != mapping.end())
            mapping.erase(pointer_index);
        int vertex1 = random_index / num_vertices;
        int vertex2 = random_index % num_vertices;
        if (vertex1 != vertex2)
            result.push_back(random_index);
        index--;
    }
//    utils::print<int>(result);
    return result;
}

Result compute_random_avg_influence(const Graph &graph, const Blockmodel &B, bool mdl = true) {
    std::cout << "computing neighbor influence" << std::endl;
    std::vector<double> influence(graph.num_vertices(), std::numeric_limits<double>::min());
    std::vector<std::vector<double>> influence_matrix(graph.num_vertices(), std::vector<double>(graph.num_vertices(), 0.0));
    std::cout << "Influence matrix, where M[x,y] = influence of y on x" << std::endl;
    double sum_influence = 0.0;
    int num = 100;
    std::vector<int> cells = pseudoshuffle_range(num, graph.num_vertices() * graph.num_vertices(), graph.num_vertices());
    #pragma omp parallel for default(shared)
    for (int i = 0; i < num; ++i) {
        int index = cells[i];
        int vertex1 = index / graph.num_vertices();
        int vertex2 = index % graph.num_vertices();
        if (vertex1 == vertex2) {
            num--;
            continue;
        }
        double max_influence_on_vertex1 = std::numeric_limits<double>::min();
//        int vertex1_block = B.block_assignment(vertex1);
        for (int block1 = 0; block1 < B.getNum_blocks(); ++block1) {
//            if (!B.is_neighbor_of(vertex1_block, block1)) continue;
//            int vertex2_block = B.block_assignment(vertex2);
            for (int block2 = 0; block2 < B.getNum_blocks(); ++block2) {
                if (block1 == block2) continue;
//                if (block1 == block2 || !B.is_neighbor_of(vertex2_block, block2)) continue;
                double tvd = total_variation_distance(B, graph, vertex1, vertex2, block1, block2, mdl);
                max_influence_on_vertex1 = std::max(tvd, max_influence_on_vertex1);
            }
        }
        #pragma omp atomic
        sum_influence += max_influence_on_vertex1;
        influence_matrix[vertex1][vertex2] = max_influence_on_vertex1;
    }
    double avg_influence = (sum_influence / double(num)) * (graph.num_vertices() - 1);
    std::cout << "avg influence = " << avg_influence << std::endl;
    return { B.block_assignment(), influence_matrix, -1, avg_influence };
}

Result compute_random_avg_std_max_influence(const Graph &graph, const Blockmodel &B, bool mdl = true) {
    std::cout << "computing neighbor influence" << std::endl;
    std::vector<double> influence(graph.num_vertices(), std::numeric_limits<double>::min());
    std::vector<std::vector<double>> influence_matrix(graph.num_vertices(), std::vector<double>(graph.num_vertices(), 0.0));
    std::cout << "Influence matrix, where M[x,y] = influence of y on x" << std::endl;
    double sum_influence = 0.0;
    int num = 100;
    std::vector<int> cells = pseudoshuffle_range(num, graph.num_vertices() * graph.num_vertices(), graph.num_vertices());
#pragma omp parallel for default(shared)
    for (int i = 0; i < num; ++i) {
        int index = cells[i];
        int vertex1 = index / graph.num_vertices();
        int vertex2 = index % graph.num_vertices();
        if (vertex1 == vertex2) {
            num--;
            continue;
        }
        double max_influence_on_vertex1 = std::numeric_limits<double>::min();
        std::vector<int> moves;
        if ((B.getNum_blocks() * B.getNum_blocks()) - B.getNum_blocks() > num)
            moves = pseudoshuffle_range(num, B.getNum_blocks() * B.getNum_blocks(), B.getNum_blocks());
        else
            moves = utils::range<int>(0, B.getNum_blocks() * B.getNum_blocks());
        std::vector<double> tvds;
        for (int j = 0; j < moves.size(); ++j) {
            int block1 = j / B.getNum_blocks();
            int block2 = j % B.getNum_blocks();
            if (block1 == block2) continue;
            double tvd = total_variation_distance(B, graph, vertex1, vertex2, block1, block2, mdl);
            tvds.push_back(tvd);
//                max_influence_on_vertex1 = std::max(tvd, max_influence_on_vertex1);
        }
        double mean_tvd = 0.0;
        double std_tvd = 0.0;
        for (const double &tvd : tvds) {
            mean_tvd += tvd;
        }
        mean_tvd /= double(tvds.size());
        for (const double &tvd : tvds) {
            std_tvd += (tvd - mean_tvd) * (tvd - mean_tvd);
        }
        std_tvd = std::sqrt(std_tvd / double(tvds.size()));
        max_influence_on_vertex1 = mean_tvd + (3 * std_tvd);
        #pragma omp atomic
        sum_influence += max_influence_on_vertex1;
        influence_matrix[vertex1][vertex2] = max_influence_on_vertex1;
    }
    double avg_influence = (sum_influence / double(num)) * (graph.num_vertices() - 1);
    std::cout << "avg influence = " << avg_influence << std::endl;
    return { B.block_assignment(), influence_matrix, -1, avg_influence };
}

Result compute_random_avg_neighbor_max_influence(const Graph &graph, const Blockmodel &B, bool mdl = true) {
    std::cout << "computing neighbor influence" << std::endl;
    std::vector<double> influence(graph.num_vertices(), std::numeric_limits<double>::min());
    std::vector<std::vector<double>> influence_matrix(graph.num_vertices(), std::vector<double>(graph.num_vertices(), 0.0));
    std::cout << "Influence matrix, where M[x,y] = influence of y on x" << std::endl;
    double sum_influence = 0.0;
    int num = 100;
    std::vector<int> cells = pseudoshuffle_range(num, graph.num_vertices() * graph.num_vertices(), graph.num_vertices());
    #pragma omp parallel for default(shared)
    for (int i = 0; i < num; ++i) {
        int index = cells[i];
        int vertex1 = index / graph.num_vertices();
        int vertex2 = index % graph.num_vertices();
        if (vertex1 == vertex2) {
            num--;
            continue;
        }
        double max_influence_on_vertex1 = std::numeric_limits<double>::min();
        int vertex1_block = B.block_assignment(vertex1);
        for (int block1 = 0; block1 < B.getNum_blocks(); ++block1) {
            if (!B.is_neighbor_of(vertex1_block, block1)) continue;
            int vertex2_block = B.block_assignment(vertex2);
            for (int block2 = 0; block2 < B.getNum_blocks(); ++block2) {
                if (block1 == block2 || !B.is_neighbor_of(vertex2_block, block2)) continue;
                double tvd = total_variation_distance(B, graph, vertex1, vertex2, block1, block2, mdl);
                max_influence_on_vertex1 = std::max(tvd, max_influence_on_vertex1);
            }
        }
        #pragma omp atomic
        sum_influence += max_influence_on_vertex1;
        influence_matrix[vertex1][vertex2] = max_influence_on_vertex1;
    }
    double avg_influence = (sum_influence / double(num)) * (graph.num_vertices() - 1);
    std::cout << "avg influence = " << avg_influence << std::endl;
    return { B.block_assignment(), influence_matrix, -1, avg_influence };
}

Result compute_random_avg_practical_max_influence(const Graph &graph, const Blockmodel &B, bool mdl = true) {
    std::cout << "computing neighbor influence" << std::endl;
    std::vector<double> influence(graph.num_vertices(), std::numeric_limits<double>::min());
    std::vector<std::vector<double>> influence_matrix(graph.num_vertices(), std::vector<double>(graph.num_vertices(), 0.0));
    std::cout << "Influence matrix, where M[x,y] = influence of y on x" << std::endl;
    double sum_influence = 0.0;
    int num = 100;
    std::vector<int> cells = pseudoshuffle_range(num, graph.num_vertices() * graph.num_vertices(), graph.num_vertices());
    #pragma omp parallel for default(shared)
    for (int i = 0; i < num; ++i) {
        int index = cells[i];
        int vertex1 = index / graph.num_vertices();
        int vertex2 = index % graph.num_vertices();
        if (vertex1 == vertex2) {
            num--;
            continue;
        }
        double max_influence_on_vertex1 = std::numeric_limits<double>::min();
        int vertex2_block = B.block_assignment(vertex1);
        for (int block1 = 0; block1 < B.getNum_blocks(); ++block1) {
            if (!B.is_neighbor_of(vertex2_block, block1) && block1 != vertex2_block) continue;
            for (int block2 = 0; block2 < B.getNum_blocks(); ++block2) {
                if (block1 == block2 || !B.is_neighbor_of(block1, block2)) continue;
                double tvd = total_variation_distance(B, graph, vertex1, vertex2, block1, block2, mdl);
                max_influence_on_vertex1 = std::max(tvd, max_influence_on_vertex1);
            }
        }
        #pragma omp atomic
        sum_influence += max_influence_on_vertex1;
        influence_matrix[vertex1][vertex2] = max_influence_on_vertex1;
    }
    double avg_influence = (sum_influence / double(num)) * (graph.num_vertices() - 1);
    std::cout << "avg influence = " << avg_influence << std::endl;
    return { B.block_assignment(), influence_matrix, -1, avg_influence };
}

Result compute_random_avg_practical_max_neighbor_influence(const Graph &graph, const Blockmodel &B, bool mdl = true) {
    std::cout << "computing neighbor influence" << std::endl;
    std::vector<double> influence(graph.num_vertices(), std::numeric_limits<double>::min());
    std::vector<std::vector<double>> influence_matrix(graph.num_vertices(), std::vector<double>(graph.num_vertices(), 0.0));
    std::cout << "Influence matrix, where M[x,y] = influence of y on x" << std::endl;
    double sum_influence = 0.0;
    int num = 100;
    std::vector<int> cells = pseudoshuffle_range(num, graph.num_vertices() * graph.num_vertices(), graph.num_vertices());
    #pragma omp parallel for default(shared)
    for (int i = 0; i < num; ++i) {
        int index = cells[i];
        int vertex1 = index / graph.num_vertices();
        int vertex2 = index % graph.num_vertices();
        if (vertex1 == vertex2) {
            num--;
            continue;
        }
        double max_influence_on_vertex1 = std::numeric_limits<double>::min();
        int vertex2_block = B.block_assignment(vertex1);
        for (int block1 = 0; block1 < B.getNum_blocks(); ++block1) {
            if (!B.is_neighbor_of(vertex2_block, block1) && block1 != vertex2_block) continue;
            for (int block2 = 0; block2 < B.getNum_blocks(); ++block2) {
                if (block1 == block2 || !B.is_neighbor_of(block1, block2)) continue;
                double tvd = neighbor_total_variation_distance(B, graph, vertex1, vertex2, block1, block2, mdl);
                max_influence_on_vertex1 = std::max(tvd, max_influence_on_vertex1);
            }
        }
        #pragma omp atomic
        sum_influence += max_influence_on_vertex1;
        influence_matrix[vertex1][vertex2] = max_influence_on_vertex1;
    }
    double avg_influence = (sum_influence / double(num)) * (graph.num_vertices() - 1);
    std::cout << "avg influence = " << avg_influence << std::endl;
    return { B.block_assignment(), influence_matrix, -1, avg_influence };
}

Graph to_graph(const std::vector<std::vector<int>> &graph_edges) {
    int num_vertices = -1;
    int num_edges = (int) graph_edges.size();
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    std::vector<bool> self_edges;
    for (const std::vector<int> &edge : graph_edges) {
        int from = edge[0];
        int to = edge[1];
        utils::insert_nodup(out_neighbors, from , to);
        utils::insert_nodup(in_neighbors, to, from);
        int max_v = std::max(from + 1, to + 1);
        num_vertices = std::max(max_v, num_vertices);
        while (self_edges.size() < num_vertices) {
            self_edges.push_back(false);
        }
        if (from == to) {
            self_edges[from] = true;
        }
    }
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<int>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<int>());
    }
    std::vector<int> assignment = utils::range<int>(0, num_vertices);
    Graph graph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges, assignment);
    return graph;
}

void print_csv(const std::vector<Result>& csv_row, double mdl, double f1, const std::string& tag = "test") {
    std::ostringstream filepath_stream;
    filepath_stream << "./influence_results/" << args.numvertices;
    std::string filepath_dir = filepath_stream.str();
    filepath_stream << "/" << args.type << ".csv";
    std::string filepath = filepath_stream.str();
    bool write_header = false;
    if (!fs::exists(filepath)) {
        fs::create_directories(filepath_dir);
        write_header = true;
    }
    std::ofstream file;
    file.open(filepath, std::ios_base::app);
    if (write_header) {
        file << "tag,numvertices,overlap,blocksizevar,undirected,algorithm";
        for (int i = 0; i < 5; ++i) {
            file << ",itr" << i << "_ll_max_influence,itr" << i << "_ll_avg_influence,itr" << i
                 << "_ll_matrix,itr" << i << "membership,itr" << i << "_mdl_max_influence,itr" << i
                 << "_mdl_avg_influence,itr" << i << "_mdl_matrix";
            if (i == 0)
                file << ",itr0.5_ll_max_influence,itr0.5_ll_avg_influence,itr0.5_ll_matrix,itr0.5_membership,"
                     << "itr0.5_mdl_max_influence,itr0.5_mdl_avg_influence,itr0.5_mdl_matrix";
        }
        file << ",mdl,f1" << std::endl;
    }
    file << tag << "," << args.numvertices << "," << args.overlap << "," << args.blocksizevar << "," << args.undirected;
    file << "," << args.algorithm;
    for (int i = 0; i < 6; ++i) {
        const Result &ll_res = csv_row[i * 2];
        file << "," << ll_res.max_influence << "," << ll_res.avg_influence << ",\"";
//        file << "," << std::get<0>(ll_res) << "," << std::get<1>(ll_res) << ",\"";  // ll_res.first << "," << ll_res.second;
//        auto matrix = std::get<2>(ll_res);
        const auto &ll_matrix = ll_res.matrix;
        for (int vertex = 0; vertex < ll_matrix.size(); ++vertex) {
            auto row = ll_matrix[vertex];
            for (int col = 0; col < row.size(); ++col) {
                if (vertex == ll_matrix.size() - 1 && col == ll_matrix.size() - 1) {
                    file << row[col];
                } else {
                    file << row[col] << ",";
                }
            }
        }
        file << "\",\"";
        for (int vertex = 0; vertex < ll_res.membership.size(); ++vertex) {
            if (vertex == ll_res.membership.size() - 1) {
                file << ll_res.membership[vertex];
            } else {
                file << ll_res.membership[vertex] << ",";
            }
        }
        file << "\"";
        const Result &mdl_res = csv_row[i * 2 + 1];
//        file << "," << ll_res.first << "," << ll_res.second;
//        file << "," << std::get<0>(ll_res) << "," << std::get<1>(ll_res) << ",\"";  // ll_res.first << "," << ll_res.second;
        file << "," << mdl_res.max_influence << "," << mdl_res.avg_influence << ",\"";
//        matrix = std::get<2>(ll_res);
        const auto &mdl_matrix = mdl_res.matrix;
        for (int vertex = 0; vertex < mdl_matrix.size(); ++vertex) {
            auto row = mdl_matrix[vertex];
            for (int col = 0; col < row.size(); ++col) {
                if (vertex == mdl_matrix.size() - 1 && col == mdl_matrix.size() - 1) {
                    file << row[col];
                } else {
                    file << row[col] << ",";
                }
            }
        }
        file << "\"";  // mdl_res also has a membership, but it's the same as the ll_res membership
    }
    file << "," << mdl << "," << f1 << std::endl;
    file.close();
}

void print_neighbor_csv(const std::vector<Result>& csv_row, double mdl, double f1, const std::string& tag = "test") {
    std::ostringstream filepath_stream;
    filepath_stream << "./neighbor_influence_comparison_results/" << args.numvertices;
    std::string filepath_dir = filepath_stream.str();
    filepath_stream << "/" << args.type << ".csv";
    std::string filepath = filepath_stream.str();
    bool write_header = false;
    if (!fs::exists(filepath)) {
        fs::create_directories(filepath_dir);
        write_header = true;
    }
    std::ofstream file;
    file.open(filepath, std::ios_base::app);
    if (write_header) {
        file << "tag,numvertices,overlap,blocksizevar,undirected,algorithm";
        for (int i = 0; i < 5; ++i) {
            file << ",itr" << i << "_neighbor_max_influence,itr" << i << "_neighbor_avg_influence,itr" << i
                 << "_neighbor_matrix,itr" << i << "membership,itr" << i << "_mdl_max_influence,itr" << i
                 << "_mdl_avg_influence,itr" << i << "_mdl_matrix";
            if (i == 0)
                file << ",itr0.5_neighbor_max_influence,itr0.5_neighbor_avg_influence,itr0.5_neighbor_matrix,"
                     << "itr0.5_membership,itr0.5_mdl_max_influence,itr0.5_mdl_avg_influence,itr0.5_mdl_matrix";
        }
        file << ",mdl,f1" << std::endl;
    }
    file << tag << "," << args.numvertices << "," << args.overlap << "," << args.blocksizevar << "," << args.undirected;
    file << "," << args.algorithm;
    for (int i = 0; i < 6; ++i) {
        const Result &neighbor_res = csv_row[i * 2];
        file << "," << neighbor_res.max_influence << "," << neighbor_res.avg_influence << ",\"";
        const auto &ll_matrix = neighbor_res.matrix;
        for (int vertex = 0; vertex < ll_matrix.size(); ++vertex) {
            auto row = ll_matrix[vertex];
            for (int col = 0; col < row.size(); ++col) {
                if (vertex == ll_matrix.size() - 1 && col == ll_matrix.size() - 1) {
                    file << row[col];
                } else {
                    file << row[col] << ",";
                }
            }
        }
        file << "\",\"";
        for (int vertex = 0; vertex < neighbor_res.membership.size(); ++vertex) {
            if (vertex == neighbor_res.membership.size() - 1) {
                file << neighbor_res.membership[vertex];
            } else {
                file << neighbor_res.membership[vertex] << ",";
            }
        }
        file << "\"";
        const Result &mdl_res = csv_row[i * 2 + 1];
        file << "," << mdl_res.max_influence << "," << mdl_res.avg_influence << ",\"";
        const auto &mdl_matrix = mdl_res.matrix;
        for (int vertex = 0; vertex < mdl_matrix.size(); ++vertex) {
            auto row = mdl_matrix[vertex];
            for (int col = 0; col < row.size(); ++col) {
                if (vertex == mdl_matrix.size() - 1 && col == mdl_matrix.size() - 1) {
                    file << row[col];
                } else {
                    file << row[col] << ",";
                }
            }
        }
        file << "\"";  // mdl_res also has a membership, but it's the same as the ll_res membership
    }
    file << "," << mdl << "," << f1 << std::endl;
    file.close();
}

void print_minimal_csv(const std::vector<Result>& csv_row, double mdl, double f1, const std::string& tag = "test") {
    std::ostringstream filepath_stream;
    filepath_stream << "./influence_results_minimal/" << args.numvertices;
    std::string filepath_dir = filepath_stream.str();
    filepath_stream << "/" << args.type << ".csv";
    std::string filepath = filepath_stream.str();
    bool write_header = false;
    if (!fs::exists(filepath)) {
        fs::create_directories(filepath_dir);
        write_header = true;
    }
    std::ofstream file;
    file.open(filepath, std::ios_base::app);
    if (write_header) {
        file << "tag,numvertices,overlap,blocksizevar,undirected,algorithm";
        for (int i = 0; i < csv_row.size() - 1; ++i) {
            file << ",itr" << i << "_avg_influence,itr" << i << "membership";
            if (i == 0)
                file << ",itr0.5_avg_influence,itr0.5_membership";
        }
        file << ",mdl,f1" << std::endl;
    }
    file << tag << "," << args.numvertices << "," << args.overlap << "," << args.blocksizevar << "," << args.undirected;
    file << "," << args.algorithm;
    for (int i = 0; i < csv_row.size(); ++i) {
        const Result &neighbor_res = csv_row[i];
        file << "," << neighbor_res.avg_influence << ",\"";
        for (int vertex = 0; vertex < neighbor_res.membership.size(); ++vertex) {
            if (vertex == neighbor_res.membership.size() - 1) {
                file << neighbor_res.membership[vertex];
            } else {
                file << neighbor_res.membership[vertex] << ",";
            }
        }
        file << "\"";
    }
    file << "," << mdl << "," << f1 << std::endl;
    file.close();
}

void stochastic_block_partition(Graph &graph, const std::string &tag = "test") {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Result> csv_row;  // log-likelihood influence, mdl-influence at 0, 0.5, 1, 2, 3, ... iterations
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices(), graph, 0.5);
    auto t1 = std::chrono::high_resolution_clock::now();
    csv_row.push_back(compute_random_avg_practical_max_neighbor_influence(graph, blockmodel));
    auto t2 = std::chrono::high_resolution_clock::now() - t1;
//    csv_row.push_back(compute_influence(graph, blockmodel, false, false));
//    csv_row.push_back(compute_influence(graph, blockmodel, false, true));
    std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices() << " vertices "
              << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    int iteration = 0;
    while (!sbp::done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to "
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::merge_blocks(blockmodel, graph, graph.num_edges());
        if (iteration == 0) {
            t1 = std::chrono::high_resolution_clock::now();
            csv_row.push_back(compute_random_avg_practical_max_neighbor_influence(graph, blockmodel));
            t2 += std::chrono::high_resolution_clock::now() - t1;
//            csv_row.push_back(compute_influence(graph, blockmodel, false, false));
//            csv_row.push_back(compute_influence(graph, blockmodel, false, true));
        }
        std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs")
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else  // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        iteration++;
        t1 = std::chrono::high_resolution_clock::now();
        csv_row.push_back(compute_random_avg_practical_max_neighbor_influence(graph, blockmodel));
        t2 += std::chrono::high_resolution_clock::now() - t1;
//        csv_row.push_back(compute_influence(graph, blockmodel, false, false));
//        csv_row.push_back(compute_influence(graph, blockmodel, false, true));
        std::cout << "Done with iteration " << iteration - 1 << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto full_duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    auto influence_duration = std::chrono::duration_cast<std::chrono::seconds>(t2);
    std::cout << "influence took " << influence_duration.count() << "/" << full_duration.count() << " seconds ("
              << 100.0 * double(influence_duration.count()) / double(full_duration.count()) << ")" << std::endl;
    double mdl = entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    double f1 = evaluate::evaluate_blockmodel(graph, blockmodel).f1_score;
    print_minimal_csv(csv_row, mdl, f1, tag);
}

void stochastic_block_partition_neighbor_influence_comparison(Graph &graph, const std::string &tag = "test") {
    std::vector<Result> csv_row;  // log-likelihood influence, mdl-influence at 0, 0.5, 1, 2, 3, ... iterations
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices(), graph, 0.5);
    csv_row.push_back(compute_random_avg_practical_max_neighbor_influence(graph, blockmodel));
//    csv_row.push_back(compute_influence(graph, blockmodel, false, true));
    std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices() << " vertices "
              << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    int iteration = 0;
    while (!sbp::done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to "
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::merge_blocks(blockmodel, graph, graph.num_edges());
        if (iteration == 0) {
            csv_row.push_back(compute_random_avg_practical_max_neighbor_influence(graph, blockmodel));
//            csv_row.push_back(compute_influence(graph, blockmodel, false, true));
        }
        std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs")
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else  // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        iteration++;
        if (iteration < 5) {
            csv_row.push_back(compute_random_avg_practical_max_neighbor_influence(graph, blockmodel));
//            csv_row.push_back(compute_influence(graph, blockmodel, false, true));
        }
    }
    double mdl = entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    double f1 = evaluate::evaluate_blockmodel(graph, blockmodel).f1_score;
    print_neighbor_csv(csv_row, mdl, f1, tag);
}

int main(int argc, char* argv[]) {
    args = Args(argc, argv);
    Graph G = Graph::load();
//    Graph G = to_graph(Graph1);
    stochastic_block_partition(G, args.tag);
//    stochastic_block_partition_neighbor_influence_comparison(G, args.tag);
//    compute_influence(Graph1);
//    compute_influence(Graph2);
//    compute_influence(Graph3);
//    compute_influence(Graph4);
//    compute_influence(Graph5);
//    compute_influence(Graph6);
//    compute_influence(Graph7);
//    compute_influence(graph);
//    compute_influence(Graph1, true);
//    compute_influence(Graph2, true);
//    compute_influence(Graph3, true);
//    compute_influence(Graph4, true);
//    compute_influence(Graph5, true);
//    compute_influence(Graph6, true);
//    compute_influence(Graph7, true);
//    compute_influence(graph, true);
//    compute_influence(Graph1, false, true);
//    compute_influence(Graph2, false, true);
//    compute_influence(Graph3, false, true);
//    compute_influence(Graph4, false, true);
//    compute_influence(Graph5, false, true);
//    compute_influence(Graph6, false, true);
//    compute_influence(Graph7, false, true);
//    compute_influence(graph, false, true);
//    compute_influence(Graph1, true, true);
//    compute_influence(Graph2, true, true);
//    compute_influence(Graph3, true, true);
//    compute_influence(Graph4, true, true);
//    compute_influence(Graph5, true, true);
//    compute_influence(Graph6, true, true);
//    compute_influence(Graph7, true, true);
//    compute_influence(graph, true, true);
}
#pragma clang diagnostic pop