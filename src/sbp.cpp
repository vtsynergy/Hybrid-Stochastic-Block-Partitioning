#include "sbp.hpp"

#include "block_merge.hpp"
#include "blockmodel/dist_blockmodel.hpp"
#include "entropy.hpp"
#include "finetune.hpp"
#include "fs.hpp"
#include "mpi_data.hpp"

#include "assert.h"
#include <fenv.h>
#include <sstream>

namespace sbp {

std::vector<Intermediate> intermediate_results;

std::vector<Intermediate> get_intermediates() {
    return intermediate_results;
}

void write_results(float iteration, std::ofstream &file, const Graph &graph, const Blockmodel &blockmodel, double mdl) {
    // fedisableexcept(FE_INVALID | FE_OVERFLOW);
    file << args.tag << "," << graph.num_vertices() << "," << args.overlap << "," << args.blocksizevar << ",";
    file << args.undirected << "," << args.algorithm << "," << iteration << ",";
    file << mdl << ","  << entropy::normalize_mdl_v1(mdl, graph.num_edges()) << ",";
    file << entropy::normalize_mdl_v2(mdl, graph.num_vertices(), graph.num_edges()) << ",";
    file << graph.modularity(blockmodel.block_assignment()) << "," << blockmodel.interblock_edges() << ",";
    file << blockmodel.block_size_variation() << std::endl;
    // feenableexcept(FE_INVALID | FE_OVERFLOW);
}

void add_intermediate(float iteration, const Graph &graph, const Blockmodel &blockmodel, double mdl) {
    double normalized_mdl_v1 = entropy::normalize_mdl_v1(mdl, graph.num_edges());
    double normalized_mdl_v2 = entropy::normalize_mdl_v2(mdl, graph.num_vertices(), graph.num_edges());
    double modularity = -1;
    if (iteration == -1)
        modularity = graph.modularity(blockmodel.block_assignment());
    double interblock_edges = blockmodel.interblock_edges();
    // fedisableexcept(FE_INVALID | FE_OVERFLOW);
    double block_size_variation = blockmodel.block_size_variation();
    // feenableexcept(FE_INVALID | FE_OVERFLOW);
    Intermediate intermediate {};
    intermediate.iteration = iteration;
    intermediate.mdl = mdl;
    intermediate.normalized_mdl_v1 = normalized_mdl_v1;
    intermediate.normalized_mdl_v2 = normalized_mdl_v2;
    intermediate.modularity = modularity;
    intermediate.interblock_edges = interblock_edges;
    intermediate.block_size_variation = block_size_variation;
    intermediate.mcmc_iterations = finetune::MCMC_iterations;
    intermediate.mcmc_time = finetune::MCMC_time;
    intermediate_results.push_back(intermediate);
    std::cout << "Iteration " << iteration << " MDL: " << mdl << " v1 normalized: " << normalized_mdl_v1
              << " v2 normalized: " << normalized_mdl_v2 << " modularity: " << modularity
              << " interblock edge %: " << interblock_edges << " block size variation: " << block_size_variation
              << " MCMC iterations: " << finetune::MCMC_iterations << " MCMC time: "
              << finetune::MCMC_time << std::endl;
}

Blockmodel stochastic_block_partition(Graph &graph, Args &args) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    Blockmodel blockmodel(graph.num_vertices(), graph, float(BLOCK_REDUCTION_RATE));
    double initial_mdl = entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    add_intermediate(0, graph, blockmodel, initial_mdl);
    BlockmodelTriplet blockmodel_triplet = BlockmodelTriplet();
    float iteration = 0;
    while (!done_blockmodeling(blockmodel, blockmodel_triplet)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::merge_blocks(blockmodel, graph, graph.num_edges());
        if (iteration < 1) {
            double mdl = entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
            add_intermediate(0.5, graph, blockmodel, mdl);
        }
        std::cout << "Starting MCMC vertex moves" << std::endl;
        double start = MPI_Wtime();
        if (args.algorithm == "async_gibbs" && iteration < float(args.asynciterations))
            blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else if (args.algorithm == "hybrid_mcmc")
            blockmodel = finetune::hybrid_mcmc(blockmodel, graph, blockmodel_triplet);
        else // args.algorithm == "metropolis_hastings"
            blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
//        iteration++;
        finetune::MCMC_time += MPI_Wtime() - start;
        add_intermediate(++iteration, graph, blockmodel, blockmodel.getOverall_entropy());
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
    }
    // only last iteration result will calculate expensive modularity
    add_intermediate(-1, graph, blockmodel, blockmodel.getOverall_entropy());
    return blockmodel;
}

bool done_blockmodeling(Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet, int min_num_blocks) {
    if (min_num_blocks > 0) {
        if ((blockmodel.getNum_blocks() <= min_num_blocks) || (blockmodel_triplet.get(2).empty == false)) {
            return true;
        }
    }
    if (blockmodel_triplet.optimal_num_blocks_found) {
        blockmodel_triplet.status();
        std::cout << "Optimal number of blocks was found" << std::endl;
        return true;
    }
    return false;
}

namespace dist {

// Blockmodel stochastic_block_partition(Graph &graph, MPI &mpi, Args &args) {
Blockmodel stochastic_block_partition(Graph &graph, Args &args) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    // DistBlockmodel blockmodel(graph, args, mpi);
    TwoHopBlockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), BLOCK_REDUCTION_RATE);
    // Blockmodel blockmodel(graph.num_vertices(), graph.out_neighbors(), BLOCK_REDUCTION_RATE);
    if (mpi.rank == 0)
        std::cout << "Performing stochastic block blockmodeling on graph with " << graph.num_vertices() << " vertices "
                  << " and " << blockmodel.getNum_blocks() << " blocks." << std::endl;
    DistBlockmodelTriplet blockmodel_triplet = DistBlockmodelTriplet();
    int iteration = 0;
    while (!dist::done_blockmodeling(blockmodel, blockmodel_triplet, 0)) {
        if (mpi.rank == 0 && blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.getNum_blocks() << " to " 
                      << blockmodel.getNum_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        blockmodel = block_merge::dist::merge_blocks(blockmodel, graph.out_neighbors(), graph.num_edges());
        if (mpi.rank == 0) std::cout << "Starting MCMC vertex moves" << std::endl;
        if (args.algorithm == "async_gibbs" && iteration < args.asynciterations)
            blockmodel = finetune::dist::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet);
        else
            blockmodel = finetune::dist::metropolis_hastings(blockmodel, graph, blockmodel_triplet);
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        iteration++;
    }
    std::cout << "Total MCMC iterations: " << finetune::MCMC_iterations << std::endl;
    return blockmodel;
}

bool done_blockmodeling(TwoHopBlockmodel &blockmodel, DistBlockmodelTriplet &blockmodel_triplet, int min_num_blocks) {
    if (mpi.rank == 0) std::cout << "distributed done_blockmodeling" << std::endl;
    if (min_num_blocks > 0) {
        if ((blockmodel.getNum_blocks() <= min_num_blocks) || (blockmodel_triplet.get(2).empty == false)) {
            return true;
        }
    }
    if (blockmodel_triplet.optimal_num_blocks_found) {
        blockmodel_triplet.status();
        std::cout << "Optimal number of blocks was found" << std::endl;
        return true;
    }
    return false;
}

}  // namespace dist

}  // namespace sbp
