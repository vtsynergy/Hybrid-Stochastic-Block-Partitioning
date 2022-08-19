
#include <chrono>
#include <execinfo.h>
#include <iostream>
#include <mpi.h>
#include <signal.h>
#include <string>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "entropy.hpp"
#include "evaluate.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "partition.hpp"
#include "sbp.hpp"



MPI_t mpi;
Args args;

void write_results(const Graph &graph, const evaluate::Eval &eval, double runtime) {
    std::vector<sbp::Intermediate> intermediate_results = sbp::get_intermediates();
    std::ostringstream filepath_stream;
    filepath_stream << args.csv << args.numvertices;
    std::string filepath_dir = filepath_stream.str();
    filepath_stream << "/" << args.type << ".csv";
    std::string filepath = filepath_stream.str();
    bool file_exists = fs::exists(filepath);
    std::cout << std::boolalpha <<  "writing results to " << filepath << " exists = " << file_exists << std::endl;
    fs::create_directories(fs::path(filepath_dir));
    std::ofstream file;
    file.open(filepath, std::ios_base::app);
    if (!file_exists) {
        file << "tag, numvertices, numedges, overlap, blocksizevar, undirected, algorithm, iteration, mdl, "
             << "normalized_mdl_v1, normalized_mdl_v2, modularity, interblock_edges, block_size_variation, f1_score, "
             << "nmi, true_mdl, true_mdl_v1, true_mdl_v2, runtime, mcmc_iterations" << std::endl;
    }
    for (const sbp::Intermediate temp : intermediate_results) {
        file << args.tag << ", " << graph.num_vertices() << ", " << graph.num_edges() << ", " << args.overlap << ", "
             << args.blocksizevar << ", " << args.undirected << ", " << args.algorithm << ", " << temp.iteration << ", "
             << temp.mdl << ", " << temp.normalized_mdl_v1 << ", " << temp.normalized_mdl_v2 << ", "
             << temp.modularity << ", " << temp.interblock_edges << ", " << temp.block_size_variation << ", "
             << eval.f1_score << ", " << eval.nmi << ", " << eval.true_mdl << ", "
             << entropy::normalize_mdl_v1(eval.true_mdl, graph.num_edges()) << ", "
             << entropy::normalize_mdl_v2(eval.true_mdl, graph.num_vertices(), graph.num_edges()) << ", "
             << runtime << ", " << temp.mcmc_iterations << std::endl;
    }
    file.close();
}

int main(int argc, char* argv[]) {
    // int rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.num_processes);

    args = Args(argc, argv);

    if (mpi.rank == 0) {
        std::cout << "Number of processes = " << mpi.num_processes << std::endl;
    }
    Graph graph = Graph::load();

    if (mpi.num_processes > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        Blockmodel blockmodel = sbp::dist::stochastic_block_partition(graph, args);
        double end = MPI_Wtime();
        if (mpi.rank == 0) {
            evaluate::Eval result = evaluate::evaluate_blockmodel(graph, blockmodel);
            std::cout << "Final F1 score = " << result.f1_score << std::endl;
            std::cout << "Community detection runtime = " << end - start << "s" << std::endl;
            write_results(graph, result, end - start);
        }
    } else {
        auto start = std::chrono::steady_clock::now();
        Blockmodel blockmodel = sbp::stochastic_block_partition(graph, args);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> runtime = end - start;
        evaluate::Eval result = evaluate::evaluate_blockmodel(graph, blockmodel);
        std::cout << "Final F1 score = " << result.f1_score << std::endl;
        std::cout << "Community detection runtime = " << runtime.count() << "s" << std::endl;
        write_results(graph, result, runtime.count());
    }

    MPI_Finalize();
}
