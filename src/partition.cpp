#include "partition.hpp"

Graph partition::partition(const Graph &graph, int rank, int num_processes, Args &args) {
    int target_num_vertices = graph.num_vertices() / num_processes;
    std::cout << "target num vertices = " << target_num_vertices << std::endl;
    if (target_num_vertices == graph.num_vertices())
        return graph;
    if (args.partition == "round_robin")
        return partition_round_robin(graph, rank, num_processes, target_num_vertices);
    if (args.partition == "random")
        return partition_random(graph, rank, num_processes, target_num_vertices);
    if (args.partition == "snowball")
        return partition_snowball(graph, rank, num_processes, target_num_vertices);
    std::cout << "The partition method " << args.partition << " doesn't exist. Defaulting to round robin." << std::endl;
    return partition_round_robin(graph, rank, num_processes, target_num_vertices);
}

Graph partition::partition_round_robin(const Graph &graph, int rank, int num_processes, int target_num_vertices) {
    NeighborList in_neighbors(target_num_vertices);
    NeighborList out_neighbors(target_num_vertices);
    int num_vertices = 0, num_edges = 0;
    std::unordered_map<int, int> translator;
    for (int i = rank; i < (int) graph.out_neighbors().size(); i += num_processes) {
        if (utils::insert(translator, i, num_vertices))
            num_vertices++;
        int from = translator[i];  // TODO: can avoid additional lookups by returning the inserted element in insert
        for (int neighbor : graph.out_neighbors(i)) {
            if ((neighbor % num_processes) - rank == 0) {
                if (utils::insert(translator, neighbor, num_vertices))
                    num_vertices++;
                int to = translator[neighbor];
                utils::insert(out_neighbors, from, to);
                utils::insert(in_neighbors, to, from);
                num_edges++;
            }
        }
    }
    std::vector<int> assignment(num_vertices, -1);
    for (const std::pair<const int, int> &element : translator) {
        assignment[element.second] = graph.assignment(element.first);
    }
    std::cout << "NOTE: rank " << rank << "/" << num_processes - 1 << " has N = " << num_vertices << " E = ";
    std::cout << num_edges << std::endl;
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, assignment);
}

Graph partition::partition_random(const Graph &graph, int rank, int num_processes, int target_num_vertices) {
    NeighborList in_neighbors(target_num_vertices);
    NeighborList out_neighbors(target_num_vertices);
    int num_vertices = 0, num_edges = 0;
    std::unordered_map<int, int> translator;
    int seed = 1234;  // TODO: make this a command-line argument
    std::vector<int> vertices = utils::range<int>(0, graph.num_vertices());
    std::shuffle(vertices.begin(), vertices.end(), std::default_random_engine(seed));
    std::vector<bool> sampled(graph.num_vertices(), false);
    for (int i = 0; i < target_num_vertices; ++i) {
        int index = (rank * target_num_vertices) + i;
        if (index >= graph.num_vertices()) break;
        sampled[vertices[index]] = true;
        translator[vertices[index]] = num_vertices;
        num_vertices++;
    }
    for (int i = 0; i < (int) graph.out_neighbors().size(); ++i) {
        if (!sampled[i]) continue;
        int from = translator[i];
        for (int neighbor : graph.out_neighbors(i)) {
            if (!sampled[neighbor]) continue;
            int to = translator[neighbor];
            utils::insert(out_neighbors, from, to);
            utils::insert(in_neighbors, to, from);
            num_edges++;
        }
    }
    std::vector<int> assignment(num_vertices, -1);
    for (const std::pair<const int, int> &element : translator) {
        assignment[element.second] = graph.assignment(element.first);
    }
    std::cout << "NOTE: rank " << rank << "/" << num_processes - 1 << " has N = " << num_vertices << " E = ";
    std::cout << num_edges << std::endl;
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, assignment);
}

Graph partition::partition_snowball(const Graph &graph, int rank, int num_processes, int target_num_vertices) {
    NeighborList in_neighbors(target_num_vertices);
    NeighborList out_neighbors(target_num_vertices);
    int num_vertices = 0, num_edges = 0;
    std::unordered_map<int, int> translator;
    // Set up random number generator
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, graph.num_vertices() - 1);
    std::vector<bool> sampled(graph.num_vertices(), false);
    std::vector<bool> neighborhood(graph.num_vertices(), false);
    std::vector<int> neighbors;
    std::vector<int> new_neighbors;
    int start;
    while (num_vertices < target_num_vertices) {
        if (neighbors.size() == 0) {  // Start/restart from a new random location
            start = distribution(generator);
            // TODO: replace this with a weighted distribution where sampled vertices have a weight of 0
            while (sampled[start]) {  // keep sampling until you find an unsampled vertex
                start = distribution(generator);
            }
            sampled[start] = true;
            neighborhood[start] = false;  // this is just a precaution, shouldn't need to be set
            translator[start] = num_vertices;
            num_vertices++;
            for (int neighbor : graph.out_neighbors(start)) {
                if (!sampled[neighbor] && !neighborhood[neighbor]) {
                    neighborhood[neighbor] = true;
                    neighbors.push_back(neighbor);
                }
            }
        }
        for (int neighbor : neighbors) {  // snowball from the current list of neighbors
            if (num_vertices == target_num_vertices) break;
            if (!sampled[neighbor]) {
                sampled[neighbor] = true;
                neighborhood[neighbor] = false;
                translator[neighbor] = num_vertices;
                num_vertices++;
                for (int new_neighbor : graph.out_neighbors(neighbor)) {
                    if (!sampled[new_neighbor] && !neighborhood[new_neighbor]) {
                        neighborhood[new_neighbor] = true;
                        new_neighbors.push_back(new_neighbor);
                    }
                }
            }
        }
        neighbors = std::vector<int>(new_neighbors);
        new_neighbors = std::vector<int>();
    }
    for (int i = 0; i < (int) graph.out_neighbors().size(); ++i) {
        if (!sampled[i]) continue;
        int from = translator[i];
        for (int neighbor : graph.out_neighbors(i)) {
            if (!sampled[neighbor]) continue;
            int to = translator[neighbor];
            utils::insert(out_neighbors, from, to);
            utils::insert(in_neighbors, to, from);
            num_edges++;
        }
    }
    std::vector<int> assignment(num_vertices, -1);
    for (const std::pair<const int, int> &element : translator) {
        assignment[element.second] = graph.assignment(element.first);
    }
    std::cout << "NOTE: rank " << rank << "/" << num_processes - 1 << " has N = " << num_vertices << " E = ";
    std::cout << num_edges << std::endl;
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, assignment);
}