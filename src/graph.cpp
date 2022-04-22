#include "graph.hpp"
#include "utils.hpp"
#include "mpi_data.hpp"

Graph Graph::load(Args &args) {
    // TODO: Add capability to process multiple "streaming" graph parts
    std::string base_path = utils::build_filepath();
    fs::path graph_path = base_path + ".tsv";
    fs::path truth_path = base_path + "_truePartition.tsv";
    // TODO: Handle weighted graphs
    std::vector<std::vector<std::string>> csv_contents = utils::read_csv(graph_path);
    NeighborList out_neighbors;
    NeighborList in_neighbors;
//    size_t num_edges = csv_contents.size();
    int num_vertices = 0;
    if (args.undirected)
        Graph::parse_undirected(in_neighbors, out_neighbors, num_vertices, csv_contents);
    else
        Graph::parse_directed(in_neighbors, out_neighbors, num_vertices, csv_contents);
    // std::cout << "num_edges before: " << num_edges << std::endl;
    int num_edges = 0;  // TODO: unnecessary re-counting of edges?
    for (const std::vector<int> &neighborhood : out_neighbors) {
        num_edges += (int)neighborhood.size();
//        for (int neighbor : neighborhood) {
//            num_edges++;
//        }
    }
    if (args.undirected) {
        num_edges /= 2;
    }
    // std::cout << "num_edges after: " << num_edges << std::endl;
    if (mpi.rank == 0)
        std::cout << "V: " << num_vertices << " E: " << num_edges << std::endl;

    csv_contents = utils::read_csv(truth_path);
    std::vector<int> assignment;
    // TODO: vertices, communities should be size_t or uint. Will need to make sure -1 returns are properly handled
    // elsewhere.
    if (!csv_contents.empty()) {
        for (std::vector<std::string> &assign: csv_contents) {
            int vertex = std::stoi(assign[0]) - 1;
            int community = std::stoi(assign[1]) - 1;
            if (vertex >= (int)assignment.size()) {
                std::vector<int> padding(vertex - assignment.size() + 1, -1);
                assignment.insert(assignment.end(), padding.begin(), padding.end());
            }
            assignment[vertex] = community;
        }
    } else {
        assignment = utils::constant(num_vertices, 0);
    }
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, assignment);
}

double Graph::modularity(const std::vector<int> &assignment) const {
    // See equation for Q_d in: https://hal.archives-ouvertes.fr/hal-01231784/document
    double result = 0.0;
    for (int vertex_i = 0; vertex_i < this->_num_vertices; ++vertex_i) {
        for (int vertex_j = 0; vertex_j < this->_num_vertices; ++vertex_j) {
            if (assignment[vertex_i] != assignment[vertex_j]) continue;
            int edge_weight = 0.0;
            for (int neighbor : this->_out_neighbors[vertex_i]) {
                if (neighbor == vertex_j) {
                    edge_weight = 1.0;
                    break;
                }
            }
            int deg_out_i = this->_out_neighbors[vertex_i].size();
            int deg_in_j = this->_in_neighbors[vertex_j].size();
            double temp = edge_weight - (double(deg_out_i * deg_in_j) / double(this->_num_edges));
            result += temp;
        }
    }
    result /= double(this->_num_edges);
    return result;
}

void Graph::parse_directed(NeighborList &in_neighbors, NeighborList &out_neighbors, int &num_vertices,
                           std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        int from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        int to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        utils::insert_nodup(in_neighbors, to, from);
    }
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<int>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<int>());
    }
}

void Graph::parse_undirected(NeighborList &in_neighbors, NeighborList &out_neighbors, int &num_vertices,
                             std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        int from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        int to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        if (from != to)
            utils::insert_nodup(out_neighbors, to, from);
        in_neighbors = NeighborList(out_neighbors);
    }
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<int>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<int>());
    }
}
