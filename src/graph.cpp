#include "graph.hpp"
#include "utils.hpp"
#include "mpi_data.hpp"

void Graph::add_edge(int from, int to) {
    utils::insert_nodup(this->_out_neighbors, from , to);
    utils::insert_nodup(this->_in_neighbors, to, from);
    this->_num_edges++;
    if (from == to) {
        this->_self_edges[from] = true;
    }
    // TODO: undirected version?
}

std::vector<int> Graph::degrees() const {
    std::vector<int> vertex_degrees;
    for (int vertex = 0; vertex < this->_num_vertices; ++vertex) {
        vertex_degrees.push_back(int(this->_out_neighbors[vertex].size() + this->_in_neighbors[vertex].size()
                                 - this->_self_edges[vertex]));
    }
    return vertex_degrees;
}

Graph Graph::load() {
    // TODO: Add capability to process multiple "streaming" graph parts
    std::string base_path = utils::build_filepath();
    fs::path graph_path = base_path + ".tsv";
    fs::path truth_path = base_path + "_truePartition.tsv";
    // TODO: Handle weighted graphs
    std::vector<std::vector<std::string>> csv_contents = utils::read_csv(graph_path);
    if (csv_contents.empty()) {
        graph_path = base_path + ".mtx";
        csv_contents = utils::read_csv(graph_path);
    }
    Graph graph;
    if (csv_contents[0][0] == "%%MatrixMarket") {
        graph = Graph::load_matrix_market(csv_contents);
    } else {
        graph = Graph::load_text(csv_contents);
    }
    if (mpi.rank == 0)
        std::cout << "V: " << graph.num_vertices() << " E: " << graph.num_edges() << std::endl;

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
        assignment = utils::constant(graph.num_vertices(), 0);
    }
    graph.assignment(assignment);
    return graph;
}

/// Loads the graph if it's in a matrix market format.
Graph Graph::load_matrix_market(std::vector<std::vector<std::string>> &csv_contents) {
    if (csv_contents[0][2] != "coordinate") {
        std::cerr << "Dense matrices are not supported!" << std::endl;
        exit(-1);
    }
    if (csv_contents[0][4] == "symmetric") {
        std::cout << "Graph is symmetric" << std::endl;
        args.undirected = true;
    }
    // Find index at which edges start
    int index = 0;
    int num_vertices, num_edges;
    for (int i = 0; i < csv_contents.size(); ++i) {
        const std::vector<std::string> &line = csv_contents[i];
//        std::cout << "line: ";
//        utils::print<std::string>(line);
        if (line[0][0] == '%') continue;
        num_vertices = std::stoi(line[0]);
        if (num_vertices != std::stoi(line[1])) {
            std::cerr << "Rectangular matrices are not supported!" << std::endl;
            exit(-1);
        }
        num_edges = std::stoi(line[2]);
        index = i + 1;
        break;
    }
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    std::vector<bool> self_edges = utils::constant<bool>(num_vertices, false);
    for (int i = index; i < csv_contents.size(); ++i) {
        const std::vector<std::string> &edge = csv_contents[i];
        int from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        int to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        utils::insert_nodup(in_neighbors, to , from);
        if (args.undirected && from != to) {  // Force symmetric graph to be directed by including reverse edges.
            utils::insert_nodup(out_neighbors, to, from);
            utils::insert_nodup(in_neighbors, from , to);
            num_edges++;
        }
        if (from == to) {
            self_edges[from] = true;
        }
    }
    // Pad the neighbors lists
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<int>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<int>());
    }
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges);
}

/// Loads the graph if it's in a text format: a list of "from to" string pairs.
Graph Graph::load_text(std::vector<std::vector<std::string>> &csv_contents) {
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    std::vector<bool> self_edges;
    int num_vertices = 0;
    if (args.undirected)
        Graph::parse_undirected(in_neighbors, out_neighbors, num_vertices, self_edges, csv_contents);
    else
        Graph::parse_directed(in_neighbors, out_neighbors, num_vertices, self_edges, csv_contents);
    int num_edges = 0;  // TODO: unnecessary re-counting of edges?
    for (const std::vector<int> &neighborhood : out_neighbors) {
        num_edges += (int)neighborhood.size();
    }
    if (args.undirected) {
        num_edges /= 2;
    }
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges);
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
            int deg_out_i = int(this->_out_neighbors[vertex_i].size());
            int deg_in_j = int(this->_in_neighbors[vertex_j].size());
            double temp = edge_weight - (double(deg_out_i * deg_in_j) / double(this->_num_edges));
            result += temp;
        }
    }
    result /= double(this->_num_edges);
    return result;
}

void Graph::parse_directed(NeighborList &in_neighbors, NeighborList &out_neighbors, int &num_vertices,
                           std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        int from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        int to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        utils::insert_nodup(in_neighbors, to, from);
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
}

void Graph::parse_undirected(NeighborList &in_neighbors, NeighborList &out_neighbors, int &num_vertices,
                             std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        int from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        int to = std::stoi(edge[1]) - 1;
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        utils::insert_nodup(out_neighbors, from , to);
        if (from != to)
            utils::insert_nodup(out_neighbors, to, from);
        while (self_edges.size() < num_vertices) {
            self_edges.push_back(false);
        }
        if (from == to) {
            self_edges[from] = true;
        }
    }
    in_neighbors = NeighborList(out_neighbors);
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<int>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<int>());
    }
}

void Graph::sort_vertices() {
    std::vector<int> vertex_degrees = this->degrees();
    std::vector<int> indices = utils::range<int>(0, this->_num_vertices);
    std::sort(indices.data(), indices.data() + indices.size(),  // sort in descending order
              [vertex_degrees](size_t i1, size_t i2) { return vertex_degrees[i1] > vertex_degrees[i2]; });
    for (int index = 0; index < this->_num_vertices; ++index) {
        int vertex = indices[index];
        if (index < 0.15 * this->_num_vertices) {
            this->_high_degree_vertices.push_back(vertex);
        } else {
            this->_low_degree_vertices.push_back(vertex);
        }
    }
}
