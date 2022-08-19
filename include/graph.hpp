/***
 * Stores a Graph.
 */
#ifndef SBP_GRAPH_HPP
#define SBP_GRAPH_HPP

#include <iostream>
#include <string>

#include "blockmodel/sparse/typedefs.hpp"
#include "fs.hpp"
#include "utils.hpp"

// TODO: replace _out_neighbors and _in_neighbors with our DictTransposeMatrix
class Graph {
public:
    explicit Graph(int num_vertices) {
        this->_num_vertices = num_vertices;
        this->_num_edges = 0;
        this->_self_edges = utils::constant<bool>(num_vertices, false);
        this->_assignment = utils::constant<int>(num_vertices, -1);
        while (this->_out_neighbors.size() < size_t(num_vertices)) {
            this->_out_neighbors.push_back(std::vector<int>());
        }
        while (this->_in_neighbors.size() < size_t(num_vertices)) {
            this->_in_neighbors.push_back(std::vector<int>());
        }
    }
    Graph(NeighborList &out_neighbors, NeighborList &in_neighbors, int num_vertices, int num_edges,
          const std::vector<bool> &self_edges = std::vector<bool>(),
          const std::vector<int> &assignment = std::vector<int>()) {
        this->_out_neighbors = out_neighbors;
        this->_in_neighbors = in_neighbors;
        this->_num_vertices = num_vertices;
        this->_num_edges = num_edges;
        this->_self_edges = self_edges;
        this->_assignment = assignment;
        this->sort_vertices();
    }
    Graph() = default;
    /// Loads the graph. Assumes the file is saved in the following directory:
    /// <args.directory>/<args.type>/<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar
    /// Assumes the graph file is named:
    /// <args.type>_<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar_<args.numvertices>_nodes.tsv
    /// Assumes the true assignment file is named:
    /// <args.type>_<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar_<args.numvertices>_trueBlockmodel.tsv
    static Graph load();
    /// Loads the graph if it's in a matrix market format.
    static Graph load_matrix_market(std::vector<std::vector<std::string>> &csv_contents);
    /// Loads the graph if it's in a text format: a list of "from to" string pairs.
    static Graph load_text(std::vector<std::vector<std::string>> &csv_contents);
    //============================================
    // GETTERS & SETTERS
    //============================================
    /// Adds an edge to the graph
    void add_edge(int from, int to);
    /// Returns a const reference to the assignmnet
    const std::vector<int> &assignment() const { return this->_assignment; }
    /// Sets the assignment vector for the given graph
    void assignment(const std::vector<int> &assignment_vector) { this->_assignment = assignment_vector; }
    /// Returns the block/community assignment of vertex `v`
    int assignment(int v) const { return this->_assignment[v]; }
    /// Sets the assignment of vertex `v` to block `b`
    void assign(int v, int b) { this->_assignment[v] = b; }
    /// Returns a vector containing the vertex degrees for every vertex in the graph
    std::vector<int> degrees() const;
    /// Returns a const reference to the in neighbors
    const NeighborList &in_neighbors() const { return this->_in_neighbors; }
    /// Returns a const reference to the in neighbors of vertex `v`
    const std::vector<int> &in_neighbors(int v) const { return this->_in_neighbors[v]; }
    /// Returns the list of high degree vertices
    const std::vector<int> &high_degree_vertices() const { return this->_high_degree_vertices; }
    /// Returns the list of low degree vertices
    const std::vector<int> &low_degree_vertices() const { return this->_low_degree_vertices; }
    /// Calculates the modularity of this graph given a particular vertex-to-block `assignment`
    double modularity(const std::vector<int> &assignment) const;
    /// Returns the number of edges in this graph
    int num_edges() const { return this->_num_edges; }
    /// Returns the number of vertices in this graph
    int num_vertices() const { return this->_num_vertices; }
    /// Returns a const reference to the out neighbors
    const NeighborList &out_neighbors() const { return this->_out_neighbors; }
    /// Returns a const reference to the out neighbors of vertex `v`
    const std::vector<int> &out_neighbors(int v) const { return this->_out_neighbors[v]; }
    /// Sorts the vertices into low and high degree vertices
    void sort_vertices();
private:
    /// For every vertex, stores the community they belong to.
    /// If assignment[v] = -1, then the community of v is not known
    std::vector<int> _assignment;
    /// Stores true if vertex is one of the highest degree vertices
//    MapVector<bool> _high_degree_vertex;
    /// Stores a list of the high degree vertices
    std::vector<int> _high_degree_vertices;
    /// Stores a list of the low degree vertices
    std::vector<int> _low_degree_vertices;
    /// For every vertex, stores the incoming neighbors as a std::vector<int>
    NeighborList _in_neighbors;
    /// For every vertex, stores the outgoing neighbors as a std::vector<int>
    NeighborList _out_neighbors;
    /// The number of vertices in the graph
    int _num_vertices;
    /// The number of edges in the graph
    int _num_edges;
    /// Stores true if a vertex has self edges, false otherwise
    std::vector<bool> _self_edges;
    /// Parses a directed graph from csv contents
    static void parse_directed(NeighborList &in_neighbors, NeighborList &out_neighbors, int &num_vertices,
                               std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents);
    /// Parses an undirected graph from csv contents
    static void parse_undirected(NeighborList &in_neighbors, NeighborList &out_neighbors, int &num_vertices,
                                 std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents);
};

#endif // SBP_GRAPH_HPP
