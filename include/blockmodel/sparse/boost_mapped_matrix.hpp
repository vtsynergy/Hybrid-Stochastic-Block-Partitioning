/***
 * Sparse Matrix based on the Boost mapped_matrix
 */
#ifndef CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_HPP

#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "csparse_matrix.hpp"
// TODO: figure out where to put utils.hpp so this never happens
#include "../../utils.hpp"

// #include <Eigen/Core>

/**
 * C++ interface of the dictionary (map of maps) sparse matrix
 */
// class BoostMappedMatrix : public CSparseMatrix {
class BoostMappedMatrix {
  public:
    BoostMappedMatrix() {}
    BoostMappedMatrix(int nrows, int ncols) : ncols(ncols), nrows(nrows) {
        // this->matrix = boost::numeric::ublas::coordinate_matrix<int>(this->nrows, this->ncols);
        this->matrix = boost::numeric::ublas::compressed_matrix<int>(this->nrows, this->ncols);
        // this->matrix = boost::numeric::ublas::mapped_matrix<int>(this->nrows, this->ncols);
        // int shape_array[2] = {this->nrows, this->ncols};
        this->shape = std::make_pair(this->nrows, this->ncols);
    }
    void add(int row, int col, int val);
    void add(int row, std::vector<int> cols, std::vector<int> values);
    BoostMappedMatrix copy();
    int get(int row, int col);
    std::vector<int> getcol(int col);
    std::vector<int> getrow(int row);
    EdgeWeights incoming_edges(int block);
    Indices nonzero();
    EdgeWeights outgoing_edges(int block);
    void sub(int row, int col, int val);
    int sum();
    std::vector<int> sum(int axis = 0);
    int trace();
    void update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
                            std::vector<int> proposed_row, std::vector<int> current_col, std::vector<int> proposed_col);
    std::vector<int> values();
    std::pair<int, int> shape;

  private:
    void check_row_bounds(int row);
    void check_col_bounds(int col);
    int ncols;
    int nrows;
    boost::numeric::ublas::compressed_matrix<int> matrix;
    // boost::numeric::ublas::mapped_matrix<int> matrix;
    // boost::numeric::ublas::coordinate_matrix<int> matrix;
};

#endif // CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_HPP
