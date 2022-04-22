/***
 * Sparse Matrix that uses a vector of unordered maps to store the blockmodel.
 */
#ifndef CPPSBP_PARTITION_SPARSE_DICT_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_DICT_MATRIX_HPP

#include <unordered_map>

#include "csparse_matrix.hpp"
#include "delta.hpp"
// TODO: figure out where to put utils.hpp so this never happens
#include "../../utils.hpp"

// #include <Eigen/Core>

/**
 * The basic list-of-maps sparse matrix.
 */
class DictMatrix : public ISparseMatrix {
  public:
    DictMatrix() = default;
    DictMatrix(int nrows, int ncols) {  // : ncols(ncols), nrows(nrows) {
        this->ncols = ncols;
        this->nrows = nrows;
        // this->matrix = boost::numeric::ublas::coordinate_matrix<int>(this->nrows, this->ncols);
//        this->matrix = std::vector<std::unordered_map<int, int>>(this->nrows, std::unordered_map<int, int>());
        this->matrix = std::vector<MapVector<int>>(this->nrows, MapVector<int>());
        // this->matrix = boost::numeric::ublas::mapped_matrix<int>(this->nrows, this->ncols);
        // int shape_array[2] = {this->nrows, this->ncols};
        this->shape = std::make_pair(this->nrows, this->ncols);
    }
    void add(int row, int col, int val) override;
    // virtual void add(int row, std::vector<int> cols, std::vector<int> values) override;
    void clearrow(int row) override;
    void clearcol(int col) override;
    ISparseMatrix* copy() const override;
    std::vector<std::tuple<int, int, int>> entries() const override;
    int get(int row, int col) const override;
    std::vector<int> getcol(int col) const override;
    MapVector<int> getcol_sparse(int col) const override;
    void getcol_sparse(int col, MapVector<int> &col_vector) const override;
    // virtual MapVector<int> getcol_sparse(int col) override;
    // virtual const MapVector<int>& getcol_sparse(int col) const override;
    std::vector<int> getrow(int row) const override;
    MapVector<int> getrow_sparse(int row) const override;
    void getrow_sparse(int row, MapVector<int> &row_vector) const override;
    // virtual MapVector<int> getrow_sparse(int row) override;
    // virtual const MapVector<int>& getrow_sparse(int row) const override;
    EdgeWeights incoming_edges(int block) const override;
    std::set<int> neighbors(int block) const override;
    Indices nonzero() const override;
    EdgeWeights outgoing_edges(int block) const override;
    /// Sets the values in a row equal to the input vector
    void setrow(int row, const MapVector<int> &vector) override;
    /// Sets the values in a column equal to the input vector
    void setcol(int col, const MapVector<int> &vector) override;
    void sub(int row, int col, int val) override;
    int edges() const override;
    void print() const override;
    std::vector<int> sum(int axis) const override;
    int trace() const override;
    void update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
                            std::vector<int> proposed_row, std::vector<int> current_col,
                            std::vector<int> proposed_col) override;
    void update_edge_counts(int current_block, int proposed_block, MapVector<int> current_row,
                            MapVector<int> proposed_row, MapVector<int> current_col,
                            MapVector<int> proposed_col) override;
    void update_edge_counts(const Delta &delta) override;
    bool validate(int row, int col, int val) const override;
    std::vector<int> values() const override;

  private:
    // void check_row_bounds(int row);
    // void check_col_bounds(int col);
    // int ncols;
    // int nrows;
//    std::vector<std::unordered_map<int, int>> matrix;
    std::vector<MapVector<int>> matrix;
    // boost::numeric::ublas::mapped_matrix<int> matrix;
    // boost::numeric::ublas::coordinate_matrix<int> matrix;
};

#endif // CPPSBP_PARTITION_SPARSE_DICT_MATRIX_HPP
