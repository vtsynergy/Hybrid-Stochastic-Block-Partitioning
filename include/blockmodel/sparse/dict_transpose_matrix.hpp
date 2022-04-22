/***
 * Sparse Matrix that uses a vector of unordered maps to store the blockmodel.
 */
#ifndef CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP

#include <map>
#include <unordered_map>

#include "csparse_matrix.hpp"
// TODO: figure out where to put utils.hpp so this never happens
#include "delta.hpp"
#include "typedefs.hpp"
#include "../../utils.hpp"

// #include <Eigen/Core>

/**
 * The list-of-maps sparse matrix, with a transpose for faster column indexing.
 * TODO: figure out where 0s are being added to the matrix, and whether or not we need to get rid of that
 */
class DictTransposeMatrix : public ISparseMatrix {
  public:
    DictTransposeMatrix() = default;
    DictTransposeMatrix(int nrows, int ncols) {  // : ncols(ncols), nrows(nrows) {
        this->ncols = ncols;
        this->nrows = nrows;
//        this->matrix = std::vector<std::unordered_map<int, int>>(this->nrows, std::unordered_map<int, int>());
//        this->matrix_transpose = std::vector<std::unordered_map<int, int>>(this->ncols, std::unordered_map<int, int>());
        this->matrix = std::vector<MapVector<int>>(this->nrows, MapVector<int>());
        this->matrix_transpose = std::vector<MapVector<int>>(this->ncols, MapVector<int>());
        this->shape = std::make_pair(this->nrows, this->ncols);
    }
    virtual void add(int row, int col, int val) override;
    virtual void add_transpose(int row, int col, int val);
    // virtual void add(int row, std::vector<int> cols, std::vector<int> values) override;
    /// Clears the value in a given row. Complexity ~O(number of blocks).
    virtual void clearrow(int row) override;
    /// Clears the values in a given column. Complexity ~O(number of blocks).
    virtual void clearcol(int col) override;
    /// Returns a copy of the current matrix.
    virtual ISparseMatrix* copy() const override;
    std::vector<std::tuple<int, int, int>> entries() const override;
    virtual int get(int row, int col) const override;
    /// Returns all values in the requested column as a dense vector.
    virtual std::vector<int> getcol(int col) const override;
    /// Returns all values in the requested column as a sparse vector (ordered map).
    virtual MapVector<int> getcol_sparse(int col) const override;
    /// Returns all values in the requested column as a sparse vector (ordered map) reference.
    // const MapVector<int>& getcol_sparse(int col) const;
    virtual void getcol_sparse(int col, MapVector<int> &col_vector) const override;
    /// Returns all values in the requested row as a dense vector.
    virtual std::vector<int> getrow(int row) const override;
    /// Returns all values in the requested column as a sparse vector (ordered map).
    virtual MapVector<int> getrow_sparse(int row) const override;
    /// Returns all values in the requested column as a sparse vector (ordered map) reference.
    // const MapVector<int>& getrow_sparse(int row) const;
    virtual void getrow_sparse(int row, MapVector<int> &row_vector) const override;
    virtual EdgeWeights incoming_edges(int block) const override;
    std::set<int> neighbors(int block) const override;
    virtual Indices nonzero() const override;
    virtual EdgeWeights outgoing_edges(int block) const override;
    /// Sets the values in a row equal to the input vector.
    virtual void setrow(int row, const MapVector<int> &vector) override;
    /// Sets the values in a column equal to the input vector.
    virtual void setcol(int col, const MapVector<int> &vector) override;
    virtual void sub(int row, int col, int val) override;
    virtual int edges() const override;
    void print() const override;
    virtual std::vector<int> sum(int axis = 0) const override;
    virtual int trace() const override;
    virtual void update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
                                    std::vector<int> proposed_row, std::vector<int> current_col,
                                    std::vector<int> proposed_col) override;
    void update_edge_counts(int current_block, int proposed_block, MapVector<int> current_row,
                            MapVector<int> proposed_row, MapVector<int> current_col,
                            MapVector<int> proposed_col) override;
    void update_edge_counts(const Delta &delta) override;
    bool validate(int row, int col, int val) const override;
    virtual std::vector<int> values() const override;

  private:
    // void check_row_bounds(int row);
    // void check_col_bounds(int col);
    // void check_row_bounds(int row) const;
    // void check_col_bounds(int col) const;
    // int ncols;
    // int nrows;
//    std::vector<std::unordered_map<int, int>> matrix;
//    std::vector<std::unordered_map<int, int>> matrix_transpose;
    std::vector<MapVector<int>> matrix;
    std::vector<MapVector<int>> matrix_transpose;
};

#endif // CPPSBP_PARTITION_SPARSE_DICT_TRANSPOSE_MATRIX_HPP
