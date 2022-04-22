/***
 * Common interface for sparse matrix types.
 */
#ifndef CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP

#include <exception>
#include <iostream>
#include <memory>
#include <sstream>

// #include <Eigen/Core>
#include "delta.hpp"
#include "typedefs.hpp"

// typedef Eigen::VectorXi Vector;

typedef struct edge_weights_t {
    std::vector<int> indices;
    std::vector<int> values;

    void print() {
        if (this->indices.empty()) {
            std::cout << "[]" << std::endl;
            return;
        }
        std::cout << "[" << this->indices[0] << ": " << this->values[0] << ", ";
        for (size_t num_printed = 1; num_printed < this->indices.size() - 1; num_printed++) {
            if (num_printed % 25 == 0) {
                std::cout << std::endl << " ";
            }
            std::cout << this->indices[num_printed] << ": " << this->values[num_printed] << ", ";
        }
        std::cout << this->indices[this->indices.size() - 1] << ": " << this->values[this->indices.size() - 1] << "]" << std::endl;
    }
} EdgeWeights;

typedef struct indices_t {
    std::vector<int> rows;
    std::vector<int> cols;
} Indices;

class IndexOutOfBoundsException: public std::exception {
public:
    IndexOutOfBoundsException(int index, int max) { // } : index(index), max(max) {
        std::ostringstream message_stream;
        message_stream << "Index " << index << " is out of bounds [0, " << max - 1 << "]";
        this->message = message_stream.str();
    }
    const char* what() const noexcept override {
        return this->message.c_str();
    }
private:
//    int index;
//    int max;
    std::string message;
};

///
/// C++ implementation of the sparse matrix interface
///
class ISparseMatrix {
public:
    // ISparseMatrix() {}
    virtual ~ISparseMatrix() = default;
    /// Add `val` to `matrix[row, col]`.
    virtual void add(int row, int col, int val) = 0;
    // virtual void add(int row, std::vector<int> cols, std::vector<int> values) = 0;
    /// Set matrix row `row` to empty.
    virtual void clearrow(int row) = 0;
    /// Set matrix column `col` to empty.
    virtual void clearcol(int col) = 0;
    /// Returns a copy of this matrix.
    virtual ISparseMatrix* copy() const = 0;
    /// Returns matrix entries in the form `std::tuple<int, int, int`.
    virtual std::vector<std::tuple<int, int, int>> entries() const = 0;
    /// Returns the value in `matrix[row, col]`.
    virtual int get(int row, int col) const = 0;
    /// Returns the column `col` as a dense vector.
    virtual std::vector<int> getcol(int col) const = 0;
    /// Returns the column `col` as a sparse vector.
    virtual MapVector<int> getcol_sparse(int col) const = 0;
    /// Populates the values in `col_vector` with the values of column `col`.
    virtual void getcol_sparse(int col, MapVector<int> &col_vector) const = 0;
    // virtual const MapVector<int>& getcol_sparse(int col) const = 0;
    /// Returns the row `row` as a dense vector.
    virtual std::vector<int> getrow(int row) const = 0;
    /// Returns the row `row` as a sparse vector.
    virtual MapVector<int> getrow_sparse(int row) const = 0;
    /// Populates the values in `row_vector` with the values of row `row`.
    virtual void getrow_sparse(int row, MapVector<int> &row_vector) const = 0;
    // virtual const MapVector<int>& getrow_sparse(int row) const = 0;
    /// TODO: docstring
    virtual EdgeWeights incoming_edges(int block) const = 0;
    /// Returns the set of all neighbors of `block`. This includes `block` if it has self-edges.
    virtual std::set<int> neighbors(int block) const = 0;
    /// TODO: docstring
    virtual Indices nonzero() const = 0;
    /// TODO: docstring
    virtual EdgeWeights outgoing_edges(int block) const = 0;
    /// Sets the values in a row equal to the input vector `vector`.
    virtual void setrow(int row, const MapVector<int> &vector) = 0;
    /// Sets the values in a column equal to the input vector `vector`.
    virtual void setcol(int col, const MapVector<int> &vector) = 0;
    /// Subtracts `val` from `matrix[row, col]`.
    virtual void sub(int row, int col, int val) = 0;
    virtual int edges() const = 0;
    virtual void print() const = 0;
    virtual std::vector<int> sum(int axis = 0) const = 0;
    virtual int trace() const = 0;
    /// Updates the blockmatrix values in the rows and columns corresponding to `current_block` and `proposed_block`.
    virtual void update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
                                    std::vector<int> proposed_row, std::vector<int> current_col,
                                    std::vector<int> proposed_col) = 0;
    /// Updates the blockmatrix values in the rows and columns corresponding to `current_block` and `proposed_block`.
    virtual void update_edge_counts(int current_block, int proposed_block, MapVector<int> current_row,
                                    MapVector<int> proposed_row, MapVector<int> current_col,
                                    MapVector<int> proposed_col) = 0;
    /// Updates the blockmatrix values using the changes to the blockmodel stored in `delta`.
    virtual void update_edge_counts(const Delta &delta) = 0;
    /// Returns true if the value in matrix[`row`, `col`] == `val`.
    virtual bool validate(int row, int col, int val) const = 0;
    virtual std::vector<int> values() const = 0;
    std::pair<int, int> shape;

protected:
    void check_row_bounds(int row) const {
        if (row < 0 || row >= this->nrows) {
            throw IndexOutOfBoundsException(row, this->nrows);
        }
    }
    void check_col_bounds(int col) const {
        if (col < 0 || col >= this->ncols) {
            throw IndexOutOfBoundsException(col, this->ncols);
        }
    }
    int ncols;
    int nrows;
};

///
/// C++ implementation of the distributed sparse matrix interface
///
class IDistSparseMatrix : public ISparseMatrix {
public:
    // IDistSparseMatrix() {}
    virtual ~IDistSparseMatrix() {}
    /// Returns true if this process owns this block.
    virtual bool stores(int block) const = 0;
    /// Returns a copy of this distributed matrix.
    virtual IDistSparseMatrix* copyDistSparseMatrix() const = 0;

protected:
    std::vector<int> _ownership;
    virtual void sync_ownership(const std::vector<int> &myblocks) = 0;

};

// typedef std::unique_ptr<ISparseMatrix> SparseMatrix;

#endif // CPPSBP_PARTITION_SPARSE_CSPARSE_MATRIX_HPP
