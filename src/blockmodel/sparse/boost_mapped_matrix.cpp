#include "boost_mapped_matrix.hpp"

void BoostMappedMatrix::add(int row, int col, int val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    matrix(row, col) += val;
}

void BoostMappedMatrix::check_row_bounds(int row) {
    if (row < 0 || row >= this->nrows) {
        throw IndexOutOfBoundsException(row, this->nrows);
    }
}

void BoostMappedMatrix::check_col_bounds(int col) {
    if (col < 0 || col >= this->ncols) {
        throw IndexOutOfBoundsException(col, this->ncols);
    }
}

BoostMappedMatrix BoostMappedMatrix::copy() {
    BoostMappedMatrix boost_mapped_matrix(this->nrows, this->ncols);
    boost_mapped_matrix.matrix = boost::numeric::ublas::mapped_matrix<int>(this->matrix);
    return boost_mapped_matrix;
}

int BoostMappedMatrix::get(int row, int col) {
    check_row_bounds(row);
    check_col_bounds(col);
    return matrix(row, col);
}

std::vector<int> BoostMappedMatrix::getrow(int row) {
    check_row_bounds(row);
    std::vector<int> row_values = utils::constant<int>(this->ncols, 0);
    // int row_values [this->ncols];
    for (int col = 0; col < ncols; ++col) {
        row_values[col] = matrix(row, col);
    }
    return row_values;  // py::array_t<int>(this->ncols, row_values);
}

std::vector<int> BoostMappedMatrix::getcol(int col) {
    // check_col_bounds(col);
    // std::vector<int> col_values = utils::constant<int>(this->nrows, 0);
    std::vector<int> col_values(this->nrows, 0);
    // int col_values [this->nrows];
    for (int row = 0; row < nrows; ++row) {
        col_values[row] = matrix(row, col);
    }
    return col_values;  // py::array_t<int>(this->nrows, col_values);
}

EdgeWeights BoostMappedMatrix::incoming_edges(int block) {
    check_col_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    for (int row = 0; row < this->nrows; ++row) {
        int value = this->matrix(row, block);
        if (value != 0) {
            indices.push_back(row);
            values.push_back(value);
        } else {
            this->matrix.erase_element(row, block);
        }
    }
    return EdgeWeights {indices, values};
}

Indices BoostMappedMatrix::nonzero() {
    std::vector<int> row_vector;
    std::vector<int> col_vector;
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            if (matrix(row, col) != 0) {
                row_vector.push_back(row);
                col_vector.push_back(col);
            }
        }
    }
    return Indices{row_vector, col_vector};
}

void BoostMappedMatrix::sub(int row, int col, int val) {
    check_row_bounds(row);
    check_col_bounds(col);
    matrix(row, col) -= val;
}

int BoostMappedMatrix::sum() {
    int total = 0;
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            total += matrix(row, col);
        }
    }
    return total;
}

std::vector<int> BoostMappedMatrix::sum(int axis) {
    if (axis < 0 || axis > 1) {
        throw IndexOutOfBoundsException(axis, 2);
    }
    if (axis == 0) {  // sum across columns
        std::vector<int> totals = utils::constant<int>(this->ncols, 0);
        for (int row = 0; row < this->nrows; ++row) {
            for (int col = 0; col < this->ncols; ++col) {
                totals[col] += this->matrix(row, col);
            }
        }
        return totals;  // py::array_t<int>(this->ncols, totals);
    } else {  // (axis == 1) sum across rows
        std::vector<int> totals = utils::constant<int>(this->nrows, 0);
        for (int row = 0; row < this->nrows; ++row) {
            for (int col = 0; col < this->ncols; ++col) {
                totals[row] += this->matrix(row, col);
            }
        }
        return totals;
    }
}

int BoostMappedMatrix::trace() {
    int total = 0;
    // Assumes that the matrix is square (which it should be in this case)
    for (int index = 0; index < this->nrows; ++index) {
        total += this->matrix(index, index);
    }
    return total;
}

// int BoostMappedMatrix::operator[] (py::tuple index) {
//     py::array_t<int> tuple_array(index);
//     auto tuple_vals = tuple_array.mutable_unchecked<1>();
//     int row = tuple_vals(0);
//     int col = tuple_vals(1);
//     return this->matrix(row, col);
// }

EdgeWeights BoostMappedMatrix::outgoing_edges(int block) {
    check_row_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    for (int col = 0; col < this->ncols; ++col) {
        int value = this->matrix(block, col);
        if (value != 0) {
            indices.push_back(col);
            values.push_back(value);
        } else {
            this->matrix.erase_element(block, col);
        }
    }
    return EdgeWeights {indices, values};
}

void BoostMappedMatrix::update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
    std::vector<int> proposed_row, std::vector<int> current_col, std::vector<int> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    for (int col = 0; col < ncols; ++col) {
        int current_val = current_row[col];
        if (current_val == 0)
            this->matrix.erase_element(current_block, col);
        else
            this->matrix(current_block, col) = current_val;
        int proposed_val = proposed_row[col];
        if (proposed_val == 0)
            this->matrix.erase_element(proposed_block, col);
        else
            this->matrix(proposed_block, col) = proposed_val;
    }
    for (int row = 0; row < nrows; ++row) {
        int current_val = current_col[row];
        if (current_val == 0)
            this->matrix.erase_element(row, current_block);
        else
            this->matrix(row, current_block) = current_val;
        int proposed_val = proposed_col[row];
        if (proposed_val == 0)
            this->matrix.erase_element(row, proposed_block);
        else
            this->matrix(row, proposed_block) = proposed_val;
    }
}

std::vector<int> BoostMappedMatrix::values() {
    // TODO: maybe return a sparse vector every time?
    std::vector<int> values;
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            int value = matrix(row, col);
            if (value != 0) {
                values.push_back(value);
            }
        }
    }
    return values;
}
