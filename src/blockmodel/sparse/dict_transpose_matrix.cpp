#include "dict_transpose_matrix.hpp"

void DictTransposeMatrix::add(int row, int col, int val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    this->matrix[row][col] += val;
//    this->matrix_transpose[col][row] += val;
}

void DictTransposeMatrix::add_transpose(int row, int col, int val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    this->matrix_transpose[col][row] += val;
}

// void DictTransposeMatrix::check_row_bounds(int row) {
//     if (row < 0 || row >= this->nrows) {
//         throw IndexOutOfBoundsException(row, this->nrows);
//     }
// }

// void DictTransposeMatrix::check_col_bounds(int col) {
//     if (col < 0 || col >= this->ncols) {
//         throw IndexOutOfBoundsException(col, this->ncols);
//     }
// }

// void DictTransposeMatrix::check_row_bounds(int row) const {
//     if (row < 0 || row >= this->nrows) {
//         throw IndexOutOfBoundsException(row, this->nrows);
//     }
// }

// void DictTransposeMatrix::check_col_bounds(int col) const {
//     if (col < 0 || col >= this->ncols) {
//         throw IndexOutOfBoundsException(col, this->ncols);
//     }
// }

void DictTransposeMatrix::clearcol(int col) {
    this->matrix_transpose[col].clear();
    for (MapVector<int> &row : this->matrix) {
        row.erase(col);
    }
}

void DictTransposeMatrix::clearrow(int row) {
    this->matrix[row].clear();
    for (MapVector<int> &col : this->matrix_transpose) {
        col.erase(row);
    }
}

ISparseMatrix* DictTransposeMatrix::copy() const {
    // std::vector<std::unordered_map<int, int>> dict_matrix(this->nrows, std::unordered_map<int, int>());
    DictTransposeMatrix dict_matrix(this->nrows, this->ncols);
    for (int i = 0; i < this->nrows; ++i) {
        const MapVector<int> row = this->matrix[i];
        dict_matrix.matrix[i] = row;  // TODO: double-check that this is a copy constructor
    }
    for (int i = 0; i < this->ncols; ++i) {
        const MapVector<int> col = this->matrix_transpose[i];
        dict_matrix.matrix_transpose[i] = col;
    }
    return new DictTransposeMatrix(dict_matrix);
}

std::vector<std::tuple<int, int, int>> DictTransposeMatrix::entries() const {
    throw std::logic_error("entries() is not implemented for DictTransposeMatrix!");
}

int DictTransposeMatrix::get(int row, int col) const {
    check_row_bounds(row);
    check_col_bounds(col);
    const MapVector<int> &row_vector = this->matrix[row];
    auto it = row_vector.find(col);
    if (it == row_vector.end())
        return 0;
    return it->second;
    // return this->matrix[row][col];
}

std::vector<int> DictTransposeMatrix::getcol(int col) const {
    check_col_bounds(col);
    std::vector<int> col_values(this->nrows, 0);
    const MapVector<int> &matrix_col = this->matrix_transpose[col];
    for (const std::pair<int, int> element : matrix_col) {
        col_values[element.first] = element.second;
    }
    return col_values;
}

MapVector<int> DictTransposeMatrix::getcol_sparse(int col) const {
    check_col_bounds(col);
    return this->matrix_transpose[col];
}

// const MapVector<int>& DictTransposeMatrix::getcol_sparse(int col) const {
//     check_col_bounds(col);
//     return this->matrix_transpose[col];
// }

void DictTransposeMatrix::getcol_sparse(int col, MapVector<int> &col_vector) const {
    check_col_bounds(col);
    col_vector = this->matrix_transpose[col];
}

std::vector<int> DictTransposeMatrix::getrow(int row) const {
    check_row_bounds(row);
    std::vector<int> row_values = utils::constant<int>(this->ncols, 0);
    // int row_values [this->ncols];
    // NOTE: could save some time by pulling this->matrix[row] out, and then iterating over it using references
    // but, this could be not thread safe
    // for (int row_index = 0; row_index < this->nrows; ++row_index) {
    const MapVector<int> &matrix_row = this->matrix[row];
    for (const std::pair<int, int> element : matrix_row) {
        row_values[element.first] = element.second;
    }
    return row_values;
}

MapVector<int> DictTransposeMatrix::getrow_sparse(int row) const {
    check_row_bounds(row);
    return this->matrix[row];
}

// const MapVector<int>& DictTransposeMatrix::getrow_sparse(int row) const {
//     check_row_bounds(row);
//     return this->matrix[row];
// }

void DictTransposeMatrix::getrow_sparse(int row, MapVector<int> &row_vector) const {
    check_row_bounds(row);
    row_vector = this->matrix_transpose[row];
}

EdgeWeights DictTransposeMatrix::incoming_edges(int block) const {
    check_col_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    const MapVector<int> &block_col = this->matrix_transpose[block];
    for (const std::pair<int, int> &element : block_col) {
        indices.push_back(element.first);
        values.push_back(element.second);
    }
    return EdgeWeights {indices, values};
}

std::set<int> DictTransposeMatrix::neighbors(int block) const {
    std::set<int> result;
    for (const std::pair<const int, int> &entry : this->matrix[block]) {
        result.insert(entry.first);
    }
    for (const std::pair<const int, int> &entry : this->matrix_transpose[block]) {
        result.insert(entry.first);
    }
    return result;
}

Indices DictTransposeMatrix::nonzero() const {
    std::vector<int> row_vector;
    std::vector<int> col_vector;
    for (int row = 0; row < nrows; ++row) {
        const MapVector<int> matrix_row = this->matrix[row];
        for (const std::pair<const int, int> &element : matrix_row) {
            row_vector.push_back(row);
            col_vector.push_back(element.first);
        }
    }
    return Indices{row_vector, col_vector};
}

void DictTransposeMatrix::setcol(int col, const MapVector<int> &vector) {
    this->matrix_transpose[col] = MapVector<int>(vector);
    for (int row = 0; row < (int) this->matrix.size(); ++row) {
        MapVector<int>::const_iterator value = vector.find(row);
        if (value == vector.end())  // value is not in vector
            this->matrix[row].erase(col);
        else
            this->matrix[row][col] = value->second;
    }
}

void DictTransposeMatrix::setrow(int row, const MapVector<int> &vector) {
    this->matrix[row] = MapVector<int>(vector);
    for (int col = 0; col < (int) this->matrix.size(); ++col) {
        MapVector<int>::const_iterator value = vector.find(col);
        if (value == vector.end())  // value is not in vector
            this->matrix_transpose[col].erase(row);
        else
            this->matrix_transpose[col][row] = value->second;
    }
}

void DictTransposeMatrix::sub(int row, int col, int val) {
    check_row_bounds(row);
    check_col_bounds(col);
    // TODO: debug mode - if matrix[row][col] doesn't exist, throw exception
    this->matrix[row][col] -= val;
    this->matrix_transpose[col][row] -= val;
}

int DictTransposeMatrix::edges() const {
    int total = 0;
    for (int row = 0; row < nrows; ++row) {
        const MapVector<int> &matrix_row = this->matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            total += element.second;
        }
    }
    return total;
}

void DictTransposeMatrix::print() const {
    std::cout << "Matrix: " << std::endl;
    for (int row = 0; row < this->nrows; ++row) {
        for (int col = 0; col < this->ncols; ++col) {
            std::cout << map_vector::get(this->matrix[row], col) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Transpose: " << std::endl;
    for (int col = 0; col < this->ncols; ++col) {
        for (int row = 0; row < this->nrows; ++row) {
            std::cout << map_vector::get(this->matrix_transpose[col], row) << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<int> DictTransposeMatrix::sum(int axis) const {
    if (axis < 0 || axis > 1) {
        throw IndexOutOfBoundsException(axis, 2);
    }
    if (axis == 0) {  // sum across columns
        std::vector<int> totals(this->ncols, 0);
        for (int row_index = 0; row_index < this->nrows; ++row_index) {
            const MapVector<int> &row = this->matrix[row_index];
            for (const std::pair<int, int> &element : row) {
                totals[element.first] += totals[element.second];
            }
        }
        return totals;  // py::array_t<int>(this->ncols, totals);
    } else {  // (axis == 1) sum across rows
        std::vector<int> totals(this->nrows, 0);
        for (int row = 0; row < this->nrows; ++row) {
            const MapVector<int> &matrix_row = this->matrix[row];
            for (const std::pair<int, int> &element : matrix_row) {
                totals[row] += element.second;
            }
        }
        return totals;
    }
}

EdgeWeights DictTransposeMatrix::outgoing_edges(int block) const {
    check_row_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    const MapVector<int> &block_row = this->matrix[block];
    for (const std::pair<int, int> &element : block_row) {
        indices.push_back(element.first);
        values.push_back(element.second);
    }
    return EdgeWeights {indices, values};
}

int DictTransposeMatrix::trace() const {
    int total = 0;
    // Assumes that the matrix is square (which it should be in this case)
    for (int index = 0; index < this->nrows; ++index) {
        total += this->get(index, index);
        // total += this->matrix[index][index];
    }
    return total;
}

void DictTransposeMatrix::update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
                                             std::vector<int> proposed_row, std::vector<int> current_col,
                                             std::vector<int> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    for (int col = 0; col < ncols; ++col) {
        int current_val = current_row[col];
        if (current_val == 0) {
            this->matrix[current_block].erase(col);
            this->matrix_transpose[col].erase(current_block);
        } else {
            this->matrix[current_block][col] = current_val;
            this->matrix_transpose[col][current_block] = current_val;
        }
        int proposed_val = proposed_row[col];
        if (proposed_val == 0) {
            this->matrix[proposed_block].erase(col);
            this->matrix_transpose[col].erase(proposed_block);
        } else {
            this->matrix[proposed_block][col] = proposed_val;
            this->matrix_transpose[col][proposed_block] = proposed_val;  
        }
    }
    for (int row = 0; row < nrows; ++row) {
        int current_val = current_col[row];
        if (current_val == 0) {
            this->matrix[row].erase(current_block);
            this->matrix_transpose[current_block].erase(row);
        } else {
            this->matrix[row][current_block] = current_val;
            this->matrix_transpose[current_block][row] = current_val;
        }
        int proposed_val = proposed_col[row];
        if (proposed_val == 0) {
            this->matrix[row].erase(proposed_block);
            this->matrix_transpose[proposed_block].erase(row);
        } else {
            this->matrix[row][proposed_block] = proposed_val;
            this->matrix_transpose[proposed_block][row] = proposed_val;
        }
    }
}

void DictTransposeMatrix::update_edge_counts(int current_block, int proposed_block, MapVector<int> current_row,
                                             MapVector<int> proposed_row, MapVector<int> current_col,
                                             MapVector<int> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    for (int col = 0; col < ncols; ++col) {
        int current_val = current_row[col];
        if (current_val == 0) {
            this->matrix[current_block].erase(col);
            this->matrix_transpose[col].erase(current_block);
        } else {
            this->matrix[current_block][col] = current_val;
            this->matrix_transpose[col][current_block] = current_val;
        }
        int proposed_val = proposed_row[col];
        if (proposed_val == 0) {
            this->matrix[proposed_block].erase(col);
            this->matrix_transpose[col].erase(proposed_block);
        } else {
            this->matrix[proposed_block][col] = proposed_val;
            this->matrix_transpose[col][proposed_block] = proposed_val;
        }
    }
    for (int row = 0; row < nrows; ++row) {
        int current_val = current_col[row];
        if (current_val == 0) {
            this->matrix[row].erase(current_block);
            this->matrix_transpose[current_block].erase(row);
        } else {
            this->matrix[row][current_block] = current_val;
            this->matrix_transpose[current_block][row] = current_val;
        }
        int proposed_val = proposed_col[row];
        if (proposed_val == 0) {
            this->matrix[row].erase(proposed_block);
            this->matrix_transpose[proposed_block].erase(row);
        } else {
            this->matrix[row][proposed_block] = proposed_val;
            this->matrix_transpose[proposed_block][row] = proposed_val;
        }
    }
//    this->matrix[current_block] = MapVector<int>(current_row);
//    this->matrix[proposed_block] = MapVector<int>(proposed_row);
//    this->matrix_transpose[current_block] = MapVector<int>(current_col);
//    this->matrix_transpose[proposed_block] = MapVector<int>(proposed_col);
//    for (int block = 0; block < nrows; ++block) {
//        // TODO: try using get function (retrieve without modifying)
//        int current_val = current_col[block];  // matrix(block, current_block)
//        if (current_val == 0)
//            this->matrix[block].erase(current_block);
//        else
//            this->matrix[block][current_block] = current_val;
//        int proposed_val = proposed_col[block];  // matrix(block, proposed_block)
//        if (proposed_val == 0)
//            this->matrix[block].erase(proposed_block);
//        else
//            this->matrix[block][proposed_block] = proposed_val;
//        current_val = current_row[block];  // matrix(current_block, block)
//        if (current_val == 0)
//            this->matrix_transpose[block].erase(current_block);
//        else
//            this->matrix_transpose[block][current_block] = current_val;
//        proposed_val = proposed_row[block];  // matrix(proposed_block, block)
//        if (proposed_val == 0)
//            this->matrix_transpose[block].erase(proposed_block);
//        else
//            this->matrix_transpose[block][proposed_block] = current_val;
//    }

}

void DictTransposeMatrix::update_edge_counts(const Delta &delta) {
//    for (const std::pair<const std::pair<int, int>, int> &entry : delta) {
//        int row = entry.first.first;
//        int col = entry.first.second;
//        int change = entry.second;
    for (const std::tuple<int, int, int> &entry : delta.entries()) {
        int row = std::get<0>(entry);
        int col = std::get<1>(entry);
        int change = std::get<2>(entry);
        this->matrix[row][col] += change;
        this->matrix_transpose[col][row] += change;
        if (this->matrix[row][col] == 0) {
            this->matrix[row].erase(col);
            this->matrix_transpose[col].erase(row);
        }
    }
}

bool DictTransposeMatrix::validate(int row, int col, int val) const {
    int matrix_value = this->getrow_sparse(row)[col];
    int transpose_value = this->getcol_sparse(col)[row];
    if (val != matrix_value || val != transpose_value) {
        std::cout << "matrix[" << row << "," << col << "] = " << matrix_value << ", matrixT[" << col << "," << row
        << "] = " << transpose_value << " while actual value = " << val << std::endl;
        return false;
    }
    return true;
}

std::vector<int> DictTransposeMatrix::values() const {
    // TODO: maybe return a sparse vector every time?
    std::vector<int> values;
    for (int row = 0; row < nrows; ++row) {
        const MapVector<int> &matrix_row = this->matrix[row];
        for (const std::pair<const int, int> &element : matrix_row) {
            values.push_back(element.second);
        }
    }
    return values;
}
