#include "dist_dict_matrix.hpp"

void DistDictMatrix::add(int row, int col, int val) {
    // TODO: bound checking includes branching, may get rid of it for performance
    check_row_bounds(row);
    check_col_bounds(col);
    this->_matrix[row][col] += val;
}

void DistDictMatrix::clearcol(int col) {
    for (MapVector<int> &row : this->_matrix)
        row.erase(col);
}

void DistDictMatrix::clearrow(int row) {
    this->_matrix[row].clear();
}

ISparseMatrix* DistDictMatrix::copy() const {
    return this->copyDistSparseMatrix();
}

IDistSparseMatrix* DistDictMatrix::copyDistSparseMatrix() const {
    DistDictMatrix *matrix = new DistDictMatrix();
    matrix->nrows = this->nrows;
    matrix->ncols = this->ncols;
    for (int i = 0; i < this->nrows; ++i) {
        const std::unordered_map<int, int> row = this->_matrix[i];
        matrix->_matrix[i] = row;  // TODO: double-check that this is a copy constructor
    }
    matrix->_ownership = std::vector<int>(this->_ownership);
    return matrix;
}

int DistDictMatrix::get(int row, int col) const {
    check_row_bounds(row);
    check_col_bounds(col);
    const MapVector<int> &row_vector = this->_matrix[row];
    auto it = row_vector.find(col);
    if (it == row_vector.end())
        return 0;
    return it->second;
}

std::vector<int> DistDictMatrix::getcol(int col) const {
    check_col_bounds(col);
    std::vector<int> col_values(this->nrows, 0);
    for (int row = 0; row < this->nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->_matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            if (element.first == col) {
                col_values[row] = element.second;
                break;
            }
        }
    }
    return col_values;
}

MapVector<int> DistDictMatrix::getcol_sparse(int col) const {
    check_col_bounds(col);
    MapVector<int> col_vector;
    for (int row = 0; row < this->nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->_matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            if (element.first == col) {
                col_vector[row] = element.second;
                break;
            }
        }
    }
    return col_vector;
}

void DistDictMatrix::getcol_sparse(int col, MapVector<int> &col_vector) const {
    check_col_bounds(col);
    for (int row = 0; row < this->nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->_matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            if (element.first == col) {
                col_vector[row] = element.second;
                break;
            }
        }
    }
}

std::vector<int> DistDictMatrix::getrow(int row) const {
    throw "Dense getrow used!";
    check_row_bounds(row);
    if (this->stores(row)) {
        std::vector<int> row_values = utils::constant<int>(this->ncols, 0);
        const MapVector<int> &matrix_row = this->_matrix[row];
        for (const std::pair<int, int> element : matrix_row) {
            row_values[element.first] = element.second;
        }
        return row_values;  // py::array_t<int>(this->ncols, row_values);
    }
    // send message asking for row
    MPI_Send(&row, 1, MPI_INT, this->_ownership[row], MSG_GETROW + omp_get_thread_num(), MPI_COMM_WORLD);
    int rowsize = -1;
    MPI_Status status;
    MPI_Recv(&rowsize, 1, MPI_INT, this->_ownership[row], MSG_SIZEROW + omp_get_thread_num(), MPI_COMM_WORLD, &status);
    // Other side
    MPI_Status status2;
    int requested;
    MPI_Recv(&requested, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status2);
    int threadID = status2.MPI_TAG % 100000;
    int tag = status2.MPI_TAG - threadID;
    if (tag == MSG_GETROW) {
        auto row = this->getrow(requested);
        // MPI_Send(&(row.data()), )
    }
}

MapVector<int> DistDictMatrix::getrow_sparse(int row) const {
    throw "Wrong sparse getrow used!";
    check_row_bounds(row);
    return this->_matrix[row];
}

void DistDictMatrix::getrow_sparse(int row, MapVector<int> &row_vector) const {
    check_row_bounds(row);
    if (this->stores(row)) {
        row_vector = this->_matrix[row];
        return;
    }
    // send message asking for row
    MPI_Send(&row, 1, MPI_INT, this->_ownership[row], MSG_GETROW + omp_get_thread_num(), MPI_COMM_WORLD);
    int rowsize = -1;
    MPI_Status status;
    MPI_Recv(&rowsize, 1, MPI_INT, this->_ownership[row], MSG_SIZEROW + omp_get_thread_num(), MPI_COMM_WORLD, &status);
    // Other side
    MPI_Status status2;
    int requested;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status2);
    // Change count type depending on status
    MPI_Get_count(&status, MPI_INT, &requested);
    MPI_Recv(&requested, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status2);
    int threadID = status2.MPI_TAG % 100000;
    int tag = status2.MPI_TAG - threadID;
    if (tag == MSG_GETROW) {
        MapVector<int> reqrow;
        std::vector<int> sendrow;
        this->getrow_sparse(requested, reqrow);
        for (auto p : reqrow) {
            sendrow.push_back(p.first);
            sendrow.push_back(p.second);
        }
        MPI_Send(sendrow.data(), sendrow.size(), MPI_INT, status2.MPI_SOURCE, MSG_SENDROW, MPI_COMM_WORLD);
    }
}

EdgeWeights DistDictMatrix::incoming_edges(int block) const {
    check_col_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    for (int row = 0; row < this->nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->_matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            if (element.first == block) {
                indices.push_back(row);
                values.push_back(element.second);
                break;
            }
        }
    }
    return EdgeWeights {indices, values};
}

Indices DistDictMatrix::nonzero() const {
    std::vector<int> row_vector;
    std::vector<int> col_vector;
    for (int row = 0; row < nrows; ++row) {
        std::unordered_map<int, int> matrix_row = this->_matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            row_vector.push_back(row);
            col_vector.push_back(element.first);
        }
    }
    return Indices{row_vector, col_vector};
}

EdgeWeights DistDictMatrix::outgoing_edges(int block) const {
    check_row_bounds(block);
    std::vector<int> indices;
    std::vector<int> values;
    const std::unordered_map<int, int> &block_row = this->_matrix[block];
    for (const std::pair<int, int> &element : block_row) {
        indices.push_back(element.first);
        values.push_back(element.second);
    }
    return EdgeWeights {indices, values};
}

bool DistDictMatrix::stores(int block) const {
    return this->_ownership[block] == mpi.rank;
}

void DistDictMatrix::setrow(int row, const MapVector<int> &vector) {
    check_row_bounds(row);
    this->_matrix[row] = MapVector<int>(vector);
}

void DistDictMatrix::setcol(int col, const MapVector<int> &vector) {
    check_col_bounds(col);
    for (int row = 0; row < (int) this->_matrix.size(); ++row) {
        MapVector<int>::const_iterator value = vector.find(row);
        if (value == vector.end())  // value is not in vector
            this->_matrix[row].erase(col);
        else
            this->_matrix[row][col] = value->second;
    }
}

void DistDictMatrix::sub(int row, int col, int val) {
    check_row_bounds(row);
    check_col_bounds(col);
    // TODO: debug mode - if matrix[row][col] doesn't exist, throw exception
    _matrix[row][col] -= val;
}

int DistDictMatrix::edges() const {
    int total = 0;
    for (int row = 0; row < nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->_matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            total += element.second;
        }
    }
    return total;
}

std::vector<int> DistDictMatrix::sum(int axis) const {
    if (axis < 0 || axis > 1) {
        throw IndexOutOfBoundsException(axis, 2);
    }
    if (axis == 0) {  // sum across columns
        std::vector<int> totals(this->ncols, 0);
        for (int row_index = 0; row_index < this->nrows; ++row_index) {
            const std::unordered_map<int, int> &row = this->_matrix[row_index];
            for (const std::pair<int, int> &element : row) {
                totals[element.first] += totals[element.second];
            }
        }
        return totals;
    } else {  // (axis == 1) sum across rows
        std::vector<int> totals(this->nrows, 0);
        for (int row = 0; row < this->nrows; ++row) {
            const std::unordered_map<int, int> &matrix_row = this->_matrix[row];
            for (const std::pair<int, int> &element : matrix_row) {
                totals[row] += element.second;
            }
        }
        return totals;
    }
}


void DistDictMatrix::sync_ownership(const std::vector<int> &myblocks) {
    int numblocks[mpi.num_processes];
    int num_blocks = myblocks.size();
    MPI_Allgather(&(num_blocks), 1, MPI_INT, &numblocks, 1, MPI_INT, MPI_COMM_WORLD);
    int offsets[mpi.num_processes];
    offsets[0] = 0;
    for (int i = 1; i < mpi.num_processes; ++i) {
        offsets[i] = offsets[i-1] + numblocks[i-1];
    }
    int global_num_blocks = offsets[mpi.num_processes-1] + numblocks[mpi.num_processes-1];
    this->_ownership = std::vector<int>(global_num_blocks, -1);
    std::cout << "rank: " << mpi.rank << " num_blocks: " << num_blocks << " and globally: " << global_num_blocks << std::endl;
    std::vector<int> allblocks(global_num_blocks, -1);
    MPI_Allgatherv(myblocks.data(), num_blocks, MPI_INT, allblocks.data(), &(numblocks[0]), &(offsets[0]), MPI_INT, MPI_COMM_WORLD);
    if (mpi.rank == 0) {
        utils::print<int>(allblocks);
    }
    int owner = 0;
    for (int i = 0; i < global_num_blocks; ++i) {
        if (owner < mpi.num_processes - 1 && i >= offsets[owner+1]) {
            owner++;
        }
        this->_ownership[allblocks[i]] = owner;
    }
    if (mpi.rank == 0) {
        utils::print<int>(this->_ownership);
    }
}

int DistDictMatrix::trace() const {
    int total = 0;
    // Assumes that the matrix is square (which it should be in this case)
    for (int index = 0; index < this->nrows; ++index) {
        // TODO: this creates 0 elements where they don't exist. To optimize memory, could add a find call first
        total += this->get(index, index);
    }
    return total;
}

void DistDictMatrix::update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
    std::vector<int> proposed_row, std::vector<int> current_col, std::vector<int> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    for (int col = 0; col < ncols; ++col) {
        int current_val = current_row[col];
        if (current_val == 0)
            this->_matrix[current_block].erase(col);
        else
            this->_matrix[current_block][col] = current_val;
        int proposed_val = proposed_row[col];
        if (proposed_val == 0)
            this->_matrix[proposed_block].erase(col);
        else
            this->_matrix[proposed_block][col] = proposed_val;
    }
    for (int row = 0; row < nrows; ++row) {
        int current_val = current_col[row];
        if (current_val == 0)
            this->_matrix[row].erase(current_block);
        else
            this->_matrix[row][current_block] = current_val;
        int proposed_val = proposed_col[row];
        if (proposed_val == 0)
            this->_matrix[row].erase(proposed_block);
        else
            this->_matrix[row][proposed_block] = proposed_val;
    }
}

void DistDictMatrix::update_edge_counts(const PairIndexVector &delta) {
    for (const std::pair<const std::pair<int, int>, int> &entry : delta) {
        int row = entry.first.first;
        int col = entry.first.second;
        int change = entry.second;
        this->_matrix[row][col] += change;
        if (this->_matrix[row][col] == 0)
            this->_matrix[row].erase(col);
    }
}

std::vector<int> DistDictMatrix::values() const {
    // TODO: maybe return a sparse vector every time?
    std::vector<int> values;
    for (int row = 0; row < nrows; ++row) {
        const std::unordered_map<int, int> &matrix_row = this->_matrix[row];
        for (const std::pair<int, int> &element : matrix_row) {
            values.push_back(element.second);
        }
    }
    return values;
}
