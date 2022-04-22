/***
 * Sparse Matrix that uses a vector of unordered maps to store the blockmodel.
 */
#ifndef CPPSBP_PARTITION_SPARSE_DIST_DICT_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_DIST_DICT_MATRIX_HPP

#include <mpi.h>
#include <unordered_map>

#include "csparse_matrix.hpp"
#include "mpi_data.hpp"
// TODO: figure out where to put utils.hpp so this never happens
#include "../../utils.hpp"
#include "typedefs.hpp"

// #include <Eigen/Core>

/**
 * C++ interface of the dictionary (map of maps) sparse matrix
 */
class DistDictMatrix : public IDistSparseMatrix {
  public:
    DistDictMatrix() {}
    // DistDictMatrix(int nrows, int ncols, const MPI &mpi, const std::vector<int> &myblocks) {
    DistDictMatrix(int nrows, int ncols, const std::vector<int> &myblocks) {
        this->ncols = ncols;
        this->nrows = nrows;
        // this->matrix = boost::numeric::ublas::coordinate_matrix<int>(this->nrows, this->ncols);
        this->_matrix = std::vector<std::unordered_map<int, int>>(this->nrows, std::unordered_map<int, int>());
        // this->matrix = boost::numeric::ublas::mapped_matrix<int>(this->nrows, this->ncols);
        // int shape_array[2] = {this->nrows, this->ncols};
        this->shape = std::make_pair(this->nrows, this->ncols);
        // this->sync_ownership(myblocks, mpi);
        this->sync_ownership(myblocks);
    }
    virtual void add(int row, int col, int val) override;
    // virtual void add(int row, std::vector<int> cols, std::vector<int> values) override;
    virtual void clearcol(int col) override;
    virtual void clearrow(int row) override;
    virtual ISparseMatrix* copy() const override;
    virtual IDistSparseMatrix* copyDistSparseMatrix() const override;
    virtual int get(int row, int col) const override;
    virtual std::vector<int> getcol(int col) const override;
    virtual MapVector<int> getcol_sparse(int col) const override;
    virtual void getcol_sparse(int col, MapVector<int> &col_vector) const override;
    // virtual MapVector<int> getcol_sparse(int col) override;
    // virtual const MapVector<int>& getcol_sparse(int col) const override;
    virtual std::vector<int> getrow(int row) const override;
    virtual MapVector<int> getrow_sparse(int row) const override;
    virtual void getrow_sparse(int row, MapVector<int> &row_vector) const override;
    // virtual MapVector<int> getrow_sparse(int row) override;
    // virtual const MapVector<int>& getrow_sparse(int row) const override;
    virtual EdgeWeights incoming_edges(int block) const override;
    virtual Indices nonzero() const override;
    virtual EdgeWeights outgoing_edges(int block) const override;
    // Returns True if this rank owns this block.
    virtual bool stores(int block) const override;
    /// Sets the values in a row equal to the input vector
    virtual void setrow(int row, const MapVector<int> &vector) override;
    /// Sets the values in a column equal to the input vector
    virtual void setcol(int col, const MapVector<int> &vector) override;
    virtual void sub(int row, int col, int val) override;
    virtual int edges() const override;
    virtual std::vector<int> sum(int axis = 0) const override;
    virtual int trace() const override;
    virtual void update_edge_counts(int current_block, int proposed_block, std::vector<int> current_row,
                                    std::vector<int> proposed_row, std::vector<int> current_col,
                                    std::vector<int> proposed_col) override;
    void update_edge_counts(const PairIndexVector &delta) override;
    virtual std::vector<int> values() const override;

  private:
    std::vector<std::unordered_map<int, int>> _matrix;
    // std::vector<int> _ownership;
    /// Syncs the ownership between all MPI processes.
    // void sync_ownership(const std::vector<int> &myblocks, const MPI &mpi) {
    virtual void sync_ownership(const std::vector<int> &myblocks) override;
    // void sync_ownership(const std::vector<int> &myblocks) {
    //     int numblocks[mpi.num_processes];
    //     int num_blocks = myblocks.size();
    //     MPI_Allgather(&(num_blocks), 1, MPI_INT, &numblocks, 1, MPI_INT, MPI_COMM_WORLD);
    //     int offsets[mpi.num_processes];
    //     offsets[0] = 0;
    //     for (int i = 1; i < mpi.num_processes; ++i) {
    //         offsets[i] = offsets[i-1] + numblocks[i-1];
    //     }
    //     int global_num_blocks = offsets[mpi.num_processes-1] + numblocks[mpi.num_processes-1];
    //     this->_ownership = std::vector<int>(global_num_blocks, -1);
    //     std::cout << "rank: " << mpi.rank << " num_blocks: " << num_blocks << " and globally: " << global_num_blocks << std::endl;
    //     std::vector<int> allblocks(global_num_blocks, -1);
    //     MPI_Allgatherv(myblocks.data(), num_blocks, MPI_INT, allblocks.data(), &(numblocks[0]), &(offsets[0]), MPI_INT, MPI_COMM_WORLD);
    //     if (mpi.rank == 0) {
    //         utils::print<int>(allblocks);
    //     }
    //     int owner = 0;
    //     for (int i = 0; i < global_num_blocks; ++i) {
    //         if (owner < mpi.num_processes - 1 && i >= offsets[owner+1]) {
    //             owner++;
    //         }
    //         this->_ownership[allblocks[i]] = owner;
    //     }
    //     if (mpi.rank == 0) {
    //         utils::print<int>(this->_ownership);
    //     }
    // }
};

#endif // CPPSBP_PARTITION_SPARSE_DIST_DICT_MATRIX_HPP
