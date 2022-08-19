/***
 * Stores the current graph blockmodeling results.
 */
#ifndef SBP_BLOCKMODEL_DELTA_HPP
#define SBP_BLOCKMODEL_DELTA_HPP

#include <iostream>
#include <limits>
#include <numeric>
#include <map>
#include <memory>
#include <queue>

// #include <Eigen/Core>
// #include "sparse/boost_mapped_matrix.hpp"
//#include "../args.hpp"
//#include "csparse_matrix.hpp"
//#include "sparse/dict_transpose_matrix.hpp"
#include "typedefs.hpp"
//#include "../utils.hpp"

class Delta {
private:
    MapVector<int> _current_block_row;
    MapVector<int> _proposed_block_row;
    MapVector<int> _current_block_col;
    MapVector<int> _proposed_block_col;
    int _current_block;
    int _proposed_block;
    int _self_edge_weight;

public:
    Delta() {
        this->_current_block = -1;
        this->_proposed_block = -1;
        this->_self_edge_weight = 0;
    }
    Delta(int current_block, int proposed_block, int buckets = 10) {
        this->_current_block = current_block;
        this->_proposed_block = proposed_block;
        this->_self_edge_weight = 0;
        this->_current_block_row = MapVector<int>(buckets);
        this->_proposed_block_row = MapVector<int>(buckets);
        this->_current_block_col = MapVector<int>(buckets);
        this->_proposed_block_col = MapVector<int>(buckets);
    }
    Delta(int current_block, int proposed_block, const MapVector<int> &block_row, const MapVector<int> &block_col,
          const MapVector<int> &proposed_row, const MapVector<int> &proposed_col) {
        this->_current_block = current_block;
        this->_proposed_block = proposed_block;
        this->_self_edge_weight = 0;
        this->zero_init(block_row, block_col, proposed_row, proposed_col);
    }
    /// Adds `value` as the delta to cell matrix[`row`,`col`].
    void add(int row, int col, int value) {
        if (row == this->_current_block)
            this->_current_block_row[col] += value;
        else if (row == this->_proposed_block)
            this->_proposed_block_row[col] += value;
        else if (col == this->_current_block)
            this->_current_block_col[row] += value;
        else if (col == this->_proposed_block)
            this->_proposed_block_col[row] += value;
        else
            throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
    }
    /// Returns all stores deltas as a list of tuples storing `row`, `col`, `delta`.
    std::vector<std::tuple<int, int, int>> entries() const {
        std::vector<std::tuple<int, int, int>> result;
        for (const std::pair<const int, int> &entry : this->_current_block_row) {
            result.emplace_back(this->_current_block, entry.first, entry.second);
        }
        for (const std::pair<const int, int> &entry : this->_proposed_block_row) {
            result.emplace_back(this->_proposed_block, entry.first, entry.second);
        }
        for (const std::pair<const int, int> &entry : this->_current_block_col) {
            result.emplace_back(entry.first, this->_current_block, entry.second);
        }
        for (const std::pair<const int, int> &entry : this->_proposed_block_col) {
            result.emplace_back(entry.first, this->_proposed_block, entry.second);
        }
        return result;
    }
    /// Returns the delta for matrix[`row`,`col`] without modifying the underlying data structure.
    int get(int row, int col) const {
        if (row == this->_current_block)
            return map_vector::get(this->_current_block_row, col);
        else if (row == this->_proposed_block)
            return map_vector::get(this->_proposed_block_row, col);
        else if (col == this->_current_block)
            return map_vector::get(this->_current_block_col, row);
        else if (col == this->_proposed_block)
            return map_vector::get(this->_proposed_block_col, row);
        throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
    }
    /// Returns the weight of the self edge for this move, if any.
    int self_edge_weight() const {
        return this->_self_edge_weight;
    }
    /// Sets the weight of the self edge for this move, if any.
    void self_edge_weight(int weight) {
        this->_self_edge_weight = weight;
    }
    /// Adds -`value` (negative `value`) as the delta to cell matrix[`row`,`col`].
    void sub(int row, int col, int value) {
        if (row == this->_current_block)
            this->_current_block_row[col] -= value;
        else if (row == this->_proposed_block)
            this->_proposed_block_row[col] -= value;
        else if (col == this->_current_block)
            this->_current_block_col[row] -= value;
        else if (col == this->_proposed_block)
            this->_proposed_block_col[row] -= value;
        else
            throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
    }
    /// Initiates the deltas with 0s for all non-zero elements currently present in `block_row`, `block_col`,
    /// `proposed_row`, and `proposed_col`.
    void zero_init(const MapVector<int> &block_row, const MapVector<int> &block_col, const MapVector<int> &proposed_row,
                   const MapVector<int> &proposed_col) {
//    void zero_init(const ISparseMatrix *matrix) {
//        const MapVector<int> &block_row = matrix->getrow_sparse(this->_current_block);
        this->_current_block_row = MapVector<int>(block_row.bucket_count());
        for (const std::pair<const int, int> &entry : block_row) {
            int col = entry.first;
            this->add(this->_current_block, col, 0);
        }
//        const MapVector<int> &block_col = matrix->getcol_sparse(this->_current_block);
        this->_current_block_col = MapVector<int>(block_col.bucket_count());
        for (const std::pair<const int, int> &entry : block_col) {
            int row = entry.first;
            this->add(row, this->_current_block, 0);
        }
//        const MapVector<int> &proposed_row = matrix->getrow_sparse(this->_proposed_block);
        this->_proposed_block_row = MapVector<int>(proposed_row.bucket_count());
        for (const std::pair<const int, int> &entry : proposed_row) {
            int col = entry.first;
            this->add(this->_proposed_block, col, 0);
        }
//        const MapVector<int> &proposed_col = matrix->getcol_sparse(this->_proposed_block);
        this->_proposed_block_col = MapVector<int>(proposed_col.bucket_count());
        for (const std::pair<const int, int> &entry : proposed_col) {
            int row = entry.first;
            this->add(row, this->_proposed_block, 0);
        }
    }
    int current_block() const { return this->_current_block; }
    int proposed_block() const { return this->_proposed_block; }
};

//class PointerDelta {
//private:
//    MapVector<int> _current_block_row;
//    MapVector<int> _proposed_block_row;
//    MapVector<int> _current_block_col;
//    MapVector<int> _proposed_block_col;
//    int _current_block;
//    int _proposed_block;
//    int _self_edge_weight;
//
//public:
//    PointerDelta() {
//        this->_current_block = -1;
//        this->_proposed_block = -1;
//        this->_self_edge_weight = 0;
//    }
//    PointerDelta(int current_block, int proposed_block, int buckets = 10) {
//        this->_current_block = current_block;
//        this->_proposed_block = proposed_block;
//        this->_self_edge_weight = 0;
//        this->_current_block_row = MapVector<int>(buckets);
//        this->_proposed_block_row = MapVector<int>(buckets);
//        this->_current_block_col = MapVector<int>(buckets);
//        this->_proposed_block_col = MapVector<int>(buckets);
//    }
//    PointerDelta(int current_block, int proposed_block, const MapVector<int> &block_row,
//                 const MapVector<int> &block_col, const MapVector<int> &proposed_row,
//                 const MapVector<int> &proposed_col) {
//        this->_current_block = current_block;
//        this->_proposed_block = proposed_block;
//        this->_self_edge_weight = 0;
//        this->zero_init(block_row, block_col, proposed_row, proposed_col);
//    }
//    /// Adds `value` as the delta to cell matrix[`row`,`col`].
//    void add(int row, int col, int value) {
//        if (row == this->_current_block)
//            this->_current_block_row[col] += value;
//        else if (row == this->_proposed_block)
//            this->_proposed_block_row[col] += value;
//        else if (col == this->_current_block)
//            this->_current_block_col[row] += value;
//        else if (col == this->_proposed_block)
//            this->_proposed_block_col[row] += value;
//        else
//            throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
//    }
//    /// Returns all stores deltas as a list of tuples storing `row`, `col`, `delta`.
//    std::vector<std::tuple<int, int, int>> entries() const {
//        std::vector<std::tuple<int, int, int>> result;
//        for (const std::pair<const int, int> &entry : this->_current_block_row) {
//            result.emplace_back(this->_current_block, entry.first, entry.second);
//        }
//        for (const std::pair<const int, int> &entry : this->_proposed_block_row) {
//            result.emplace_back(this->_proposed_block, entry.first, entry.second);
//        }
//        for (const std::pair<const int, int> &entry : this->_current_block_col) {
//            result.emplace_back(entry.first, this->_current_block, entry.second);
//        }
//        for (const std::pair<const int, int> &entry : this->_proposed_block_col) {
//            result.emplace_back(entry.first, this->_proposed_block, entry.second);
//        }
//        return result;
//    }
//    /// Returns the delta for matrix[`row`,`col`] without modifying the underlying data structure.
//    int get(int row, int col) const {
//        if (row == this->_current_block)
//            return map_vector::get(this->_current_block_row, col);
//        else if (row == this->_proposed_block)
//            return map_vector::get(this->_proposed_block_row, col);
//        else if (col == this->_current_block)
//            return map_vector::get(this->_current_block_col, row);
//        else if (col == this->_proposed_block)
//            return map_vector::get(this->_proposed_block_col, row);
//        throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
//    }
//    /// Returns the weight of the self edge for this move, if any.
//    int self_edge_weight() const {
//        return this->_self_edge_weight;
//    }
//    /// Sets the weight of the self edge for this move, if any.
//    void self_edge_weight(int weight) {
//        this->_self_edge_weight = weight;
//    }
//    /// Adds -`value` (negative `value`) as the delta to cell matrix[`row`,`col`].
//    void sub(int row, int col, int value) {
//        if (row == this->_current_block)
//            this->_current_block_row[col] -= value;
//        else if (row == this->_proposed_block)
//            this->_proposed_block_row[col] -= value;
//        else if (col == this->_current_block)
//            this->_current_block_col[row] -= value;
//        else if (col == this->_proposed_block)
//            this->_proposed_block_col[row] -= value;
//        else
//            throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
//    }
//    /// Initiates the deltas with 0s for all non-zero elements currently present in `block_row`, `block_col`,
//    /// `proposed_row`, and `proposed_col`.
//    void zero_init(const MapVector<int> &block_row, const MapVector<int> &block_col, const MapVector<int> &proposed_row,
//                   const MapVector<int> &proposed_col) {
////    void zero_init(const ISparseMatrix *matrix) {
////        const MapVector<int> &block_row = matrix->getrow_sparse(this->_current_block);
//        this->_current_block_row = MapVector<int>(block_row.bucket_count());
//        for (const std::pair<const int, int> &entry : block_row) {
//            int col = entry.first;
//            this->add(this->_current_block, col, 0);
//        }
////        const MapVector<int> &block_col = matrix->getcol_sparse(this->_current_block);
//        this->_current_block_col = MapVector<int>(block_col.bucket_count());
//        for (const std::pair<const int, int> &entry : block_col) {
//            int row = entry.first;
//            this->add(row, this->_current_block, 0);
//        }
////        const MapVector<int> &proposed_row = matrix->getrow_sparse(this->_proposed_block);
//        this->_proposed_block_row = MapVector<int>(proposed_row.bucket_count());
//        for (const std::pair<const int, int> &entry : proposed_row) {
//            int col = entry.first;
//            this->add(this->_proposed_block, col, 0);
//        }
////        const MapVector<int> &proposed_col = matrix->getcol_sparse(this->_proposed_block);
//        this->_proposed_block_col = MapVector<int>(proposed_col.bucket_count());
//        for (const std::pair<const int, int> &entry : proposed_col) {
//            int row = entry.first;
//            this->add(row, this->_proposed_block, 0);
//        }
//    }
//    int current_block() const { return this->_current_block; }
//    int proposed_block() const { return this->_proposed_block; }
//};

#endif // SBP_BLOCKMODEL_DELTA_HPP
