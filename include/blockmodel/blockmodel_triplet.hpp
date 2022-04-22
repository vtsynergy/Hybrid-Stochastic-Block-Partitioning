/**
 * Stores the triplet of blockmodels needed for the fibonacci search.
 */
#ifndef CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
#define CPPSBP_PARTITION_PARTITION_TRIPLET_HPP

#include <iostream>
#include <limits>

#include "blockmodel.hpp"
#include "dist_blockmodel.hpp"

class BlockmodelTriplet {

public:
    /// Creates an empty blockmodel triplet.
    BlockmodelTriplet() : optimal_num_blocks_found(false) {}
    /// True if the optimal number of blocks has been found.
    bool optimal_num_blocks_found;
    /// Updates the blockmodel triplet with the provided blockmodel (inserts it into the correct spot, moves/deletes
    /// the remaining blockmodels as needed).
    void update(Blockmodel &blockmodel);
    /// Prints the number of blocks and overall entropy of every blockmodel in the blockmodel triplet.
    void status();
    /// Returns the blockmodel at index `i`. IndexOutOfBoundError if i >= 3.
    Blockmodel &get(int i) { return blockmodels[i]; }
    /// Returns true if the golden ratio has not been reached, false otherwise. In practice, returns true if
    /// blockmodels[2] == 0.
    bool golden_ratio_not_reached();
    /// Returns true if the optimal number of blocks has been found.
    bool is_done();
    /// Returns the blockmodel on which to perform the next iteration of SBP.
    Blockmodel get_next_blockmodel(Blockmodel &old_blockmodel);

protected:
    /// Blockmodels arranged in order of decreasing number of blocks.
    /// If the first blockmodel is empty, then the golden ratio bracket has not yet been established.
    Blockmodel blockmodels[3];
    /// TODO
    int lower_difference();
    /// TODO
    int upper_difference();
};

class DistBlockmodelTriplet {

public:
    DistBlockmodelTriplet() : optimal_num_blocks_found(false) {}
    /// TODO
    bool optimal_num_blocks_found;
    /// TODO
    TwoHopBlockmodel &get(int i) { return this->blockmodels[i]; }
    /// TODO
    TwoHopBlockmodel get_next_blockmodel(TwoHopBlockmodel &old_blockmodel);
    /// TODO
    bool golden_ratio_not_reached();
    /// TODO
    bool is_done();
    /// TODO
    void update(TwoHopBlockmodel &blockmodel);
    /// TODO
    void status();

private:
    /// Blockmodels arranged in order of decreasing number of blocks.
    /// If the first blockmodel is empty, then the golden ratio bracket has not yet been established.
    /// TODO
    TwoHopBlockmodel blockmodels[3];
    /// TODO
    int lower_difference();
    /// TODO
    int upper_difference();
};

#endif // CPPSBP_PARTITION_PARTITION_TRIPLET_HPP
