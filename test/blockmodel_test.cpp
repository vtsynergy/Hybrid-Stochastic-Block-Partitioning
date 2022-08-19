#include <vector>

#include <gtest/gtest.h>

#include "blockmodel.hpp"

#include "toy_example.hpp"

class BlockmodelTest : public ToyExample {
};

class BlockmodelComplexTest : public ComplexExample {
};

TEST_F(BlockmodelTest, BlockDegreesAreCorrectlyInstantiated) {
    EXPECT_EQ(B.degrees_out(0), 8);
    EXPECT_EQ(B.degrees_out(1), 7);
    EXPECT_EQ(B.degrees_out(2), 8);
    EXPECT_EQ(B.degrees_in(0), 9);
    EXPECT_EQ(B.degrees_in(1), 7);
    EXPECT_EQ(B.degrees_in(2), 7);
    EXPECT_EQ(B.degrees(0), 10);
    EXPECT_EQ(B.degrees(1), 9);
    EXPECT_EQ(B.degrees(2), 9);
}

TEST_F(BlockmodelTest, MoveVertexWithDenseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(7, 2, Proposal.proposal, Updates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelTest, MoveVertexWithSparseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(7, 2, Proposal.proposal, SparseUpdates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelTest, MoveVertexWithBlockmodelDeltasIsCorrect) {
    B.move_vertex(7, Proposal.proposal, Deltas, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithDenseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(6, 3, Proposal.proposal, Updates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithSparseEdgeCountUpdatesIsCorrect) {
    B.move_vertex(6, 3, Proposal.proposal, SparseUpdates, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithBlockmodelDeltasIsCorrect) {
    B.move_vertex(6, Proposal.proposal, Deltas, new_block_degrees.block_degrees_out, new_block_degrees.block_degrees_in, new_block_degrees.block_degrees);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}

TEST_F(BlockmodelComplexTest, MoveVertexWithBlockmodelDeltasAndOnTheFlyBlockDegreesIsCorrect) {
    B.move_vertex(6, Deltas, Proposal);
    for (int row = 0; row < B.getNum_blocks(); ++row) {
        for (int col = 0; col < B.getNum_blocks(); ++col) {
            int val1 = B.blockmatrix()->get(row, col);
            int val2 = B2.blockmatrix()->get(row, col);
            EXPECT_EQ(val1, val2)
                                << "Blockmatrices differ at " << row << "," << col << " : using updates, value = " << val1
                                << " using assignment, value = " << val2;
        }
    }
    B.validate(graph);
}
