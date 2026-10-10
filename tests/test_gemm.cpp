#include <gtest/gtest.h>

import std;
import rho.core.matrix;
import rho.linalg.gemm;

using rho::core::aligned_vector;
using rho::core::matrix;

TEST(Gemm, IdentityMatrix) {
    aligned_vector<float> a = {1.0F, 2.0F, 3.0F, 4.0F};
    aligned_vector<float> b = {1.0F, 0.0F, 0.0F, 1.0F};
    const matrix<float> A{2, 2, std::move(a)};
    const matrix<float> B{2, 2, std::move(b)};

    matrix<float> C{2, 2};
    C.fill(0.0F);

    rho::linalg::gemm(A.view(), B.view(), C.view());

    ASSERT_EQ(C.rows(), 2U);
    ASSERT_EQ(C.cols(), 2U);

    EXPECT_FLOAT_EQ(C(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(C(0, 1), 2.0F);
    EXPECT_FLOAT_EQ(C(1, 0), 3.0F);
    EXPECT_FLOAT_EQ(C(1, 1), 4.0F);
}

TEST(Gemm, Int_A23_B32_C22) {
    const matrix<int> A{2, 3, aligned_vector<int>{1, 2, 3, 4, 5, 6}};
    const matrix<int> B{3, 2, aligned_vector<int>{1, 2, 3, 4, 5, 6}};
    matrix<int> C{2, 2};
    C.fill(0);

    rho::linalg::gemm(A.view(), B.view(), C.view());

    ASSERT_EQ(C.rows(), 2);
    ASSERT_EQ(C.cols(), 2);

    EXPECT_FLOAT_EQ(C(0, 0), 22);
    EXPECT_FLOAT_EQ(C(0, 1), 28);
    EXPECT_FLOAT_EQ(C(1, 0), 49);
    EXPECT_FLOAT_EQ(C(1, 1), 64);
}

TEST(Gemm, Float_A23_B32_C22) {
    const matrix<float> A{2, 3, aligned_vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
    const matrix<float> B{3, 2, aligned_vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}};
    matrix<float> C{2, 2};
    C.fill(0.0f);

    rho::linalg::gemm(A.view(), B.view(), C.view());

    ASSERT_EQ(C.rows(), 2);
    ASSERT_EQ(C.cols(), 2);

    EXPECT_FLOAT_EQ(C(0, 0), 22.0f);
    EXPECT_FLOAT_EQ(C(0, 1), 28.0f);
    EXPECT_FLOAT_EQ(C(1, 0), 49.0f);
    EXPECT_FLOAT_EQ(C(1, 1), 64.0f);
}

TEST(Gemm, Double_A23_B32_C22) {
    const matrix<double> A{2, 3, aligned_vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
    const matrix<double> B{3, 2, aligned_vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
    matrix<double> C{2, 2};
    C.fill(0.0);

    rho::linalg::gemm(A.view(), B.view(), C.view());

    ASSERT_EQ(C.rows(), 2);
    ASSERT_EQ(C.cols(), 2);

    EXPECT_FLOAT_EQ(C(0, 0), 22.0);
    EXPECT_FLOAT_EQ(C(0, 1), 28.0);
    EXPECT_FLOAT_EQ(C(1, 0), 49.0);
    EXPECT_FLOAT_EQ(C(1, 1), 64.0);
}

TEST(Gemm, AccumulatesIntoC) {
    const matrix<int> A{1, 2, aligned_vector<int>{1, 2}};
    const matrix<int> B{2, 1, aligned_vector<int>{3, 4}};
    matrix<int> C{1, 1};
    C.fill(100);

    rho::linalg::gemm(A.view(), B.view(), C.view());

    EXPECT_EQ(C(0, 0), 111); // 100 + 1*3 + 2*4
}

TEST(Transpose, TwoByThree) {
    const matrix<int> A{2, 3, aligned_vector<int>{1, 2, 3, 4, 5, 6}};
    matrix<int> At{3, 2};

    rho::linalg::transpose<int>(A.view(), At.view());

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(At(j, i), A(i, j)) << "i = " << i << ", j = " << j;
        }
    }
}
