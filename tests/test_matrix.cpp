#include <gtest/gtest.h>

import std;
import rho.core.matrix;

using rho::core::aligned_buffer;
using rho::core::aligned_vector;
using rho::core::matrix;
using rho::core::matrix_view;

// check matrix views
static_assert(std::is_convertible_v<matrix_view<float>, matrix_view<const float>>);
static_assert(!std::is_convertible_v<matrix_view<const float>, matrix_view<float>>);

TEST(AlignedBuffer, DataIs64ByteAligned) {
	for (const std::size_t n : {1U, 3U, 100U, 785U}) {
		const aligned_buffer<float> buf{n};
		EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buf.data()) % 64U, 0U) << "n = " << n;
	}
}

TEST(AlignedBuffer, MoveLeavesSourceEmpty) {
	aligned_buffer<float> a{16};
	const float *p = a.data();
	const aligned_buffer<float> b{std::move(a)};
	EXPECT_EQ(b.data(), p); // the memory was moved, not copied
	EXPECT_EQ(b.size(), 16U);
	EXPECT_EQ(a.size(), 0U); // reading the moved-from object; should be 0U now
}

TEST(Matrix, RowsMatch) {
	const matrix<int> m{2, 3, aligned_vector<int>{0, 1, 2, 3, 4, 5}};
	EXPECT_EQ(m(0, 2), 2);
	EXPECT_EQ(m(1, 0), 3); // row 1 starts after the 3 elements of row 0
	EXPECT_EQ(m.row(1)[1], 4);
	EXPECT_EQ(m.view()(1, 2), 5);
}

TEST(Matrix, WrongBufferSizeThrows) {
	EXPECT_THROW((matrix<int>{2, 3, aligned_vector<int>{1, 2, 3}}), std::invalid_argument);
}

TEST(Matrix, OverflowingSizeThrows) {
	constexpr auto huge = std::numeric_limits<std::size_t>::max();
	EXPECT_THROW((matrix<float>{huge, 2}), std::length_error);
}

TEST(Matrix, MoveLeavesSourceEmpty) {
	// essentially same test as AlignedBuffer
	// since it's used in Matrix
	matrix<float> a{2, 3};
	const matrix<float> b{std::move(a)};
	EXPECT_EQ(b.rows(), 2U);
	EXPECT_EQ(b.cols(), 3U);
	EXPECT_EQ(a.rows(), 0U);
	EXPECT_EQ(a.cols(), 0U);
}

TEST(MatrixView, RowRangeStopsOutOfRange) {
	const matrix<int> m{4, 2, aligned_vector<int>{0, 1, 2, 3, 4, 5, 6, 7}};

	EXPECT_DEATH(static_cast<void>(m.view().row_range(3, 2)), "row_range: rows out of range");
	EXPECT_DEATH(static_cast<void>(m.view().row_range(5, 0)), "row_range: rows out of range");
}
