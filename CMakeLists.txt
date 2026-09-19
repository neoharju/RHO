cmake_minimum_required(VERSION 4.2)

include(CMakePrintHelpers)

# need specific id due to CMakes experimental module support
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.4.0")
	set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
		"f35a9ac6-8463-4d38-8eec-5d6008153e7d"
	)
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "4.3.0")
	set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
		"451f2fe2-a8a2-47c3-bc32-94786d8fc91b"
	)
elseif(CMAKE_VERSION VERSION_LESS "4.3.0")
	set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
		"d0edc3af-4c50-42ea-a356-e2862fe7a444"
	)
endif()
cmake_print_variables(CMAKE_VERSION CMAKE_EXPERIMENTAL_CXX_IMPORT_STD)

project(RHO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_SCAN_FOR_MODULES ON)
# Initializes CXX_MODULE_STD ON for every target, so `import std;`
# is usable without setting the property per-target.
set(CMAKE_CXX_MODULE_STD ON)

option(RHO_BUILD_TESTS "Build RHO tests" OFF)

# --- RHO LIBRARY ----------------------------------
add_library(rho_lib)

target_sources(rho_lib
	PUBLIC
	FILE_SET CXX_MODULES
	BASE_DIRS
		${PROJECT_SOURCE_DIR}/modules
	FILES
		modules/core/matrix.cppm
		modules/linalg/gemm.cppm
		modules/nn/activation.cppm
		modules/nn/loss.cppm
)


target_compile_options(rho_lib
	PRIVATE
	-march=native
	-Wall
	-Wextra
	-Wpedantic
)
# -------------------------------------------------

add_executable(RHO
	src/main.cpp
)

target_link_libraries(RHO
	PRIVATE
	rho_lib
)

target_compile_options(RHO
	PRIVATE
	-march=native
	-Wall
	-Wextra
	-Wpedantic
)
# ------------------------------------------------

# LTO if available in Release mode
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

if(ipo_supported)
	set_target_properties(RHO PROPERTIES
		INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
	)
else()
	message(WARNING "LTO not supported: ${ipo_error}")
endif()

# --- Tests --------------------------------------

if(RHO_BUILD_TESTS)

	enable_testing()
	include(FetchContent)

	FetchContent_Declare(
		googletest
		GIT_REPOSITORY https://github.com/google/googletest.git
		GIT_TAG v1.18.0
	)

	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

	FetchContent_MakeAvailable(googletest)

	add_executable(RHO_tests
		tests/test_gemm.cpp
		tests/test_nn.cpp
	)

	target_link_libraries(RHO_tests
		PRIVATE
		rho_lib
		GTest::gtest_main
	)

	include(GoogleTest)

	gtest_discover_tests(RHO_tests)

endif()
