cmake -S . -B build-tests -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRHO_BUILD_TESTS=ON
cmake --build build-tests && ./build-tests/RHO_tests
