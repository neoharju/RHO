include(CMakePrintHelpers)

cmake_minimum_required(VERSION 4.2)

# need specific id due to experimental
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
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_SCAN_FOR_MODULES ON)
# Initializes CXX_MODULE_STD ON for every target, so `import std;`
# is usable without setting the property per-target.
set(CMAKE_CXX_MODULE_STD ON)
add_executable(RHO)

# Module interface units
target_sources(RHO
  PRIVATE
    FILE_SET CXX_MODULES
    BASE_DIRS ${PROJECT_SOURCE_DIR}/modules
    FILES
      modules/foo.cppm
)
target_sources(RHO
  PRIVATE
    src/main.cpp
)

target_compile_options(RHO
    PRIVATE
    -march=native
        -Wall
        -Wextra
        -Wpedantic
)
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

