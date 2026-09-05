# RHO

RHO (rhodopsin) is a basic neural network to classify black and white images made in C++.

## Prerequisites

```sh
sudo apt install build-essential cmake ninja-build
```

> [!NOTE]
> Tested with:
>
> - **GCC:** `15.2.0`, `16.1.1`
> - **CMake:** `4.2.0`, `4.4.0`
> - **Ninja:** `1.13.2-1`

## Build and run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release 
cmake --build build -j$(nproc) && ./build/RHO
```

Or with script `./build.sh`

## Documentation

[Specification Document](docs/specificationdocument.md)

### Weekly reports

[Week 1](/docs/reports/week1.md)
