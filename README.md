# RHO

RHO (rhodopsin) is a basic neural network to classify black and white images made in C++.

## Docker

A dockerfile is added for convenience in case build dependencies can not be
installed. The dockerfile will build a statically linked binary.

### Build and run the program with docker:
> [!NOTE]
> - **rho-tests:latest**   `682MB` due to test framework
> - **rho:latest**         `3.96MB`

MNIST dataset (`not used yet`) needs to be mounted. It is not included in the image,
so use the fetch script first to download it.

```sh
sudo docker build -t rho .
sudo docker run --rm rho
# sudo docker run --rm -v "$PWD/data:/app/data:ro" rho
```
### Build and run the tests
```sh
sudo docker build --target tests -t rho-tests .
sudo docker run --rm -t rho-tests
# sudo docker run --rm -t -v "$PWD/data:/src/data:ro" rho-tests
```

## Prerequisites

```sh
sudo apt install build-essential cmake ninja-build gzip
```

> [!NOTE]
> Tested with:
>
> - **GCC:** `15.2.0`, `16.1.1`
> - **CMake:** `4.2.0`, `4.4.0`
> - **Ninja:** `1.13.2-1`

### MNIST dataset
Download from pytorch mirror using 
`./scripts/fetch_mnist.sh`

## Build and run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release 
cmake --build build -j$(nproc) && ./build/RHO
```

Or with script `./build.sh`

## Build and test
Run the following script to build and run tests in debug mode:
`./tests.sh`

## Documentation

[Specification Document](docs/specificationdocument.md)

### Weekly reports

[Week 1](/docs/reports/week1.md)

[Week 2](/docs/reports/week2.md)

[Week 3](/docs/reports/week3.md)

[Week 4](/docs/reports/week4.md)

[Week 5](/docs/reports/week5.md)
