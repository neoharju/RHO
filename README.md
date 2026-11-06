# RHO

RHO (rhodopsin) is a basic neural network to classify black and white images made in C++.

## Docker

A dockerfile is added for convenience in case build dependencies can not be
installed. The dockerfile will build a statically linked binary.

### Build and run the program with docker:
> [!NOTE]
> - **rho-tests:latest**   `682MB` due to test framework
> - **rho:latest**         `3.96MB`

MNIST dataset needs to be mounted. It is not included in the image,
so use the fetch script first to download it.

```sh
./scripts/fetch_mnist.sh
sudo docker build -t rho .
sudo docker run --rm -v "$PWD/data:/app/data:ro" rho
```
### Build and run the tests
```sh
sudo docker build --target tests -t rho-tests .
sudo docker run --rm -t -v "$PWD/data:/src/data:ro" rho-tests
```

If you run the image on a different computer, add `--build-arg RHO_MARCH=x86-64-v3`
to `docker build`.

## Prerequisites

- A Linux computer with an x86-64 CPU that has AVX2 and FMA.
- GCC 16 or newer, CMake 4.2 or newer, and Ninja.
- For the download script: bash, curl, gzip and sha256sum.

```sh
sudo apt install build-essential cmake ninja-build gzip curl
```

Make sure that the versions are new enough: `g++ --version` and `cmake --version`.
If they are too old, use Docker.

> [!NOTE]
> Tested with:
>
> - **GCC:** `16.1.1`, `16.2.0`
> - **CMake:** `4.2.0`, `4.4.0`, `4.4.5`
> - **Ninja:** `1.13.2-1`

### MNIST dataset
The MNIST data is not in the repository. Download it from the PyTorch mirror

```sh
./scripts/fetch_mnist.sh
```

## Build and run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release 
cmake --build build -j$(nproc) && ./build/RHO
```

Or with script `./build.sh`

Start the program from the project root, because it reads the MNIST files from `data/`.

## Build and test
Run the following script to build and run tests in debug mode:

```sh
./tests.sh
```

If the MNIST files are not in `data/`, one test is skipped.
See the [User Guide](docs/userguide.md) for the settings, the output and the error messages.

## Documentation

[Specification Document](docs/specificationdocument.md)

[Implementation Document](docs/implementation.md)

[Testing Document](docs/testing.md)

[User Guide](docs/userguide.md)

### Weekly reports

[Week 1](docs/reports/week1.md)

[Week 2](docs/reports/week2.md)

[Week 3](docs/reports/week3.md)

[Week 4](docs/reports/week4.md)

[Week 5](docs/reports/week5.md)

[Week 6](docs/reports/week6.md)

[Week 7](docs/reports/week7.md)
