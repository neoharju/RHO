# User guide

RHO trains a neural network (a multilayer perceptron) that classifies the handwritten
digits 0 to 9 of the MNIST data set. This guide tells you how to build, run and test the
program.

## 1. Requirements

- A Linux computer with an x86-64 CPU that has AVX2 and FMA, for example Intel Haswell
  (2013) or newer, or AMD Zen or newer.
- GCC 16 or newer. The project is tested with GCC 16.1 and 16.2. The build is not
  tested with other compilers, and Clang does not support modules.
- CMake 4.2 or newer (tested with 4.2.0, 4.4.0 and 4.4.5) and Ninja.
- For the download script: bash, curl, gzip and sha256sum.

On Debian or Ubuntu, install the tools with:

```sh
sudo apt install build-essential cmake ninja-build gzip curl
```

Make sure that the versions are new enough: `g++ --version` and `cmake --version`. Many
distributions have older versions. If your versions are too old, use Docker (section 7).

## 2. Get the MNIST data

The MNIST data is not in the Git repository. Run this script in the project root:

```sh
./scripts/fetch_mnist.sh
```

The script downloads the four MNIST files from the PyTorch mirror into `data/`. It checks
the SHA-256 sum of each file and unpacks the files. If the files are already in `data/`,
the script only checks them.

## 3. Build and run

Build the Release version and start it from the project root:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/RHO
```

The script `./build.sh` does the same steps. The program reads the MNIST files from
`data/` in the current directory. For this reason, start the program from the project
root.

The program uses all CPU cores. To set the number of threads, use `OMP_NUM_THREADS`, for
example `OMP_NUM_THREADS=4 ./build/RHO`. The results do not change with the number of
threads.

Build options (add them to the first `cmake` command as `-D<option>=<value>`):

| Option | Default | Use |
|---|---|---|
| `RHO_MARCH` | `native` | Value for `-march`. Use `x86-64-v3` if you run the program on a different computer. |
| `RHO_BUILD_TESTS` | `OFF` | `ON` also builds the tests (`RHO_tests`). |
| `RHO_COVERAGE` | `OFF` | `ON` builds with gcov coverage (see the testing document). |
| `RHO_SANITIZE` | empty | Value for `-fsanitize`, for example `address,undefined`. |

## 4. Output

Example output with the default settings:

```text
Network [784, 256, 128, 10]: 55000 train, 5000 validation, 10000 test samples
epoch  1  lr 0.1000  train loss 0.2267 acc 93.08%  val loss 0.1380 acc 95.66%  0.5s
...
epoch 10  lr 0.0630  train loss 0.0036 acc 99.98%  val loss 0.0907 acc 97.64%  0.5s
Test: loss 0.0705, accuracy 98.19% (9819 / 10000)
```

- The first line shows the layer sizes and the number of samples in each set. The
  validation set is a part of the MNIST training file.
- Each `epoch` line shows the learning rate (`lr`), the loss and the accuracy of the
  training batches, the loss and the accuracy on the validation set, and the time of the
  epoch.
- The last line shows the result on the 10 000 test images.

## 5. Change the settings

The program has no command line options. The settings are constants at the start of
`src/main.cpp`, in namespace `app`:

| Constant | Default | Meaning |
|---|---|---|
| `data_dir` | `"data"` | Folder of the MNIST files, relative to the current directory. |
| `layer_sizes` | `{784, 256, 128, 10}` | Layer widths: input, hidden layers, classes. |
| `epochs` | `10` | Number of passes over the training data. |
| `batch_size` | `64` | Number of samples in one mini-batch. |
| `learning_rate` | `0.1f` | Learning rate of the first epoch. |
| `lr_decay` | `0.95f` | After each epoch, the program multiplies the learning rate by this value. |
| `validation_size` | `5000` | Number of training samples for validation. |
| `seed` | `42` | One seed for the split, the start weights and the shuffle. |

Change a value, then build again (section 3). Keep 784 as the first layer size and 10 as
the last layer size. The same seed and settings give the same results.

## 6. Run the tests

```sh
./tests.sh
```

The script builds the tests in the Debug build (`build-tests/`) and runs all of them. If
the MNIST files are not in `data/`, one test (`Mnist.SmallNetworkLearnsToNinetyPercent`)
is skipped. The [testing document](testing.md) tells more about the tests and the coverage
report.

## 7. Docker

If you cannot install GCC 16 and CMake 4.2, use Docker. The image builds a static
program. The image does not contain MNIST: run `./scripts/fetch_mnist.sh` first.

```sh
sudo docker build -t rho .
sudo docker run --rm -v "$PWD/data:/app/data:ro" rho
```

Build and run the tests:

```sh
sudo docker build --target tests -t rho-tests .
sudo docker run --rm -t -v "$PWD/data:/src/data:ro" rho-tests
```

If you run the image on a different computer, add `--build-arg RHO_MARCH=x86-64-v3` to
`docker build`.

## 8. Error messages

For a data error, the program writes a message and stops with exit code 1.

| Message | Cause | What to do |
|---|---|---|
| `✘ Cannot open 'data/train-images-idx3-ubyte': No such file or directory (2)` | The MNIST files are not in `data/`, or you started the program in a different directory. | Run `./scripts/fetch_mnist.sh`. Start the program from the project root. |
| `✘ Cannot read '…': it is not an IDX file (still zipped?).` | The file is still compressed, or it is not an IDX file. | Run `./scripts/fetch_mnist.sh` again. |
| `✘ Cannot read '…': its size does not match its header.` | The file is damaged or not complete. | Delete `data/` and run `./scripts/fetch_mnist.sh` again. |
| `✘ Cannot keep … samples for validation: the training set has ….` | `validation_size` is too large. | Make `validation_size` smaller in `src/main.cpp`. |
| `… precondition failed: …` | A programming error, for example a wrong first or last layer size. | Undo the last change in `src/main.cpp`. |

## 9. API documentation

Run `doxygen` in the project root. Then open `build-docs/html/index.html` in a web
browser.
