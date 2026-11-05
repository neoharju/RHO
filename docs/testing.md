# Testing document

## 1. Summary

- RHO has 80 unit tests in 24 test suites. The tests use GoogleTest 1.18.0.
- All tests pass in the Debug build and in the Release build (GCC 16.2.0, CMake 4.4.5).
- The line coverage of the library (modules/) is 90 %.
- One end-to-end test uses the real MNIST data. If the MNIST files are not in `data/`,
  GoogleTest skips this test.

## 2. Coverage report

The report shows the line coverage of the library code in `modules/`. The report does not
include the tests and `src/main.cpp`. Section 5 gives the commands.

| File | Lines | Covered | Coverage |
|---|---:|---:|---:|
| `modules/core/matrix.cppm` | 55 | 55 | 100 % |
| `modules/core/memory.cppm` | 31 | 27 | 87 % |
| `modules/core/rng.cppm` | 77 | 75 | 97 % |
| `modules/io/dataset.cppm` | 87 | 76 | 87 % |
| `modules/io/idx.cppm` | 91 | 40 | 44 % |
| `modules/io/normalization.cppm` | 6 | 6 | 100 % |
| `modules/linalg/gemm.cppm` | 181 | 180 | 99 % |
| `modules/nn/activation.cppm` | 17 | 17 | 100 % |
| `modules/nn/dense.cppm` | 53 | 53 | 100 % |
| `modules/nn/loss.cppm` | 62 | 62 | 100 % |
| `modules/nn/mlp.cppm` | 65 | 64 | 98 % |
| `modules/optim/optim.cppm` | 11 | 11 | 100 % |
| `modules/train/trainer.cppm` | 44 | 43 | 97 % |
| **Total** | **780** | **709** | **90 %** |


## 3. What was tested and how

The tests compare the results with known values, with simple reference code, or with
statistical limits. All random inputs use fixed seeds, so each run is the same.

| Area (test suites) | Tests | What the tests check |
|---|---:|---|
| GEMM (`Gemm`, `GemmTn`, `GemmNt`, `GemmBatch`) | 19 | `gemm()`, `gemm_tn()` and `gemm_nt()` agree with a `long double` reference (tolerance 1e-5·(k + 1)). In the Release build, they give the same bits as the simple i-k-j loop. The results are bit-identical with 1, 2, 3 and 4 threads.  Wrong shapes aborts the program. |
| Gradient check (`GradientCheck`) | 1 | Backpropagation for numeric gradients See below. |
| Training pipeline (`Training`), from S. Bjorkqvist [12] | 4 | The network overfits 48 samples with random labels (loss less than 0.01, all samples correct). After one step, the gradient of each parameter matrix is not 0, and the loss is smaller. One SGD step changes all 6 parameter matrices. A batch of 6 samples gives the same outputs as the 6 samples one at a time. |
| Network and layers (`Mlp`, `Dense`) | 7 | Bad topologies throw. The same seed gives the same weights. He initialization has mean 0 and standard deviation sqrt(2/784), both within 1e-3. Bad shapes and labels stop the program. Ten training steps give bit-identical weights with 1, 2 and 4 threads. `dense::backward()` agrees with simple sums for batch 8 and batch 3. |
| Loss and activation (`ReLU`, `Softmax`, `SoftmaxCrossEntropy`, `CrossEntropy`, `ArgmaxRow`, `CountCorrect`) | 12 | ReLU forward and backward for int, float and double with −2, −0.0, 0, 3.5 and 1e-30 (−0.0 gives +0.0, and ReLU'(0) = 0). The logits (1000, 1000, 1000) give the finite loss ln 3. Each gradient row sums to 0, each softmax row sums to 1, and cross-entropy is the mean of −ln p. If two scores are equal, `argmax_row()` gives the smaller index. |
| Random numbers and shuffle (`xoshiro256pp`, `Shuffle`) | 17 | The same seed gives the same sequence. Chi-square tests with p = 0.001: `bounded(10)` with 100 000 values (less than 27.88), and the 6 orders of a shuffle of 3 elements with 60 000 shuffles (less than 20.52). `normal()` has mean 0 and variance 1 within 0.02. Other tests: the rejection loop of `bounded()`, seed 0, and `jump()`. |
| Data pipeline (`Batcher`, `Split`, `Normalization`) | 11 | The tests write small IDX files with known pixels and labels. 10 samples in sequential order with batch size 4 give batches of 4, 4 and 2 samples, and each label stays with its image. Each shuffled epoch contains each sample one time, in a new order. Mean 1 and standard deviation 2 give (x − 1) / 2. A split of 100 ids into 70 and 30 loses no id. `fit()` learns 40 samples to 100 % validation accuracy. |
| Memory and matrix (`AlignedBuffer`, `AlignedAllocator`, `Matrix`, `MatrixView`) | 8 | Data addresses are multiples of 64 bytes. A move leaves the source empty. Sizes that overflow throw. `row_range()` outside the matrix stops the program. |
| End-to-end on MNIST (`Mnist`) | 1 | A 784-64-10 network trains on the first 10 000 training images for 4 epochs. The loss must become smaller in each epoch, and the test accuracy must be more than 90 %. So, the result was 94.54 %. |

### Gradient check

The gradient check tests the core of the project: backpropagation.

- The network is 13-17-16-10, with a batch of 7 samples (max batch 8). These sizes make
  the GEMM use its edge tiles. The test does one SGD step first, so that the biases are
  not 0.
- For each of the 696 parameters, the test compares the backpropagation gradient with the
  central difference (L(w + h) − L(w − h)) / 2h. L is a reference loss in double
  precision, and h = 1e-6. The tolerance is 1e-6 + 1e-4·|gradient|. The largest error was
  approximately 1e-7.
- The float loss of the network must be equal to the reference loss within 1e-5.
- The reference must use double precision. With a float loss, the rounding error is too
  large for a small h, and a large h can cross the kink of ReLU at 0 (see CS231n [11]).
- A mutation check put four bugs into the backward pass, one at a time. The gradient check
  found all four bugs. Two of them (`db` does not add the last row of `dz`; `db` does not
  start from 0) passed all the other 79 tests.

## 4. Types of inputs

- Small matrices with values that were calculated by hand.
- Random values (uniform in [−1, 1), or normal) from xoshiro256++ with fixed seeds.
- Sizes at the limits: 0 rows or columns, 1×1×1, sizes that are not multiples of the
  6×16 tile, and sizes larger than one cache block.
- Special values: NaN, −0.0, 0, 1e-30 and 1000.
- Bad arguments: e.g. shapes that do not agree, labels that are not 0 to 9, batch size 0, and
  ids that are not in the data set.
- Small IDX files that the tests write, and the real MNIST files.

## 5. How can tests be reproduced

Download MNIST first (for the end-to-end test), then build and run all tests in the Debug
build:

```sh
./scripts/fetch_mnist.sh
./tests.sh
```

Run the tests in the Release build, or with a fixed number of threads, or only one suite:

```sh
cmake -S . -B build-release-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DRHO_BUILD_TESTS=ON
cmake --build build-release-tests && ./build-release-tests/RHO_tests
OMP_NUM_THREADS=1 ./build-tests/RHO_tests
./build-tests/RHO_tests --gtest_filter='GradientCheck.*'
```

Make the coverage report. You need gcovr 8 or newer (`pip install gcovr`). gcov must come
from the same GCC version as the compiler: if necessary, add
`--gcov-executable gcov-16`. The tests take approximately 2 minutes in this build.

```sh
cmake -S . -B build-coverage -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRHO_BUILD_TESTS=ON -DRHO_COVERAGE=ON
cmake --build build-coverage && ./build-coverage/RHO_tests
gcovr -r . build-coverage --filter 'modules/' --merge-lines --txt
```

Use `--merge-lines`. Each instantiation of a template (for example `fit()`) has its own
copy of the lines. Without this option, gcovr shows a line as not covered if one copy did
not run.

## 6. Empirical results

The Release program (`./build.sh`) trains the default network: 784-256-128-10, batch 64,
10 epochs, learning rate 0.1 multiplied by 0.95 after each epoch, seed 42. It uses 55 000
training samples and 5 000 validation samples. The training values come from the batches
of each epoch, before the update of each batch.

| Epoch | Learning rate | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1000 | 0.2267 | 93.08 % | 0.1380 | 95.66 % |
| 2 | 0.0950 | 0.0886 | 97.35 % | 0.1255 | 96.24 % |
| 3 | 0.0903 | 0.0559 | 98.30 % | 0.0952 | 97.04 % |
| 4 | 0.0857 | 0.0365 | 98.91 % | 0.0934 | 97.32 % |
| 5 | 0.0815 | 0.0239 | 99.34 % | 0.1096 | 97.06 % |
| 6 | 0.0774 | 0.0157 | 99.64 % | 0.0888 | 97.50 % |
| 7 | 0.0735 | 0.0103 | 99.81 % | 0.0876 | 97.66 % |
| 8 | 0.0698 | 0.0067 | 99.92 % | 0.0917 | 97.58 % |
| 9 | 0.0663 | 0.0047 | 99.97 % | 0.0927 | 97.50 % |
| 10 | 0.0630 | 0.0036 | 99.98 % | 0.0907 | 97.64 % |

- Test set: loss 0.0705 and accuracy 98.19 % (9 819 of 10 000). The goal in the
  specification was more than 90 %.
- The validation loss does not become smaller after epoch 7, but the training accuracy
  increases to 99.98 %. This shows that the network starts to overfit. Early stopping or
  regularization can help (see the implementation document, section 5).
- Time: 0.4 to 0.5 seconds for one epoch, and 4.6 seconds for the full run with the data
  load. Using a 2.1 GHz 4 core Intel i7 laptop. 
- A second run with `OMP_NUM_THREADS=2` gave the same loss and accuracy values in all
  epochs. The results do not change with the number of threads.

The sources [11] and [12] are in the [implementation document](implementation.md#7-sources).
