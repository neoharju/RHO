# Implementation document


The code follows modern C++ modules paradigm and uses C++17, C++20, and C++23 
standards mainly (e.g. `std::unexpected`), and scratches some C++26 experimental
implementations e.g. `std::optional`.The author wanted to use C++26 `std::simd`,
but it seems to be premature still.

GCC 16.2 was used in development as modern modules had non-existent support on
clang using CMake (as far as the author is aware), and it also caused
issues with `clang-tidy`, which was also therefore unused.

## 1. General structure
```text
src/main.cpp ──► rho.app.cli          command line options (parse and check)
      │
      └────────► rho.train               fit(), train_epoch(), evaluate()
                   ├─► rho.io.dataset    dataset, compute_normalization, split, batcher
                   │     ├─► rho.io.idx            IDX file reader, error types
                   │     ├─► rho.io.normalization  z-score: x * scale + bias
                   │     └─► rho.core.rng          xoshiro256++, shuffle
                   └─► rho.optim         sgd
                         └─► rho.nn.mlp      network: forward, forward_backward
                               ├─► rho.nn.dense       z = xW + b and its gradients
                               │     └─► rho.linalg.gemm   GEMM kernels, transpose
                               ├─► rho.nn.activation  ReLU
                               └─► rho.nn.loss        softmax cross-entropy, accuracy
rho.core.matrix, rho.core.memory: matrix, matrix_view, aligned_buffer, require()
```
####  Rough logic:
1. Data is loaded via `dataset::load` that reads the MNIST files into an array.
2. `train_validation_split` splits the ids, and `compute_normalization`
   normalizes the training ids.
3. `batcher` makes 64 images (can be changed) into a batch
4. `mlp` runs the layers where an optimizer e.g. `sgd` updates the weights,
    `fit` repeats it for each epoch

- Memory is aligned in 64-bytes using `aligned_buffer` and `matrix`.
  - `aligned_allocator` (and thus `aligned_buffer` uses ideas from the
    `boost`-library.
- `matrix_view` is a pointer to the data with shape information.

## 2. Algorithms

One training step for a batch X (B × 784) and labels y:

```text
forward:   A_0 = X
           for l = 1 .. L:   Z_l = A_(l-1) · W_l + b_l            # GEMM, O(B · n_(l-1) · n_l)
                             A_l = ReLU(Z_l)   (not for l = L)    # O(B · n_l)
loss:      L = mean_i( LSE(Z_L[i]) − Z_L[i, y_i] )                # O(B · C)
           D_L = (softmax(Z_L) − onehot(y)) / B
backward:  for l = L .. 1:
               dW_l = A_(l-1)ᵀ · D_l                              # GEMM, O(B · n_(l-1) · n_l)
               db_l = sum of the rows of D_l                      # O(B · n_l)
               if l > 1:  D_(l-1) = (D_l · W_lᵀ) ⊙ [Z_(l-1) > 0]  # GEMM + ReLU', same order
update:    W_l = W_l − lr · dW_l,  b_l = b_l − lr · db_l           # O(W)
```

### GEMM
- The module `rho.linalg.gemm` uses a packed, cache-blocked GEMM using a 
  modified BLIS [9][10] method with AVX2 intrinsics. The author implemented the BLIS
  GEMM originally to another course (Aalto: Programming Parallel Computers). The 
  original BLIS methods is designed to encourage the CPU to keep one part of data
  in L3-cache and load it straight to registers where it is reused with data
  loaded from L1/L2 cache. The idea helps keeping the upcoming relevant data in 
  the CPU caches, and frequently used data in registers and L1-cache. 

  But the design to encourage loads from L3 was not ideal in our case since the 
  MNIST images are only of the size `28x28 = 768` elements, which fit easily to
  L1. 
- AVX-512 would have been more ideal performance wise, but it limits peer-testers
  by CPU a lot more.

- The GEMM has five loops. Each loop level keeps one part of the data in one cache level:

```text
for jc in steps of NC = 2048:              # B~ panel, KC x NC = 3 MiB     (L2/L3)
    for pc in steps of KC <= 384:          # blocks
        pack B[pc.., jc..] -> B~           # micro-panels of KC x 16
        for ic in steps of MC = 24:        # A~ block, MC x KC = 36 KiB    (L1/L2)
            pack A[ic.., pc..] -> A~       # micro-panels of 6 x KC
            for each B~ micro-panel:       # 24 KiB                        (L1)
                for each A~ micro-panel:
                    micro_kernel: C tile (6 x 16) += A~ micro-panel * B~ micro-panel
```
- Note, the values for NC, KC, MC, and even the C-tile size, can be
  changed and optimized for specific CPUs with different cache sizes.
- The GEMM uses parallelization simply with OpenMP using a `#pragma omp parallel` region.
  - Optimizing with e.g. `std::jthreads` was slower and would have required more
    investigation on scheduling and such to match OpenMP's performance.

### Backward pass
- `dense::backward()` computes:
  - `dW = x^T * dz` with `gemm_tn()`
  - `dW = dz * W^T` with `gemm()`

### Misc
- `xoshiro256++` was chosen as a pseudo random generator as the state is 256
  bits. C++ `rand()` is only 15 bits, while `std::mt19937` mersenne twister is
  by name 19937 bits, thus very good, but also a lot slower than our xoshiro
  variant.
  - xoshiro256++ implementation's C-code was available in [6] and translated
    into C++. This also includes the split-mix.
  - The jump-functions remain unused for now.
- `Box-Muller` followed largely the wikipedia implementation [8], but only chosing
  one z-part following a java-implementation in stack-overflow.
- `He` and `Glorot` initializations come from [4], [5] and used in many examples.
- Shuffling with bias reductions was implemented using Lemires examples in his 
  blog website [6].


## 3. Time and space complexity

From the pseudocode, with N samples, B batch size, E epochs, d = 784, C = 10 classes, n_l the
layer widths and W = Σ n_(l−1)·n_l the number of weights:

| Part | Time | Note |
|---|---|---|
| Load and check the data | O(N·d) | each byte one time |
| Normalization statistics | O(N·d) | exact integer sums |
| Shuffle per epoch | O(N) | Fisher–Yates |
| Batch copy + normalize | O(B·d) per batch | O(N·d) per epoch |
| Forward pass | O(B·W) per batch | 2·W FLOP per sample |
| Backward pass | O(B·W) per batch | dW: 2·W, dx: 2·(W − d·n_1) FLOP per sample |
| SGD step and Wᵀ copy | O(W) per batch | O(N·W / B) per epoch |
| One epoch | **O(N·W)** | the GEMMs dominate |
| Full training | **O(E·N·W)** | |
| Test evaluation | O(N_test·W) | forward pass only |

For the default network (784-256-128-10), W = 234 752. One training sample costs about
1.0 MFLOP, so one epoch (55 000 training + 5 000 validation samples) is about 58 GFLOP.

| Memory | Size | Default network |
|---|---|---|
| MNIST data (bytes) | O(N·d) | 47 MB train + 7.8 MB test |
| Sample ids | O(N) | 0.3 MB |
| W, dW, b, db, Wᵀ copies | O(W) | about 2 MB |
| Batch and activation buffers | O(B·(d + Σ n_l)) | about 0.6 MB for B = 64 |
| GEMM packing buffers (per thread) | O((MC + min(n, NC)) · KC) per thread, at most 3 MiB + 36 KiB | about 0.1 MB per thread |


## 4. Performance
- The GEMM implementation is about 3x faster than the school-boys'
  `i-k-j`-performance variant from `i-j-k`.

- A PyTorch implementation at root (torch_mnist.py) is doing 5 seconds per epoch
  with the same settings as RHO is doing in 0.5 seconds per epoch, so it is 10x
  slower.

## 5. Suggestions for improvement
- MNIST training specifically:
 - generate more data by engineering existing training set:
   - rotate/blur/shift/cut etc existing data to create more training data
 - Test with different networks
- Further optimize parallelization techniques
  - minimize thread-wait regions
  - Explore packing and batch sizes
  - optimize block/kernel sizes more
- Include AVX-512 support
- Include AVX10 support using deep-learning intrinsics with 8-bit/16-bit floats.
- When a complete network is ready, one could lessen the abstraction, as much as
  possible to reduce overhead: fused kernels.
- Add regularization techniques and early stopping.
- Consider more layers like Dropout, Convolution layers, Transformers etc.
- Make dataload to support more datasets.
- UI implementation similar to Tensorboard
- Transferlearning from previous checkpoints or other trained models.
- GPU kernel support (CUDA)

6. Use of large language models
- `Antrophic Claude` was used to:
  - generate markdown tables in markdown docs from given data.
  - transform author made code-comments into doxygen style per requirements
    - this also caused some comments to be more `robotic`
    - this also added comments to places, which is weird to have comments in but
      left by the author, since it will be transferred to the 
  - fix an issue mnist download script, where it also added size checks, which
    feels a bit redundant since I already had sha-sum tests.
  - spotting bugs
  - explaining task-array idea in C++ that I had used in python, but did not
    know how to implement in C++.
  - help fixing expected_death tests to not sit idle for 3seconds, but continue instantly
- In general, `claude` was quite bad in modern C++, and needed babysitting to
  install modern GCC v16.2, use the CMake I had made with specific CMake version
  checks, just to help with a few google-test issues. It is probably mostly
  trained on C++17 code.

# 7. Sources

1. Y. LeCun, C. Cortes, C. Burges. *The MNIST database of handwritten digits.* http://yann.lecun.com/exdb/mnist/
2. M. Nielsen. *Neural Networks and Deep Learning*, ch. 1–3. http://neuralnetworksanddeeplearning.com/
3. Wikipedia. *Backpropagation.* https://en.wikipedia.org/wiki/Backpropagation
4. K. He et al. *Delving Deep into Rectifiers* (He initialization), ICCV 2015. https://arxiv.org/abs/1502.01852
5. X. Glorot, Y. Bengio. *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010.
6. D. Blackman, S. Vigna. *Scrambled Linear Pseudorandom Number Generators.* https://arxiv.org/abs/1805.01407 and https://prng.di.unimi.it/
7. D. Lemire. *A fast alternative to the modulo reduction* and *Fast random shuffling*, 2016. https://lemire.me/blog/
8. Wikipedia. *Box–Muller transform.* https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
9. K. Goto, R. van de Geijn. *Anatomy of High-Performance Matrix Multiplication*, ACM TOMS 34(3), 2008.
10. F. Van Zee, R. van de Geijn. *BLIS: A Framework for Rapidly Instantiating BLAS Functionality*, ACM TOMS 41(3), 2015.
11. Stanford CS231n. *Neural Networks Part 3: gradient checks.* https://cs231n.github.io/neural-networks-3/#gradcheck
12. S. Bjorkqvist. *Writing automated tests for neural networks.* https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/
13. Boost.Align, `aligned_allocator`. https://www.boost.org/doc/libs/release/libs/align/

