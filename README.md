# ECE1755 - Profiling SpinQuant LLM on Apple M4 Pro

This project profiles **SpinQuant** (a quantized LLM model) and the original Llama model using **ExecuTorch** on Apple M4 Pro CPU. We use `os_signpost` instrumentation to capture detailed profiling of prefill/decode phases, forward passes, Fast Hadamard Transform (FHT) kernels, and XNNPACK delegates.

This is the final project for **ECE1755: Parallel Computer Architecture and Programming** at the University of Toronto.

## Environment Setup

### Prerequisites

- macOS with Apple Silicon (tested on M4 Pro)
- [Miniconda](https://docs.anaconda.com/miniconda/) installed
- Xcode with command-line tools

### 1. Clone and create conda environment

```bash
git clone -b release/1.1 https://github.com/jayjmha/ece1755-spinquant-profiling-APPLE.git
cd ece1755-spinquant-profiling-APPLE
conda create -yn executorch python=3.10.0
conda activate executorch
```

### 2. Install ExecuTorch

```bash
./install_executorch.sh
```

> **Note (Xcode 26 / AppleClang 21):** The upstream ExecuTorch `release/1.1` has a build failure in `third-party/flatcc` due to new compiler warnings treated as errors. This repo already includes the fix (`FLATCC_ALLOW_WERROR=OFF` in `third-party/CMakeLists.txt`). If you are building from the upstream repo instead, see the [fix details below](#flatcc-build-fix-details).

### 3. Install Llama requirements and build

```bash
./examples/models/llama/install_requirements.sh
make llama-cpu
```

### 4. Run inference

```bash
cmake-out/examples/models/llama/llama_main \
  --model_path=<model.pte> \
  --tokenizer_path=<tokenizer.model> \
  --prompt="<prompt>"
```

## Profiling with os_signpost

We instrument the following regions with `os_signpost`:

1. **Prefill / Decode** — The two main inference phases
2. **Each forward pass** — One large prefill phase + each individual decoding step
3. **FHT (Fast Hadamard Transform)** — Kernel introduced in SpinQuant
4. **XNNPACK delegates** — `XNNConvert`, `XNNFullyConnected`, labeled in signpost

### Generating a processor trace

Use the tracing script (requires an Instruments template to be set up first):

```bash
./path/to/ece1755-spinquant-profiling/APPLE_xctrace.sh
```

## Appendix

### Flatcc build fix details

When building upstream ExecuTorch `release/1.1` with Xcode 26, `./install_executorch.sh` fails with:

```
pprintint.h:388:13: error: implicit conversion loses integer precision: 'int' to 'int8_t' [-Werror,-Wimplicit-int-conversion-on-negation]
grisu3_print.h:186:33: error: initializer-string for character array is too long [-Werror,-Wunterminated-string-initialization]
```

**Fix** in `third-party/CMakeLists.txt`:

1. In the `ExternalProject_Add(flatcc_ep ...)` block, add `-DFLATCC_ALLOW_WERROR=OFF` to `CMAKE_ARGS`.
2. Before the `add_subdirectory(flatcc)` call, add `set(FLATCC_ALLOW_WERROR OFF CACHE BOOL "")`.

Then clean and rebuild:

```bash
rm -rf cmake-out pip-out build
./install_executorch.sh
```
