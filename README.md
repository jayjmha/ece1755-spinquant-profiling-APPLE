# ECE1755 - Profiling SpinQuant LLM on Apple M4 Pro

This project profiles **SpinQuant** (a quantized LLM model) and the original Llama model using **ExecuTorch** on Apple M4 Pro CPU using Xcode Instruments. We use `os_signpost` instrumentation to capture detailed profiling of prefill/decode phases, forward passes, Fast Hadamard Transform (FHT) kernels, and XNNPACK delegates.

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

### 1. Prefill / Decode signpost

**File:** `extension/llm/runner/text_llm_runner.cpp`

Added `os_signpost` interval markers around the Prefill and Decode phases in `TextLLMRunner::generate()`. All instrumentation is guarded by `#ifdef __APPLE__`.

**Changes:**

- Include `<os/signpost.h>` at the top:
  ```cpp
  #ifdef __APPLE__
  #include <os/signpost.h>
  #endif
  ```

- Create a log handle and signpost ID at the start of `generate()`:
  ```cpp
  #ifdef __APPLE__
  static os_log_t log =
      os_log_create("com.executorch.spinquant", "PointsOfInterest");
  static os_signpost_id_t spid = os_signpost_id_generate(log);
  #endif
  ```

- Wrap the prefill call with signpost begin/end:
  ```cpp
  #ifdef __APPLE__
  os_signpost_interval_begin(log, spid, "Prefill");
  #endif
  auto prefill_res = text_prefiller_->prefill(prompt_tokens, pos_);
  #ifdef __APPLE__
  os_signpost_interval_end(log, spid, "Prefill");
  #endif
  ```

- Wrap the decode (token generation) call with signpost begin/end:
  ```cpp
  #ifdef __APPLE__
  os_signpost_interval_begin(log, spid, "Decode");
  #endif
  auto generate_result = text_token_generator_->generate(
      prompt_tokens, pos_, max_new_tokens - 1, ...);
  #ifdef __APPLE__
  os_signpost_interval_end(log, spid, "Decode");
  #endif
  ```

These signposts appear in Instruments under the **Points of Interest** category with the subsystem `com.executorch.spinquant`.

### 2. Each forward pass signpost

**File:** `extension/llm/runner/text_decoder_runner.cpp`

Added `os_signpost` interval markers around every `TextDecoderRunner::step()` call. This captures each individual forward pass — the one large prefill forward pass and each decode step. All instrumentation is guarded by `#ifdef __APPLE__`.

**Changes:**

- Include `<os/signpost.h>` at the top:
  ```cpp
  #ifdef __APPLE__
  #include <os/signpost.h>
  #endif
  ```

- At the start of `step()`, create a per-call signpost ID and begin the interval (includes position and token count metadata):
  ```cpp
  #ifdef __APPLE__
  static os_log_t log =
      os_log_create("com.executorch.spinquant", "PointsOfInterest");
  os_signpost_id_t fwd_spid = os_signpost_id_generate(log);
  os_signpost_interval_begin(log, fwd_spid, "ForwardPass",
      "pos=%lld tokens=%zd", start_pos, tokens->numel());
  #endif
  ```

- End the interval before each return path (both kv-cache and non-kv-cache branches):
  ```cpp
  #ifdef __APPLE__
  os_signpost_interval_end(log, fwd_spid, "ForwardPass");
  #endif
  return outputs_res.get()[0].toTensor();
  ```

Each forward pass appears as a separate "ForwardPass" interval in Instruments. The prefill forward pass will have `tokens=N` (number of prompt tokens), while each decode step will have `tokens=1`.

### 3. FHT (Fast Hadamard Transform) signpost

**File:** `extension/llm/custom_ops/op_fast_hadamard_transform.cpp`

Added `os_signpost` interval markers around the `fast_hadamard_transform_out` kernel (`llama::fast_hadamard_transform.out`), which is the custom op introduced by SpinQuant. All instrumentation is guarded by `#ifdef __APPLE__`.

**Changes:**

- Include `<os/signpost.h>` at the top:
  ```cpp
  #ifdef __APPLE__
  #include <os/signpost.h>
  #endif
  ```

- At the start of `fast_hadamard_transform_out()`, begin the signpost interval:
  ```cpp
  #ifdef __APPLE__
  static os_log_t log =
      os_log_create("com.executorch.spinquant", "PointsOfInterest");
  os_signpost_id_t fht_spid = os_signpost_id_generate(log);
  os_signpost_interval_begin(log, fht_spid, "FHT");
  #endif
  ```

- End the interval before the return:
  ```cpp
  #ifdef __APPLE__
  os_signpost_interval_end(log, fht_spid, "FHT");
  #endif
  return out;
  ```

Each FHT invocation appears as an "FHT" interval in Instruments. This kernel only runs in SpinQuant models, so it will not appear when profiling the original (non-quantized) model.

### 4. XNNPACK delegate signpost (per-operator with op name)

**File:** `backends/xnnpack/third-party/XNNPACK/src/runtime.c`

Added `os_signpost` interval markers around each individual XNNPACK operator inside `xnn_invoke_runtime()`. Each signpost includes the actual operator name (e.g., `Fully Connected`, `Convert`, etc.) as metadata. All instrumentation is guarded by `#ifdef __APPLE__`.

**Changes:**

- Include `<os/signpost.h>` after the existing includes:
  ```c
  #ifdef __APPLE__
  #include <os/signpost.h>
  #endif
  ```

- Inside the operator loop in `xnn_invoke_runtime()`, wrap each `xnn_run_operator_with_index()` call:
  ```c
  #ifdef __APPLE__
  static os_log_t xnn_log_handle = NULL;
  if (xnn_log_handle == NULL) {
    xnn_log_handle = os_log_create("com.executorch.spinquant", "PointsOfInterest");
  }
  const char* op_name = xnn_operator_type_to_string_v2(runtime->opdata[i].operator_objects[j]);
  os_signpost_id_t op_spid = os_signpost_id_generate(xnn_log_handle);
  os_signpost_interval_begin(xnn_log_handle, op_spid, "XNNDelegate", "%s", op_name);
  #endif
  const enum xnn_status status = xnn_run_operator_with_index(...);
  #ifdef __APPLE__
  os_signpost_interval_end(xnn_log_handle, op_spid, "XNNDelegate", "%s", op_name);
  #endif
  ```

Each XNNPACK operator appears as an "XNNDelegate" interval in Instruments with the operator name (e.g., `Fully Connected (NC, F32)`, `Convert (NC, F16, F32)`) shown as metadata. This allows distinguishing between XNNConvert and XNNFullyConnected calls directly in the trace.

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
