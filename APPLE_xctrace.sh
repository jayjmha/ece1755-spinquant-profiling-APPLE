#!/bin/bash

LLAMA=/path/to/executorch/cmake-out/examples/models/llama/llama_main
MODEL=/path/to/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8.pte
TOKENIZER=/path/to/tokenizer.model
PROMPT="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nExplain the water cycle in detail. Describe how water evaporates from oceans, lakes, rivers, and soil due to solar energy and wind, then rises into the atmosphere where it cools and condenses to form clouds and fog. Explain the different types of precipitation, including rain, snow, sleet, and hail, and the atmospheric conditions that produce each one. Also describe how water flows back through surface runoff, rivers, and underground groundwater aquifer systems before eventually returning to the ocean to complete the full cycle. Mention how human activity and climate change are affecting this process recent days.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
TEMPLATE="${1:-TopDown}"
OUTPUT="${2:-spinquant_3B.trace}"

# Remove old trace if exists
rm -rf "$OUTPUT"

# Launch llama_main suspended
$LLAMA \
  --model_path="$MODEL" \
  --tokenizer_path="$TOKENIZER" \
  --prompt="$PROMPT" \
  --temperature 0 --max_new_tokens=129 &
LLAMA_PID=$!
kill -STOP $LLAMA_PID
echo "llama_main PID: $LLAMA_PID (stopped)"

# Attach xctrace directly to the llama_main process (not bash)
xctrace record --template "$TEMPLATE" \
  --output "$OUTPUT" \
  --attach "$LLAMA_PID" &
XCTRACE_PID=$!
echo "xctrace PID: $XCTRACE_PID, waiting for it to attach..."

# Give xctrace time to set up
sleep 5

# Resume llama_main
echo "Resuming llama_main..."
kill -CONT $LLAMA_PID

# Wait for llama_main to finish
wait $LLAMA_PID
echo "llama_main finished. Stopping xctrace..."

# Give xctrace a moment to flush, then stop it
sleep 2
kill -INT $XCTRACE_PID 2>/dev/null
wait $XCTRACE_PID 2>/dev/null

echo "Done. Trace saved to: $OUTPUT"
