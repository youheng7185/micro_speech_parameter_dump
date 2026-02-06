/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/common.h"
#include <stdio.h>

#define MP_INT(name, val) \
  MicroPrintf("%s = %d\n", name, (int)(val))

#define MP_PTR(name, ptr) \
  MicroPrintf("%s = %p\n", name, (const void*)(ptr))

// Print first N int8 values
inline void MicroPrintInt8Array(const char* name,
                                const int8_t* data,
                                int count, int max_n = 16) {
  MicroPrintf("%s:", name);
  int n = count < max_n ? count : max_n;
  for (int i = 0; i < n; i++) {
    MicroPrintf(" %d", data[i]);
  }
  if (count > max_n) MicroPrintf(" ...");
  MicroPrintf("\n");
}

// Print first N int32 values
inline void MicroPrintInt32Array(const char* name,
                                 const int32_t* data,
                                 int count, int max_n = 16) {
  MicroPrintf("%s:", name);
  int n = count < max_n ? count : max_n;
  for (int i = 0; i < n; i++) {
    MicroPrintf(" %ld", (long)data[i]);
  }
  if (count > max_n) MicroPrintf(" ...");
  MicroPrintf("\n");
}

inline void MicroDumpInt8AsCArray(const char* name,
                                  const int8_t* data,
                                  int length) {
  MicroPrintf("\n// ===== BEGIN %s =====\n", name);
  MicroPrintf("const int8_t %s[%d] = {\n", name, length);

  for (int i = 0; i < length; i++) {
    // indent every line
    if ((i % 16) == 0) {
      MicroPrintf("  ");
    }

    MicroPrintf("%d", data[i]);

    if (i != length - 1) {
      MicroPrintf(", ");
    }

    if ((i % 16) == 15) {
      MicroPrintf("\n");
    }
  }

  if ((length % 16) != 0) {
    MicroPrintf("\n");
  }

  MicroPrintf("};\n");
  MicroPrintf("// ===== END %s =====\n\n", name);
}

inline void DumpInt8TensorAsCArray_OneLine(
    const char* name,
    const int8_t* data,
    int size) {

  MicroPrintf("\n// ===== BEGIN %s =====\n", name);
  MicroPrintf("static const int8_t %s[%d] = {\n", name, size);

  char line[160];
  int pos = 0;

  for (int i = 0; i < size; i++) {
    pos += snprintf(&line[pos], sizeof(line) - pos,
                    "%d", data[i]);

    if (i != size - 1) {
      pos += snprintf(&line[pos], sizeof(line) - pos, ", ");
    }

    if ((i & 0x0F) == 0x0F || i == size - 1) {
      MicroPrintf("  %s\n", line);
      pos = 0;
      line[0] = '\0';
    }
  }

  MicroPrintf("};\n");
  MicroPrintf("// ===== END %s =====\n\n", name);
}

inline void DumpInt32ArrayAsCArray_OneLine(
    const char* name,
    const int32_t* data,
    int size) {

  MicroPrintf("\n// ===== BEGIN %s =====\n", name);
  MicroPrintf("static const int32_t %s[%d] = {\n", name, size);

  char line[160];
  int pos = 0;

  for (int i = 0; i < size; i++) {
    pos += snprintf(&line[pos], sizeof(line) - pos, "%d", data[i]);

    if (i != size - 1) {
      pos += snprintf(&line[pos], sizeof(line) - pos, ", ");
    }

    if ((i & 0x0F) == 0x0F || i == size - 1) {
      MicroPrintf("  %s\n", line);
      pos = 0;
      line[0] = '\0';
    }
  }

  MicroPrintf("};\n");
  MicroPrintf("// ===== END %s =====\n\n", name);
}


namespace tflite {
namespace reference_integer_ops {

// For per-channel functions, since it is defined in quantization spec that
// weights are symmetric
// (https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric),
// zero_point (params.weights_offset) is always 0.
// However, for per-tensor functions, params.weights_offset is still applied for
// backward compatibility.
template <typename InputType, typename WeightType, typename OutputType,
          typename BiasType>
void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape& input_shape,
    const InputType* input_data, const RuntimeShape& filter_shape,
    const WeightType* filter_data, const RuntimeShape& bias_shape,
    const BiasType* bias_data, const RuntimeShape& output_shape,
    OutputType* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();

  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      BiasType acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled = MultiplyByQuantizedMultiplier(
          acc, output_multiplier[out_c], output_shift[out_c]);
      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] =
          static_cast<OutputType>(acc_scaled);
    }
  }
}

// This implementation receives the scales in float and performs requant in
// float to avoid loss of precision.
template <typename InputType, typename WeightType, typename OutputType,
          typename BiasType>
void FullyConnectedPerChannel(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const InputType* input_data, const RuntimeShape& filter_shape,
    const WeightType* filter_data, const RuntimeShape& bias_shape,
    const BiasType* bias_data, const RuntimeShape& output_shape,
    float input_scale, float output_scale, const float* filter_scales,
    OutputType* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();

  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      BiasType acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }

      const float scale = filter_scales[out_c];
      const double filter_scale = static_cast<double>(scale);
      const double effective_output_scale = static_cast<double>(input_scale) *
                                            filter_scale /
                                            static_cast<double>(output_scale);
      int32_t acc_scaled = static_cast<int32_t>(
          round(static_cast<double>(acc) * effective_output_scale));

      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] =
          static_cast<OutputType>(acc_scaled);
    }
  }
}

template <typename InputType, typename WeightType, typename OutputType,
          typename BiasType>
void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape,
                    const InputType* input_data,
                    const RuntimeShape& filter_shape,
                    const WeightType* filter_data,
                    const RuntimeShape& bias_shape, const BiasType* bias_data,
                    const RuntimeShape& output_shape, OutputType* output_data) {

  // MicroPrintf("\n=== FullyConnected Debug ===\n");

  // // Quantization & activation parameters
  // MicroPrintf("input_offset = %d\n", params.input_offset);
  // MicroPrintf("filter_offset = %d\n", params.weights_offset);
  // MicroPrintf("output_offset = %d\n", params.output_offset);
  // MicroPrintf("output_multiplier = %d\n", params.output_multiplier);
  // MicroPrintf("output_shift = %d\n", params.output_shift);
  // MicroPrintf("act_min = %d\n", params.quantized_activation_min);
  // MicroPrintf("act_max = %d\n", params.quantized_activation_max);

  // // Shapes
  // MicroPrintf("input_shape dims = %d\n", input_shape.DimensionsCount());
  // MicroPrintf("filter_shape dims = %d\n", filter_shape.DimensionsCount());
  // MicroPrintf("bias_shape dims = %d\n", bias_shape.DimensionsCount());
  // MicroPrintf("output_shape dims = %d\n", output_shape.DimensionsCount());

  // MicroPrintf("input_shape values: ");
  // for (int i = 0; i < input_shape.DimensionsCount(); i++)
  //   MicroPrintf("%d ", input_shape.Dims(i));
  // MicroPrintf("\n");

  // MicroPrintf("filter_shape values: ");
  // for (int i = 0; i < filter_shape.DimensionsCount(); i++)
  //   MicroPrintf("%d ", filter_shape.Dims(i));
  // MicroPrintf("\n");

  // MicroPrintf("output_shape values: ");
  // for (int i = 0; i < output_shape.DimensionsCount(); i++)
  //   MicroPrintf("%d ", output_shape.Dims(i));
  // MicroPrintf("\n");

  // MicroPrintf("bias_shape flat size = %d\n", bias_shape.FlatSize());

  // // Dump input tensor
  // DumpInt8TensorAsCArray_OneLine("fc_input_data",
  //                                reinterpret_cast<const int8_t*>(input_data),
  //                                input_shape.FlatSize());

  // // Dump filter tensor
  // DumpInt8TensorAsCArray_OneLine("fc_filter_data",
  //                                reinterpret_cast<const int8_t*>(filter_data),
  //                                filter_shape.FlatSize());

  // // Dump bias tensor
  // if (bias_data)
  //   DumpInt32ArrayAsCArray_OneLine("fc_bias_data",
  //                                  reinterpret_cast<const int32_t*>(bias_data),
  //                                  bias_shape.FlatSize());

  // // Optional: dump output_multiplier & output_shift
  // MicroPrintf("output_multiplier = %d\n", params.output_multiplier);
  // MicroPrintf("output_shift = %d\n", params.output_shift);

  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      BiasType acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled =
          MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] =
          static_cast<OutputType>(acc_scaled);
    }
  }
    // Dump output tensor
  DumpInt8TensorAsCArray_OneLine("fc_output_data",
                                 reinterpret_cast<const int8_t*>(output_data),
                                 output_shape.FlatSize());

}

// This implementation receives the scales in float and performs requant in
// float to avoid loss of precision.
template <typename InputType, typename WeightType, typename OutputType,
          typename BiasType>
void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape,
                    const InputType* input_data,
                    const RuntimeShape& filter_shape,
                    const WeightType* filter_data,
                    const RuntimeShape& bias_shape, const BiasType* bias_data,
                    const RuntimeShape& output_shape, float input_scale,
                    float output_scale, float filter_scale,
                    OutputType* output_data) {
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int output_dim_count = output_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      BiasType acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      const double effective_output_scale = static_cast<double>(input_scale) *
                                            static_cast<double>(filter_scale) /
                                            static_cast<double>(output_scale);
      int32_t acc_scaled = static_cast<int32_t>(
          round(static_cast<double>(acc) * effective_output_scale));
      acc_scaled += output_offset;
      acc_scaled = std::max(acc_scaled, output_activation_min);
      acc_scaled = std::min(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] =
          static_cast<OutputType>(acc_scaled);
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_FULLY_CONNECTED_H_