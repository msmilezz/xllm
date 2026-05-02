/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <string_view>

namespace xllm {

struct RecRuntimeConfig {
  bool enable_task_queue = true;
  bool enable_prefix_cache = false;
  bool enable_schedule_overlap = false;
  bool enable_chunked_prefill = false;
  bool enable_graph = false;
  bool enable_graph_mode_decode_no_padding = false;
  bool enable_prefill_piecewise_graph = false;
  bool enable_xattention_one_stage = false;
  bool enable_rec_prefill_only = false;
  bool enable_constrained_decoding = false;
  bool enable_rec_fast_sampler = false;
  bool enable_topk_sorted = false;
  int32_t block_size = 128;
  int32_t max_tokens_per_batch = 4096;
  int32_t max_seqs_per_batch = 4;
  int32_t rec_worker_max_concurrency = 1;
  int32_t beam_width = 128;
  int32_t max_decode_rounds = 0;
};

class ScopedRecRuntimeConfig final {
 public:
  explicit ScopedRecRuntimeConfig(const RecRuntimeConfig& runtime_config);
  ~ScopedRecRuntimeConfig();

  ScopedRecRuntimeConfig(const ScopedRecRuntimeConfig&) = delete;
  ScopedRecRuntimeConfig& operator=(const ScopedRecRuntimeConfig&) = delete;
  ScopedRecRuntimeConfig(ScopedRecRuntimeConfig&&) = delete;
  ScopedRecRuntimeConfig& operator=(ScopedRecRuntimeConfig&&) = delete;

 private:
  const RecRuntimeConfig* previous_config_ = nullptr;
};

const RecRuntimeConfig* try_get_scoped_rec_runtime_config();

void apply_rec_runtime_process_environment(
    const RecRuntimeConfig& runtime_config,
    std::string_view source = "unknown");

bool get_rec_runtime_enable_task_queue();
bool get_rec_runtime_enable_prefix_cache();
bool get_rec_runtime_enable_schedule_overlap();
bool get_rec_runtime_enable_chunked_prefill();
bool get_rec_runtime_enable_graph();
bool get_rec_runtime_enable_graph_mode_decode_no_padding();
bool get_rec_runtime_enable_prefill_piecewise_graph();
bool get_rec_runtime_enable_xattention_one_stage();
bool get_rec_runtime_enable_prefill_only();
bool get_rec_runtime_enable_constrained_decoding();
bool get_rec_runtime_enable_fast_sampler();
bool get_rec_runtime_enable_topk_sorted();
int32_t get_rec_runtime_block_size();
int32_t get_rec_runtime_max_tokens_per_batch();
int32_t get_rec_runtime_max_seqs_per_batch();
int32_t get_rec_runtime_worker_max_concurrency();
int32_t get_rec_runtime_beam_width();
int32_t get_rec_runtime_max_decode_rounds();

}  // namespace xllm
