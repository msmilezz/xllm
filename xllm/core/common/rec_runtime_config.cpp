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

#include "common/rec_runtime_config.h"

#include <glog/logging.h>

#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "common/global_flags.h"

namespace xllm {
namespace {

thread_local const RecRuntimeConfig* g_scoped_rec_runtime_config = nullptr;
std::mutex g_rec_runtime_env_mutex;
std::optional<bool> g_task_queue_env_override;

}  // namespace

ScopedRecRuntimeConfig::ScopedRecRuntimeConfig(
    const RecRuntimeConfig& runtime_config)
    : previous_config_(g_scoped_rec_runtime_config) {
  g_scoped_rec_runtime_config = &runtime_config;
}

ScopedRecRuntimeConfig::~ScopedRecRuntimeConfig() {
  g_scoped_rec_runtime_config = previous_config_;
}

const RecRuntimeConfig* try_get_scoped_rec_runtime_config() {
  return g_scoped_rec_runtime_config;
}

void apply_rec_runtime_process_environment(
    const RecRuntimeConfig& runtime_config,
    std::string_view source) {
  std::lock_guard<std::mutex> lock(g_rec_runtime_env_mutex);

  const bool enable_task_queue = runtime_config.enable_task_queue;
  const char* expected_value = enable_task_queue ? "1" : "0";
  const char* current_value = std::getenv("TASK_QUEUE_ENABLE");
  if (g_task_queue_env_override.has_value() &&
      g_task_queue_env_override.value() != enable_task_queue) {
    LOG(WARNING) << "Ignore conflicting TASK_QUEUE_ENABLE request from "
                 << source << ", requested=" << expected_value << ", applied="
                 << (g_task_queue_env_override.value() ? "1" : "0");
    return;
  }

  if (current_value == nullptr ||
      std::string(current_value) != expected_value) {
    const int32_t ret = setenv("TASK_QUEUE_ENABLE", expected_value, 1);
    CHECK_EQ(ret, 0) << "Failed to set TASK_QUEUE_ENABLE=" << expected_value;
  }

  g_task_queue_env_override = enable_task_queue;
  LOG(INFO) << "Applied TASK_QUEUE_ENABLE=" << expected_value
            << " for REC runtime source=" << source;
}

bool get_rec_runtime_enable_task_queue() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->enable_task_queue : true;
}

bool get_rec_runtime_enable_prefix_cache() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->enable_prefix_cache
                                   : FLAGS_enable_prefix_cache;
}

bool get_rec_runtime_enable_schedule_overlap() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->enable_schedule_overlap
                                   : FLAGS_enable_schedule_overlap;
}

bool get_rec_runtime_enable_chunked_prefill() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->enable_chunked_prefill
                                   : FLAGS_enable_chunked_prefill;
}

bool get_rec_runtime_enable_graph() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->enable_graph
                                   : FLAGS_enable_graph;
}

bool get_rec_runtime_enable_graph_mode_decode_no_padding() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr
             ? runtime_config->enable_graph_mode_decode_no_padding
             : FLAGS_enable_graph_mode_decode_no_padding;
}

bool get_rec_runtime_enable_prefill_piecewise_graph() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr
             ? runtime_config->enable_prefill_piecewise_graph
             : FLAGS_enable_prefill_piecewise_graph;
}

bool get_rec_runtime_enable_xattention_one_stage() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->enable_xattention_one_stage
                                   : FLAGS_enable_xattention_one_stage;
}

bool get_rec_runtime_enable_prefill_only() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->enable_rec_prefill_only
                                   : FLAGS_enable_rec_prefill_only;
}

bool get_rec_runtime_enable_constrained_decoding() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->enable_constrained_decoding
                                   : FLAGS_enable_constrained_decoding;
}

bool get_rec_runtime_enable_fast_sampler() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->enable_rec_fast_sampler
                                   : FLAGS_enable_rec_fast_sampler;
}

bool get_rec_runtime_enable_topk_sorted() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->enable_topk_sorted
                                   : FLAGS_enable_topk_sorted;
}

int32_t get_rec_runtime_block_size() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->block_size
                                   : FLAGS_block_size;
}

int32_t get_rec_runtime_max_tokens_per_batch() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->max_tokens_per_batch
                                   : FLAGS_max_tokens_per_batch;
}

int32_t get_rec_runtime_max_seqs_per_batch() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->max_seqs_per_batch
                                   : FLAGS_max_seqs_per_batch;
}

int32_t get_rec_runtime_worker_max_concurrency() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->rec_worker_max_concurrency
                                   : FLAGS_rec_worker_max_concurrency;
}

int32_t get_rec_runtime_beam_width() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->beam_width
                                   : FLAGS_beam_width;
}

int32_t get_rec_runtime_max_decode_rounds() {
  const RecRuntimeConfig* runtime_config = try_get_scoped_rec_runtime_config();
  return runtime_config != nullptr ? runtime_config->max_decode_rounds
                                   : FLAGS_max_decode_rounds;
}

}  // namespace xllm
