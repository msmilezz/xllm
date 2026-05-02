/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "word_embedding_impl.h"

#include <algorithm>
#include <mutex>
#include <sstream>

#include "core/util/env_var.h"

namespace xllm {
namespace layer {

namespace {

constexpr int64_t kEmbeddingIdPreviewCount = 16;

bool should_debug_word_embedding_ids() {
  return util::get_bool_env("XLLM_DEBUG_WORD_EMBEDDING_IDS", false);
}

bool should_clone_word_embedding_ids() {
  return util::get_bool_env("XLLM_DEBUG_WORD_EMBEDDING_CLONE_IDS", false);
}

bool should_serialize_word_embedding() {
  return util::get_bool_env("XLLM_DEBUG_WORD_EMBEDDING_GLOBAL_MUTEX", false);
}

void maybe_log_word_embedding_id_range(const torch::Tensor& input_ids,
                                       int64_t vocab_size,
                                       const torch::Device& device) {
  if (!should_debug_word_embedding_ids()) {
    return;
  }

  torch::Tensor flattened_input_ids = input_ids.reshape({-1});
  torch::Tensor input_ids_cpu = flattened_input_ids.to(
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
  if (input_ids_cpu.numel() == 0) {
    LOG(INFO) << "word embedding ids debug, device=" << device
              << ", vocab_size=" << vocab_size << ", numel=0";
    return;
  }

  int64_t min_id = input_ids_cpu.min().item<int64_t>();
  int64_t max_id = input_ids_cpu.max().item<int64_t>();
  int64_t preview_count =
      std::min<int64_t>(input_ids_cpu.numel(), kEmbeddingIdPreviewCount);
  const int64_t* input_ids_ptr = input_ids_cpu.data_ptr<int64_t>();
  std::ostringstream preview_stream;
  for (int64_t index = 0; index < preview_count; ++index) {
    if (index > 0) {
      preview_stream << ",";
    }
    preview_stream << input_ids_ptr[index];
  }

  if (device.type() == c10::DeviceType::PrivateUse1 && device.index() == 1) {
    LOG(INFO) << "word embedding ids debug, device=" << device
              << ", vocab_size=" << vocab_size
              << ", numel=" << input_ids_cpu.numel() << ", min_id=" << min_id
              << ", max_id=" << max_id << ", preview=[" << preview_stream.str()
              << "]";
  }

  if (min_id < 0 || max_id >= vocab_size) {
    LOG(ERROR) << "word embedding ids out of range, device=" << device
               << ", vocab_size=" << vocab_size
               << ", numel=" << input_ids_cpu.numel() << ", min_id=" << min_id
               << ", max_id=" << max_id << ", preview=[" << preview_stream.str()
               << "]";
  }
}

}  // namespace

WordEmbeddingImpl::WordEmbeddingImpl(const ModelContext& context)
    : WordEmbeddingImpl(context.get_model_args().vocab_size(),
                        context.get_model_args().hidden_size(),
                        context.get_parallel_args(),
                        context.get_tensor_options()) {}

WordEmbeddingImpl::WordEmbeddingImpl(int64_t num_embeddings,
                                     int64_t embedding_dim,
                                     const ParallelArgs& parallel_args,
                                     const torch::TensorOptions& options)
    : parallel_args_(parallel_args) {
  rank_ = parallel_args_.tp_group_->rank();
  world_size_ = parallel_args_.tp_group_->world_size();

  CHECK(embedding_dim % world_size_ == 0)
      << "out_features " << embedding_dim << " not divisible by world_size "
      << world_size_;
  const int64_t embedding_dim_per_partition = embedding_dim / world_size_;

  // register the weight parameter
  weight_ = register_parameter(
      "weight",
      torch::empty({num_embeddings, embedding_dim_per_partition}, options),
      /*requires_grad=*/false);
}

// The input to the module is a list of indices, and the output is the
// corresponding word embeddings.
torch::Tensor WordEmbeddingImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  static std::mutex word_embedding_mutex;
  torch::Tensor embedding_input = input;
  if (should_clone_word_embedding_ids()) {
    embedding_input = input.contiguous().clone();
  }

  maybe_log_word_embedding_id_range(
      embedding_input, weight_.size(0), embedding_input.device());

  std::optional<std::lock_guard<std::mutex>> lock_guard;
  if (should_serialize_word_embedding()) {
    lock_guard.emplace(word_embedding_mutex);
  }
  torch::Tensor output = F::embedding(embedding_input, weight_);
  if (world_size_ > 1) {
    output = xllm::parallel_state::gather(output, parallel_args_.tp_group_);
  }
  return output;
}

// load the weight from the checkpoint
void WordEmbeddingImpl::load_state_dict(const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  LOAD_SHARDED_WEIGHT(weight, 1);
}

}  // namespace layer
}  // namespace xllm
