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

#include <gtest/gtest.h>

#include "common/global_flags.h"

namespace xllm {
namespace {

class ScopedFlagState final {
 public:
  ScopedFlagState()
      : enable_prefix_cache_(FLAGS_enable_prefix_cache),
        enable_schedule_overlap_(FLAGS_enable_schedule_overlap),
        enable_chunked_prefill_(FLAGS_enable_chunked_prefill),
        enable_graph_(FLAGS_enable_graph),
        enable_graph_mode_decode_no_padding_(
            FLAGS_enable_graph_mode_decode_no_padding),
        enable_prefill_piecewise_graph_(FLAGS_enable_prefill_piecewise_graph),
        enable_xattention_one_stage_(FLAGS_enable_xattention_one_stage),
        enable_rec_prefill_only_(FLAGS_enable_rec_prefill_only),
        enable_constrained_decoding_(FLAGS_enable_constrained_decoding),
        enable_rec_fast_sampler_(FLAGS_enable_rec_fast_sampler),
        enable_topk_sorted_(FLAGS_enable_topk_sorted),
        block_size_(FLAGS_block_size),
        max_tokens_per_batch_(FLAGS_max_tokens_per_batch),
        max_seqs_per_batch_(FLAGS_max_seqs_per_batch),
        rec_worker_max_concurrency_(FLAGS_rec_worker_max_concurrency),
        beam_width_(FLAGS_beam_width),
        max_decode_rounds_(FLAGS_max_decode_rounds) {}

  ~ScopedFlagState() {
    FLAGS_enable_prefix_cache = enable_prefix_cache_;
    FLAGS_enable_schedule_overlap = enable_schedule_overlap_;
    FLAGS_enable_chunked_prefill = enable_chunked_prefill_;
    FLAGS_enable_graph = enable_graph_;
    FLAGS_enable_graph_mode_decode_no_padding =
        enable_graph_mode_decode_no_padding_;
    FLAGS_enable_prefill_piecewise_graph = enable_prefill_piecewise_graph_;
    FLAGS_enable_xattention_one_stage = enable_xattention_one_stage_;
    FLAGS_enable_rec_prefill_only = enable_rec_prefill_only_;
    FLAGS_enable_constrained_decoding = enable_constrained_decoding_;
    FLAGS_enable_rec_fast_sampler = enable_rec_fast_sampler_;
    FLAGS_enable_topk_sorted = enable_topk_sorted_;
    FLAGS_block_size = block_size_;
    FLAGS_max_tokens_per_batch = max_tokens_per_batch_;
    FLAGS_max_seqs_per_batch = max_seqs_per_batch_;
    FLAGS_rec_worker_max_concurrency = rec_worker_max_concurrency_;
    FLAGS_beam_width = beam_width_;
    FLAGS_max_decode_rounds = max_decode_rounds_;
  }

 private:
  bool enable_prefix_cache_;
  bool enable_schedule_overlap_;
  bool enable_chunked_prefill_;
  bool enable_graph_;
  bool enable_graph_mode_decode_no_padding_;
  bool enable_prefill_piecewise_graph_;
  bool enable_xattention_one_stage_;
  bool enable_rec_prefill_only_;
  bool enable_constrained_decoding_;
  bool enable_rec_fast_sampler_;
  bool enable_topk_sorted_;
  int32_t block_size_;
  int32_t max_tokens_per_batch_;
  int32_t max_seqs_per_batch_;
  int32_t rec_worker_max_concurrency_;
  int32_t beam_width_;
  int32_t max_decode_rounds_;
};

}  // namespace

TEST(RecRuntimeConfigTest, FallsBackToGlobalFlagsWhenNoScope) {
  ScopedFlagState scoped_flags;

  FLAGS_enable_prefix_cache = true;
  FLAGS_enable_schedule_overlap = true;
  FLAGS_enable_chunked_prefill = true;
  FLAGS_enable_graph = true;
  FLAGS_enable_graph_mode_decode_no_padding = true;
  FLAGS_enable_prefill_piecewise_graph = true;
  FLAGS_enable_xattention_one_stage = true;
  FLAGS_enable_rec_prefill_only = true;
  FLAGS_enable_constrained_decoding = true;
  FLAGS_enable_rec_fast_sampler = true;
  FLAGS_enable_topk_sorted = true;
  FLAGS_block_size = 256;
  FLAGS_max_tokens_per_batch = 8192;
  FLAGS_max_seqs_per_batch = 32;
  FLAGS_rec_worker_max_concurrency = 3;
  FLAGS_beam_width = 64;
  FLAGS_max_decode_rounds = 5;

  EXPECT_TRUE(get_rec_runtime_enable_prefix_cache());
  EXPECT_TRUE(get_rec_runtime_enable_schedule_overlap());
  EXPECT_TRUE(get_rec_runtime_enable_chunked_prefill());
  EXPECT_TRUE(get_rec_runtime_enable_graph());
  EXPECT_TRUE(get_rec_runtime_enable_graph_mode_decode_no_padding());
  EXPECT_TRUE(get_rec_runtime_enable_prefill_piecewise_graph());
  EXPECT_TRUE(get_rec_runtime_enable_xattention_one_stage());
  EXPECT_TRUE(get_rec_runtime_enable_prefill_only());
  EXPECT_TRUE(get_rec_runtime_enable_constrained_decoding());
  EXPECT_TRUE(get_rec_runtime_enable_fast_sampler());
  EXPECT_TRUE(get_rec_runtime_enable_topk_sorted());
  EXPECT_EQ(get_rec_runtime_block_size(), 256);
  EXPECT_EQ(get_rec_runtime_max_tokens_per_batch(), 8192);
  EXPECT_EQ(get_rec_runtime_max_seqs_per_batch(), 32);
  EXPECT_EQ(get_rec_runtime_worker_max_concurrency(), 3);
  EXPECT_EQ(get_rec_runtime_beam_width(), 64);
  EXPECT_EQ(get_rec_runtime_max_decode_rounds(), 5);
}

TEST(RecRuntimeConfigTest, ScopedConfigOverridesAndRestoresPreviousState) {
  ScopedFlagState scoped_flags;

  FLAGS_enable_prefix_cache = false;
  FLAGS_enable_schedule_overlap = false;
  FLAGS_enable_chunked_prefill = false;
  FLAGS_enable_graph = false;
  FLAGS_enable_graph_mode_decode_no_padding = false;
  FLAGS_enable_prefill_piecewise_graph = false;
  FLAGS_enable_xattention_one_stage = false;
  FLAGS_enable_rec_prefill_only = false;
  FLAGS_enable_constrained_decoding = false;
  FLAGS_enable_rec_fast_sampler = false;
  FLAGS_enable_topk_sorted = false;
  FLAGS_block_size = 128;
  FLAGS_max_tokens_per_batch = 4096;
  FLAGS_max_seqs_per_batch = 4;
  FLAGS_rec_worker_max_concurrency = 1;
  FLAGS_beam_width = 8;
  FLAGS_max_decode_rounds = 1;

  RecRuntimeConfig parent_config;
  parent_config.enable_prefix_cache = true;
  parent_config.enable_schedule_overlap = true;
  parent_config.enable_chunked_prefill = true;
  parent_config.enable_graph = true;
  parent_config.enable_graph_mode_decode_no_padding = true;
  parent_config.enable_prefill_piecewise_graph = true;
  parent_config.enable_xattention_one_stage = true;
  parent_config.enable_rec_prefill_only = true;
  parent_config.enable_constrained_decoding = true;
  parent_config.enable_rec_fast_sampler = true;
  parent_config.enable_topk_sorted = true;
  parent_config.block_size = 192;
  parent_config.max_tokens_per_batch = 7000;
  parent_config.max_seqs_per_batch = 17;
  parent_config.rec_worker_max_concurrency = 2;
  parent_config.beam_width = 23;
  parent_config.max_decode_rounds = 7;

  {
    ScopedRecRuntimeConfig scope(parent_config);
    EXPECT_TRUE(get_rec_runtime_enable_prefix_cache());
    EXPECT_TRUE(get_rec_runtime_enable_schedule_overlap());
    EXPECT_TRUE(get_rec_runtime_enable_chunked_prefill());
    EXPECT_TRUE(get_rec_runtime_enable_graph());
    EXPECT_TRUE(get_rec_runtime_enable_graph_mode_decode_no_padding());
    EXPECT_TRUE(get_rec_runtime_enable_prefill_piecewise_graph());
    EXPECT_TRUE(get_rec_runtime_enable_xattention_one_stage());
    EXPECT_TRUE(get_rec_runtime_enable_prefill_only());
    EXPECT_TRUE(get_rec_runtime_enable_constrained_decoding());
    EXPECT_TRUE(get_rec_runtime_enable_fast_sampler());
    EXPECT_TRUE(get_rec_runtime_enable_topk_sorted());
    EXPECT_EQ(get_rec_runtime_block_size(), 192);
    EXPECT_EQ(get_rec_runtime_max_tokens_per_batch(), 7000);
    EXPECT_EQ(get_rec_runtime_max_seqs_per_batch(), 17);
    EXPECT_EQ(get_rec_runtime_worker_max_concurrency(), 2);
    EXPECT_EQ(get_rec_runtime_beam_width(), 23);
    EXPECT_EQ(get_rec_runtime_max_decode_rounds(), 7);

    RecRuntimeConfig child_config;
    child_config.enable_graph = false;
    child_config.enable_constrained_decoding = false;
    child_config.beam_width = 99;
    child_config.max_decode_rounds = 11;
    {
      ScopedRecRuntimeConfig child_scope(child_config);
      EXPECT_FALSE(get_rec_runtime_enable_graph());
      EXPECT_FALSE(get_rec_runtime_enable_constrained_decoding());
      EXPECT_EQ(get_rec_runtime_beam_width(), 99);
      EXPECT_EQ(get_rec_runtime_max_decode_rounds(), 11);
    }

    EXPECT_TRUE(get_rec_runtime_enable_graph());
    EXPECT_TRUE(get_rec_runtime_enable_constrained_decoding());
    EXPECT_EQ(get_rec_runtime_beam_width(), 23);
    EXPECT_EQ(get_rec_runtime_max_decode_rounds(), 7);
  }

  EXPECT_FALSE(get_rec_runtime_enable_prefix_cache());
  EXPECT_FALSE(get_rec_runtime_enable_schedule_overlap());
  EXPECT_FALSE(get_rec_runtime_enable_chunked_prefill());
  EXPECT_FALSE(get_rec_runtime_enable_graph());
  EXPECT_FALSE(get_rec_runtime_enable_graph_mode_decode_no_padding());
  EXPECT_FALSE(get_rec_runtime_enable_prefill_piecewise_graph());
  EXPECT_FALSE(get_rec_runtime_enable_xattention_one_stage());
  EXPECT_FALSE(get_rec_runtime_enable_prefill_only());
  EXPECT_FALSE(get_rec_runtime_enable_constrained_decoding());
  EXPECT_FALSE(get_rec_runtime_enable_fast_sampler());
  EXPECT_FALSE(get_rec_runtime_enable_topk_sorted());
  EXPECT_EQ(get_rec_runtime_block_size(), 128);
  EXPECT_EQ(get_rec_runtime_max_tokens_per_batch(), 4096);
  EXPECT_EQ(get_rec_runtime_max_seqs_per_batch(), 4);
  EXPECT_EQ(get_rec_runtime_worker_max_concurrency(), 1);
  EXPECT_EQ(get_rec_runtime_beam_width(), 8);
  EXPECT_EQ(get_rec_runtime_max_decode_rounds(), 1);
}

}  // namespace xllm
