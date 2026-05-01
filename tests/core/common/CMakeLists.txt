include(cc_test)

cc_test(
  NAME
    common_test
  SRCS
    rec_runtime_config_test.cpp
    rate_limiter_test.cpp
  DEPS
    common
    absl::synchronization
    absl::time
    GTest::gtest_main
    gflags::gflags
    glog::glog
)
