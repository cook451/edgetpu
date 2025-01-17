#include "src/cpp/tools/tflite_graph_util.h"

#include "absl/flags/parse.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/cpp/test_utils.h"

namespace coral {
namespace tools {
namespace {

// This test compares the results of model concatenation with recorded golden
// results. The test models are real production models.
void TestConcatModels(
    const std::string& graph0_path, const std::string& graph1_path,
    const std::string& expected_graph,
    const std::vector<std::string>& bypass_output_tensors = {}) {
  const std::string result_path =
      GenerateRandomFilePath("merged_model", ".tflite");
  // Remove output model generated by previous tests.
  std::remove(result_path.c_str());

  ConcatTfliteModels(graph0_path, graph1_path, result_path,
                     bypass_output_tensors);

  std::string result_str;
  ReadFileOrDie(result_path, &result_str);

  std::string expected_str;
  ReadFileOrDie(expected_graph, &expected_str);

  EXPECT_EQ(expected_str, result_str) << result_path;
}

TEST(TfliteGraphUtilTest, ConcatMobilenetClassification) {
  const std::string graph0_path = TestDataPath(
      "tools/mobilenet_quant_v1_224_feature_layers-custom_op.tflite");
  const std::string graph1_path =
      TestDataPath("tools/mobilenet_quant_v1_224_head_layers.tflite");
  const std::string expected_graph = TestDataPath(
      "tools/mobilenet_quant_v1_1.0_224_partial_delegation.tflite");
  TestConcatModels(graph0_path, graph1_path, expected_graph);
}

TEST(TfliteGraphUtilTest, ConcatMobilenetClassificationWithBypass) {
  const std::string graph0_path = TestDataPath(
      "tools/mobilenet_quant_v1_224_feature_layers-custom_op.tflite");
  const std::string graph1_path =
      TestDataPath("tools/mobilenet_quant_v1_224_head_layers.tflite");
  const std::string expected_graph = TestDataPath(
      "tools/mobilenet_quant_v1_1.0_224_partial_delegation_with_bypass.tflite");
  // Like ConcatMobilenetClassification but also bypass the AvgPool tensor
  // out to the output.
  TestConcatModels(graph0_path, graph1_path, expected_graph,
                   {"MobilenetV1/Logits/AvgPool_1a/AvgPool"});
}

TEST(TfliteGraphUtilTest, ConcatMobilenetSSD) {
  const std::string graph0_path = TestDataPath(
      "tools/ssd_mobilenet_v1_coco_quant_postprocess_base_custom_op.tflite");
  const std::string graph1_path = TestDataPath(
      "tools/ssd_mobilenet_v1_coco_quant_postprocess_head_layers.tflite");
  const std::string expected_graph = TestDataPath(
      "tools/"
      "ssd_mobilenet_v1_coco_quant_postprocess_partial_delegation.tflite");
  TestConcatModels(graph0_path, graph1_path, expected_graph);
}

}  // namespace
}  // namespace tools
}  // namespace coral

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  return RUN_ALL_TESTS();
}
