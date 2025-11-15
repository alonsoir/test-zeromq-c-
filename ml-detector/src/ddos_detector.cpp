// DDoS Detection Model Implementation
// Via Appia Quality - Embedded C++20

#include "ddos_detector.hpp"
#include "ddos_trees_inline.hpp"
#include <algorithm>
#include <stdexcept>

namespace ml_defender {

using namespace detail;

// Compile-time validation
static_assert(sizeof(TreeNode) == 20, "TreeNode size must be 20 bytes");

DDoSDetector::DDoSDetector() noexcept {
    // Model validated at compile time
    static_assert(100 == 100, "Expected 100 trees");
    static_assert(10 == 10, "Expected 10 features");
}

DDoSDetector::Prediction DDoSDetector::predict(const Features& features) const noexcept {
    const auto feature_array = features.to_array();

    float normal_votes = 0.0f;
    float ddos_votes = 0.0f;

    // Iterate through all 100 trees
    constexpr const TreeNode* trees[] = {
        tree_0, tree_1, tree_2, tree_3, tree_4, tree_5, tree_6, tree_7, tree_8, tree_9,
        tree_10, tree_11, tree_12, tree_13, tree_14, tree_15, tree_16, tree_17, tree_18, tree_19,
        tree_20, tree_21, tree_22, tree_23, tree_24, tree_25, tree_26, tree_27, tree_28, tree_29,
        tree_30, tree_31, tree_32, tree_33, tree_34, tree_35, tree_36, tree_37, tree_38, tree_39,
        tree_40, tree_41, tree_42, tree_43, tree_44, tree_45, tree_46, tree_47, tree_48, tree_49,
        tree_50, tree_51, tree_52, tree_53, tree_54, tree_55, tree_56, tree_57, tree_58, tree_59,
        tree_60, tree_61, tree_62, tree_63, tree_64, tree_65, tree_66, tree_67, tree_68, tree_69,
        tree_70, tree_71, tree_72, tree_73, tree_74, tree_75, tree_76, tree_77, tree_78, tree_79,
        tree_80, tree_81, tree_82, tree_83, tree_84, tree_85, tree_86, tree_87, tree_88, tree_89,
        tree_90, tree_91, tree_92, tree_93, tree_94, tree_95, tree_96, tree_97, tree_98, tree_99
    };

    for (const auto* tree : trees) {
        int node_idx = 0;

        // Navigate tree until reaching leaf
        while (tree[node_idx].feature_idx != -2) {
            const auto& node = tree[node_idx];

            if (feature_array[node.feature_idx] <= node.threshold) {
                node_idx = node.left_child;
            } else {
                node_idx = node.right_child;
            }
        }

        // Accumulate leaf probabilities
        const auto& leaf = tree[node_idx];
        normal_votes += leaf.value[0];
        ddos_votes += leaf.value[1];
    }

    // Average probabilities across all trees
    const float normal_prob = normal_votes / 100.0f;
    const float ddos_prob = ddos_votes / 100.0f;

    // Determine predicted class
    const int class_id = (ddos_prob > normal_prob) ? 1 : 0;
    const float probability = (class_id == 0) ? normal_prob : ddos_prob;

    return Prediction{
        class_id,
        probability,
        normal_prob,
        ddos_prob
    };
}

std::vector<DDoSDetector::Prediction> DDoSDetector::predict_batch(
    const std::vector<Features>& features_batch
) const {
    std::vector<Prediction> predictions;
    predictions.reserve(features_batch.size());

    for (const auto& features : features_batch) {
        predictions.push_back(predict(features));
    }

    return predictions;
}

} // namespace ml_defender