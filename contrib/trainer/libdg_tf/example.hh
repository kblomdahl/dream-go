// Copyright 2022 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _EXAMPLE_HH
#define _EXAMPLE_HH
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using Eigen::half;

struct Example;

// see `candidate.rs`
extern "C" int set_example_shape(
    Example* example,
    int num_examples,
    int num_unrolls
);

// see `parse_sgf_example.rs`
extern "C" int parse_sgf_example(
    Example* example,
    const char* raw_sgf_content,
    size_t raw_sgf_content_length
);

// see `parse_sgf_example.rs`
struct Example {
    half* features_;
    int features_shape_[5];

    half* motion_features_;
    int motion_features_shape_[5];

    half* lz_features_;
    int lz_features_shape_[5];

    float* additional_targets_;
    float* additional_targets_mask_;
    int additional_targets_shape_[5];

    float* value_;
    int value_shape_[3];

    float* policy_;
    int policy_shape_[3];

    Example(int num_examples, int num_unrolls) {
        set_example_shape(this, num_examples, num_unrolls);
    }

    TensorShape MakeFeaturesShape() {
        return TensorShape({
            features_shape_[0],
            features_shape_[1],
            features_shape_[2],
            features_shape_[3],
            features_shape_[4]
        });
    }

    TensorShape MakeMotionFeaturesShape() {
        return TensorShape({
            motion_features_shape_[0],
            motion_features_shape_[1],
            motion_features_shape_[2],
            motion_features_shape_[3],
            motion_features_shape_[4]
        });
    }

    TensorShape MakeLzFeaturesShape() {
        return TensorShape({
            lz_features_shape_[0],
            lz_features_shape_[1],
            lz_features_shape_[2],
            lz_features_shape_[3],
            lz_features_shape_[4]
        });
    }

    TensorShape MakeTargetsShape() {
        return TensorShape({
            additional_targets_shape_[0],
            additional_targets_shape_[1],
            additional_targets_shape_[2],
            additional_targets_shape_[3],
            additional_targets_shape_[4]
        });
    }

    TensorShape MakeTargetsMaskShape() {
        return TensorShape({
            additional_targets_shape_[0],
            additional_targets_shape_[1],
            additional_targets_shape_[4]
        });
    }

    TensorShape MakeValueShape() {
        return TensorShape({
            value_shape_[0],
            value_shape_[1],
            value_shape_[2]
        });
    }

    TensorShape MakePolicyShape() {
        return TensorShape({
            policy_shape_[0],
            policy_shape_[1],
            policy_shape_[2]
        });
    }
};

#endif
