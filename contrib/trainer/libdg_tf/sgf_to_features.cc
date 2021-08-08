// Copyright 2020 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/guarded_philox_random.h"

#include "boost_per_move.hh"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using Eigen::half;

struct Example {
    int index;
    int color;
    float policy[362];
    float ownership[361];
    int winner;
    int number;
    float komi;
    half lz_features[6498];
    half features[0];
};

extern "C" int get_num_features();
extern "C" int extract_single_example(
    const char* raw_sgf_content,
    Example* examples,
    int num_examples
);

/**
 * Op that returns the features / labels from a random move in the given SGF.
 */
class SgfToFeaturesOp : public OpKernel {
    public:
        explicit SgfToFeaturesOp(OpKernelConstruction* context)
        : OpKernel(context)
        {
            // pass
        }

        void Compute(OpKernelContext* context) override {
            const Tensor* sgf_tensor;
            const Tensor* num_examples_tensor;

            OP_REQUIRES_OK(context, context->input("sgf", &sgf_tensor));
            OP_REQUIRES(
                context,
                TensorShapeUtils::IsScalar(sgf_tensor->shape()),
                errors::InvalidArgument(
                    "Input `sgf` should be a scalar but received shape: ",
                    sgf_tensor->shape().DebugString()
                )
            );

            OP_REQUIRES_OK(context, context->input("num_examples", &num_examples_tensor));
            OP_REQUIRES(
                context,
                TensorShapeUtils::IsScalar(num_examples_tensor->shape()),
                errors::InvalidArgument(
                    "Input `num_examples` should be a scalar but received shape: ",
                    num_examples_tensor->shape().DebugString()
                )
            );
            OP_REQUIRES(
                context,
                num_examples_tensor->flat<int>()(0) >= 1,
                errors::InvalidArgument(
                    "Input `num_examples` must be at least 1 but received: ",
                    num_examples_tensor->flat<int>()(0)
                )
            );

            // run the internal operation
            const auto num_features = get_num_features();
            const auto features_size = 361 * num_features;
            const auto num_examples = num_examples_tensor->flat<int>()(0);
            const auto sizeof_example = sizeof(Example) + sizeof(half) * features_size;
            Example* examples = (Example*)malloc(sizeof_example * num_examples);
            const auto sgf = sgf_tensor->flat<tstring>();

            const auto status = extract_single_example(
                sgf(0).c_str(),
                examples,
                num_examples
            );

            // allocate output
            Tensor* lz_features_tensor;
            Tensor* features_tensor;
            Tensor* policy_tensor;
            Tensor* winner_tensor;
            Tensor* ownership_tensor;
            Tensor* komi_tensor;
            Tensor* boost_tensor;
            Tensor* has_ownership_tensor;

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    0,
                    TensorShape({num_examples, 19, 19, 18}),
                    &lz_features_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    1,
                    TensorShape({num_examples, 19, 19, num_features}),
                    &features_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    2,
                    TensorShape({num_examples, 362}),
                    &policy_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    3,
                    TensorShape({num_examples, 1}),
                    &winner_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    4,
                    TensorShape({num_examples, 361}),
                    &ownership_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    5,
                    TensorShape({num_examples, 1}),
                    &komi_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    6,
                    TensorShape({num_examples, 1}),
                    &boost_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    7,
                    TensorShape({num_examples, 1}),
                    &has_ownership_tensor
                )
            );

            auto lz_features = lz_features_tensor->flat<half>();
            auto features = features_tensor->flat<half>();
            auto policy = policy_tensor->flat<float>();
            auto winner = winner_tensor->flat<float>();
            auto ownership = ownership_tensor->flat<float>();
            auto komi = komi_tensor->flat<float>();
            auto boost = boost_tensor->flat<float>();
            auto has_ownership = has_ownership_tensor->flat<float>();

            if (status == 0) {
                for (auto n = 0; n < num_examples; ++n) {
                    auto example = (Example*)((char*)examples + n * sizeof_example);

                    for (auto i = 0; i < 6498; ++i) {
                        lz_features(n * 6498 + i) = example->lz_features[i];
                    }

                    // copy the features
                    for (auto i = 0; i < features_size; ++i) {
                        features(n * features_size + i) = example->features[i];
                    }

                    // copy policy and fix them up in case they are incomplete
                    float total_policy = 0.0;

                    for (auto i = 0; i < 362; ++i) {
                        policy(n * 362 + i) = example->policy[i];

                        total_policy += policy(n * 362 + i);
                    }

                    OP_REQUIRES(
                        context,
                        total_policy >= 0.0 && total_policy <= 1.0,
                        errors::InvalidArgument(
                            "Output `policy` has invalid sum: ",
                            total_policy
                        )
                    );

                    OP_REQUIRES(
                        context,
                        example->index >= 0 && example->index < 362,
                        errors::InvalidArgument(
                            "Output `policy` has invalid index: ",
                            example->index
                        )
                    );

                    policy(n * 362 + example->index) += 1.0 - total_policy;

                    // calculate the winner
                    if (example->color == example->winner) {
                        winner(n) = 1.0;
                    } else {
                        winner(n) = -1.0;
                    }

                    // set ownership
                    has_ownership(n) = 0.0;
                    for (auto i = 0; i < 361; ++i) {
                        ownership(n * 361 + i) = example->ownership[i];
                        if (example->ownership[i] != 0.0) {
                            has_ownership(n) = 1.0;
                        }
                    }

                    // set komi
                    komi(n) = example->komi;
                    if (example->color == 1) { // is black
                        komi(n) = -komi(n);
                    }

                    // set boost
                    boost(n) = boost_for_move_number(example->number);
                }
            } else {
                // zero out the labels, which we use to determine if it
                // succeeded or not.
                //
                // do not bother with the features, since they are large and
                // we do not check them anyway.
                for (auto n = 0; n < num_examples; ++n) {
                    for (auto i = 0; i < 6498; ++i) {
                        lz_features(n * 6498 + i) = Eigen::half(0.0f);
                    }

                    for (auto i = 0; i < features_size; ++i) {
                        features(n * features_size + i) = Eigen::half(0.0f);
                    }

                    for (auto i = 0; i < 362; ++i) {
                        policy(n * 362 + i) = 0;
                    }

                    for (auto i = 0; i < 361; ++i) {
                        ownership(n * 361 + i) = 0;
                    }

                    winner(n) = 0.0;
                    komi(n) = 0.0;
                    boost(n) = 0.0;
                    has_ownership(n) = 0.0;
                }
            }

            free(examples);
        }
};

REGISTER_OP("SgfToFeatures")
    .Input("sgf: string")
    .Input("num_examples: int32")
    .Output("lz_features: float16")
    .Output("features: float16")
    .Output("policy: float32")
    .Output("winner: float32")
    .Output("ownership: float32")
    .Output("komi: float32")
    .Output("boost: float32")
    .Output("has_ownership: float32")
    .SetShapeFn([](InferenceContext* c) {
        DimensionHandle n;
        TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &n));

        c->set_output(0, c->MakeShape({n, 19, 19, 18})); // lz_features
        c->set_output(1, c->MakeShape({n, 19, 19, get_num_features()})); // features
        c->set_output(2, c->MakeShape({n, 362})); // policy
        c->set_output(3, c->MakeShape({n, 1})); // value
        c->set_output(4, c->MakeShape({n, 361})); // ownership
        c->set_output(5, c->MakeShape({n, 1})); // komi
        c->set_output(6, c->MakeShape({n, 1})); // boost
        c->set_output(7, c->MakeShape({n, 1})); // has_ownership

        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("SgfToFeatures").Device(DEVICE_CPU), SgfToFeaturesOp);
