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
    int index_next;
    int color;
    float policy[362];
    float policy_next[362];
    float ownership[361];
    int winner;
    int number;
    float komi;
    half features[0];
};

extern "C" int get_num_features();
extern "C" int extract_single_example(
    const char* raw_sgf_content,
    Example* example
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

            OP_REQUIRES_OK(context, context->input("sgf", &sgf_tensor));
            OP_REQUIRES(
                context,
                TensorShapeUtils::IsScalar(sgf_tensor->shape()),
                errors::InvalidArgument(
                    "Input `sgf` should be a scalar but received shape: ",
                    sgf_tensor->shape().DebugString()
                )
            );

            // run the internal operation
            const auto num_features = get_num_features();
            const auto features_size = 361 * num_features;
            Example* example = (Example*)malloc(sizeof(Example) + sizeof(half) * features_size);
            const auto sgf = sgf_tensor->flat<string>();

            const auto status = extract_single_example(
                sgf(0).c_str(),
                example
            );

            // allocate output
            Tensor* features_tensor;
            Tensor* policy_tensor;
            Tensor* policy_next_tensor;
            Tensor* winner_tensor;
            Tensor* ownership_tensor;
            Tensor* komi_tensor;
            Tensor* boost_tensor;
            Tensor* has_ownership_tensor;

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    0,
                    TensorShape({19, 19, num_features}),
                    &features_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    1,
                    TensorShape({362}),
                    &policy_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    2,
                    TensorShape({362}),
                    &policy_next_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    3,
                    TensorShape({1}),
                    &winner_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    4,
                    TensorShape({361}),
                    &ownership_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    5,
                    TensorShape({1}),
                    &komi_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    6,
                    TensorShape({1}),
                    &boost_tensor
                )
            );

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    7,
                    TensorShape({1}),
                    &has_ownership_tensor
                )
            );

            if (status == 0) {
                // copy the features
                auto features = features_tensor->flat<half>();

                for (auto i = 0; i < features_size; ++i) {
                    features(i) = example->features[i];
                }

                // copy policy and fix them up in case they are incomplete
                auto policy = policy_tensor->flat<float>();
                auto policy_next = policy_next_tensor->flat<float>();
                float total_policy = 0.0;
                float total_policy_next = 0.0;

                for (auto i = 0; i < 362; ++i) {
                    policy(i) = example->policy[i];
                    policy_next(i) = example->policy_next[i];

                    total_policy += policy(i);
                    total_policy_next += policy_next(i);
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

                OP_REQUIRES(
                    context,
                    total_policy_next >= 0.0 && total_policy_next <= 1.0,
                    errors::InvalidArgument(
                        "Output `policy_next` has invalid sum: ",
                        total_policy_next
                    )
                );

                OP_REQUIRES(
                    context,
                    example->index_next >= 0 && example->index_next < 362,
                    errors::InvalidArgument(
                        "Output `policy_next` has invalid index: ",
                        example->index_next
                    )
                );

                policy(example->index) += 1.0 - total_policy;
                policy_next(example->index_next) += 1.0 - total_policy_next;

                // calculate the winner
                auto winner = winner_tensor->flat<float>();

                if (example->color == example->winner) {
                    winner(0) = 1.0;
                } else {
                    winner(0) = -1.0;
                }

                // set ownership
                auto ownership = ownership_tensor->flat<float>();
                auto has_ownership = has_ownership_tensor->flat<float>();

                has_ownership(0) = 0.0;
                for (auto i = 0; i < 361; ++i) {
                    ownership(i) = example->ownership[i];
                    if (example->ownership[i] != 0.0) {
                        has_ownership(0) = 1.0;
                    }
                }

                // set komi
                auto komi = komi_tensor->flat<float>();

                komi(0) = example->komi;
                if (example->color == 1) { // is black
                    komi(0) = -komi(0);
                }

                // set boost
                auto boost = boost_tensor->flat<float>();

                boost(0) = boost_for_move_number(example->number);
            } else {
                // zero out the labels, which we use to determine if it
                // succeeded or not.
                //
                // do not bother with the features, since they are large and
                // we do not check them anyway.
                auto policy = policy_tensor->flat<float>();
                auto policy_next = policy_next_tensor->flat<float>();
                auto winner = winner_tensor->flat<float>();
                auto ownership = ownership_tensor->flat<float>();
                auto komi = komi_tensor->flat<float>();
                auto boost = boost_tensor->flat<float>();
                auto has_ownership = has_ownership_tensor->flat<float>();

                for (auto i = 0; i < 362; ++i) {
                    policy(i) = 0.0;
                    policy_next(i) = 0.0;
                }

                for (auto i = 0; i < 361; ++i) {
                    ownership(i) = 0.0;
                }

                winner(0) = 0.0;
                komi(0) = 0.0;
                boost(0) = 0.0;
                has_ownership(0) = 0.0;
            }

            free(example);
        }
};

REGISTER_OP("SgfToFeatures")
    .Input("sgf: string")
    .Output("features: float16")
    .Output("policy: float32")
    .Output("policy_next: float32")
    .Output("winner: float32")
    .Output("ownership: float32")
    .Output("komi: float32")
    .Output("boost: float32")
    .Output("has_ownership: float32")
    .SetShapeFn([](InferenceContext* c) {
        std::vector<DimensionHandle> dims;
        dims.emplace_back(c->MakeDim(19));
        dims.emplace_back(c->MakeDim(19));
        dims.emplace_back(c->MakeDim(get_num_features()));

        c->set_output(0, c->MakeShape(dims)); // features
        c->set_output(1, c->MakeShape({362})); // policy
        c->set_output(2, c->MakeShape({362})); // policy_next
        c->set_output(3, c->MakeShape({1})); // value
        c->set_output(4, c->MakeShape({361})); // ownership
        c->set_output(5, c->MakeShape({1})); // komi
        c->set_output(6, c->MakeShape({1})); // boost
        c->set_output(7, c->MakeShape({1})); // has_ownership

        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("SgfToFeatures").Device(DEVICE_CPU), SgfToFeaturesOp);
