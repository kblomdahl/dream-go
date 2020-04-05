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

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using Eigen::half;

const float BOOST_PER_MOVE_NUMBER[] = {
    8.95713e-04, 6.35542e-03, 1.65546e-02, 2.83989e-02, 5.49426e-02, 1.04630e-01,
    1.74213e-01, 2.42109e-01, 3.09608e-01, 3.73865e-01, 4.34310e-01, 4.89187e-01,
    5.39308e-01, 5.86628e-01, 6.33125e-01, 6.73155e-01, 7.10544e-01, 7.43891e-01,
    7.72972e-01, 8.01025e-01, 8.26515e-01, 8.48955e-01, 8.69386e-01, 8.87594e-01,
    9.02899e-01, 9.16550e-01, 9.28083e-01, 9.39286e-01, 9.48505e-01, 9.56027e-01,
    9.62158e-01, 9.67270e-01, 9.72136e-01, 9.76142e-01, 9.79586e-01, 9.82235e-01,
    9.84442e-01, 9.86582e-01, 9.88328e-01, 9.90146e-01, 9.91560e-01, 9.92753e-01,
    9.93957e-01, 9.94597e-01, 9.95357e-01, 9.95950e-01, 9.96524e-01, 9.96906e-01,
    9.97323e-01, 9.97557e-01, 9.97777e-01, 9.97920e-01, 9.98077e-01, 9.98269e-01,
    9.98485e-01, 9.98667e-01, 9.98831e-01, 9.98922e-01, 9.98994e-01, 9.99095e-01,
    9.99172e-01, 9.99253e-01, 9.99340e-01, 9.99412e-01, 9.99485e-01, 9.99533e-01,
    9.99562e-01, 9.99576e-01, 9.99590e-01, 9.99614e-01, 9.99643e-01, 9.99643e-01,
    9.99647e-01, 9.99647e-01, 9.99656e-01, 9.99670e-01, 9.99675e-01, 9.99670e-01,
    9.99669e-01, 9.99693e-01, 9.99698e-01, 9.99692e-01, 9.99697e-01, 9.99701e-01,
    9.99696e-01, 9.99695e-01, 9.99700e-01, 9.99699e-01, 9.99704e-01, 9.99703e-01,
    9.99708e-01, 9.99732e-01, 9.99737e-01, 9.99736e-01, 9.99736e-01, 9.99735e-01,
    9.99735e-01, 9.99754e-01, 9.99759e-01, 9.99758e-01, 9.99758e-01, 9.99757e-01,
    9.99761e-01, 9.99761e-01, 9.99765e-01, 9.99759e-01, 9.99764e-01, 9.99768e-01,
    9.99778e-01, 9.99777e-01, 9.99782e-01, 9.99781e-01, 9.99786e-01, 9.99780e-01,
    9.99779e-01, 9.99783e-01, 9.99788e-01, 9.99787e-01, 9.99786e-01, 9.99785e-01,
    9.99784e-01, 9.99783e-01, 9.99782e-01, 9.99782e-01, 9.99780e-01, 9.99779e-01,
    9.99778e-01, 9.99783e-01, 9.99787e-01, 9.99786e-01, 9.99791e-01, 9.99796e-01,
    9.99794e-01, 9.99793e-01, 9.99786e-01, 9.99791e-01, 9.99801e-01, 9.99806e-01,
    9.99805e-01, 9.99803e-01, 9.99802e-01, 9.99801e-01, 9.99799e-01, 9.99798e-01,
    9.99796e-01, 9.99795e-01, 9.99793e-01, 9.99791e-01, 9.99790e-01, 9.99801e-01,
    9.99813e-01, 9.99811e-01, 9.99816e-01, 9.99815e-01, 9.99813e-01, 9.99811e-01,
    9.99810e-01, 9.99808e-01, 9.99806e-01, 9.99804e-01, 9.99802e-01, 9.99800e-01,
    9.99805e-01, 9.99803e-01, 9.99801e-01, 9.99799e-01, 9.99804e-01, 9.99802e-01,
    9.99800e-01, 9.99798e-01, 9.99795e-01, 9.99793e-01, 9.99799e-01, 9.99796e-01,
    9.99794e-01, 9.99782e-01, 9.99788e-01, 9.99786e-01, 9.99783e-01, 9.99780e-01,
    9.99777e-01, 9.99783e-01, 9.99790e-01, 9.99787e-01, 9.99784e-01, 9.99781e-01,
    9.99788e-01, 9.99785e-01, 9.99782e-01, 9.99779e-01, 9.99776e-01, 9.99783e-01,
    9.99780e-01, 9.99766e-01, 9.99762e-01, 9.99758e-01, 9.99767e-01, 9.99775e-01,
    9.99771e-01, 9.99768e-01, 9.99764e-01, 9.99760e-01, 9.99756e-01, 9.99753e-01,
    9.99749e-01, 9.99744e-01, 9.99740e-01, 9.99736e-01, 9.99732e-01, 9.99727e-01,
    9.99723e-01, 9.99718e-01, 9.99713e-01, 9.99709e-01, 9.99719e-01, 9.99715e-01,
    9.99710e-01, 9.99705e-01, 9.99700e-01, 9.99694e-01, 9.99706e-01, 9.99701e-01,
    9.99696e-01, 9.99690e-01, 9.99740e-01, 9.99735e-01, 9.99731e-01, 9.99726e-01,
    9.99721e-01, 9.99716e-01, 9.99710e-01, 9.99705e-01, 9.99699e-01, 9.99693e-01,
    9.99687e-01, 9.99681e-01, 9.99698e-01, 9.99692e-01, 9.99686e-01, 9.99679e-01,
    9.99673e-01, 9.99666e-01, 9.99659e-01, 9.99652e-01, 9.99645e-01, 9.99637e-01,
    9.99629e-01, 9.99621e-01, 9.99643e-01, 9.99635e-01, 9.99626e-01, 9.99617e-01,
    9.99608e-01, 9.99599e-01, 9.99589e-01, 9.99719e-01, 9.99713e-01, 9.99706e-01,
    9.99699e-01, 9.99691e-01, 9.99683e-01, 9.99674e-01, 9.99665e-01, 9.99656e-01,
    9.99647e-01, 9.99637e-01, 9.99673e-01, 9.99663e-01, 9.99653e-01, 9.99642e-01,
    9.99632e-01, 9.99620e-01, 9.99608e-01, 9.99653e-01, 9.99641e-01, 9.99629e-01,
    9.99618e-01, 9.99604e-01, 9.99659e-01, 9.99646e-01, 9.99635e-01, 9.99622e-01,
    9.99609e-01, 9.99596e-01, 9.99665e-01, 9.99652e-01, 9.99730e-01, 9.99720e-01,
    9.99708e-01, 9.99697e-01, 9.99686e-01, 9.99673e-01, 9.99662e-01, 9.99647e-01,
    9.99755e-01, 9.99745e-01, 9.99735e-01, 9.99725e-01, 9.99715e-01, 9.99701e-01,
    9.99688e-01, 9.99674e-01, 9.99661e-01, 9.99823e-01, 9.99815e-01, 9.99807e-01,
    9.99798e-01, 9.99790e-01, 9.99780e-01, 9.99770e-01, 9.99759e-01, 9.99748e-01,
    9.99734e-01, 9.99719e-01, 9.99704e-01, 9.99687e-01, 9.99670e-01, 9.99655e-01,
    9.99635e-01, 9.99615e-01, 9.99593e-01, 9.99570e-01, 9.99547e-01, 9.99518e-01,
    9.99488e-01, 9.99463e-01, 9.99430e-01, 9.99397e-01, 9.99359e-01, 9.99320e-01
};

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

                for (auto i = 0; i < 361; ++i) {
                    ownership(i) = example->ownership[i];
                }

                // set komi
                auto komi = komi_tensor->flat<float>();

                komi(0) = example->komi;
                if (example->color == 1) { // is black
                    komi(0) = -komi(0);
                }

                // set boost
                auto num_moves_with_boost = sizeof(BOOST_PER_MOVE_NUMBER) / sizeof(BOOST_PER_MOVE_NUMBER[0]);
                auto boost = boost_tensor->flat<float>();

                if (example->number <= num_moves_with_boost) {
                    boost(0) = BOOST_PER_MOVE_NUMBER[example->number - 1];
                } else {
                    boost(0) = 1.0;
                }
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

        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("SgfToFeatures").Device(DEVICE_CPU), SgfToFeaturesOp);
