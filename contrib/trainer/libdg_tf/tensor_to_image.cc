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

#define IMAGE_SHAPE {1, 115, 115, 3}
#define IMAGE_STRIDE 115

#define SPRITE_WIDTH 6
#define SPRITE_HEIGHT 6

/*
 * ```
 * 
 *  xxx
 * x   x
 * x   x
 * x   x
 *  xxx
 * ```
 */
static const float WHITE_STONE[] = {
    // row 0
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,

    // row 1
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0,

    // row 2
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,

    // row 3
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,

    // row 4
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,

    // row 5
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0,
};

/*
 * ```
 * 
 *  xxx
 * xxxxx
 * xxxxx
 * xxxxx
 *  xxx
 * ```
 */
static const float BLACK_STONE[] = {
    // row 0
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,

    // row 1
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0,

    // row 2
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,

    // row 3
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,

    // row 4
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,

    // row 5
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    1.0, 1.0, 1.0,
};

class TensorToHeatImageOp : public OpKernel {
    public:
        explicit TensorToHeatImageOp(OpKernelConstruction* context)
        : OpKernel(context)
        {
            // pass
        }

        void Compute(OpKernelContext* context) override {
            const Tensor* player_tensor;
            const Tensor* opponent_tensor;
            const Tensor* heat_tensor;

            OP_REQUIRES_OK(context, context->input("player", &player_tensor));
            OP_REQUIRES(
                context,
                player_tensor->shape() == TensorShape({19, 19}),
                errors::InvalidArgument(
                    "Input `player` should be shape [19, 19] but received shape: ",
                    player_tensor->shape().DebugString()
                )
            );

            OP_REQUIRES_OK(context, context->input("opponent", &opponent_tensor));
            OP_REQUIRES(
                context,
                opponent_tensor->shape() == TensorShape({19, 19}),
                errors::InvalidArgument(
                    "Input `opponent` should be shape [19, 19] but received shape: ",
                    opponent_tensor->shape().DebugString()
                )
            );
 
            OP_REQUIRES_OK(context, context->input("heat", &heat_tensor));
            OP_REQUIRES(
                context,
                heat_tensor->shape() == TensorShape({19, 19}),
                errors::InvalidArgument(
                    "Input `heat` should be shape [19, 19] but received shape: ",
                    heat_tensor->shape().DebugString()
                )
            );

            // allocate the output tensor
            Tensor* image_tensor;

            OP_REQUIRES_OK(
                context,
                context->allocate_output(
                    0,
                    TensorShape(IMAGE_SHAPE),
                    &image_tensor
                )
            );

            // fill in the current board state
            auto player = player_tensor->flat<half>();
            auto opponent = opponent_tensor->flat<half>();
            auto heat = heat_tensor->flat<float>();
            auto image = image_tensor->flat<float>();
            float max_heat = -1.0;
            float min_heat = 1.0;

            for (auto i = 0; i < image.size(); ++i) {
                image(i) = 1.0;
            }

            for (auto i = 0; i < 361; ++i) {
                if (heat(i) > max_heat) {
                    max_heat = heat(i);
                }

                if (heat(i) < min_heat) {
                    min_heat = heat(i);
                }
            }

            for (auto y = 0; y < 19; ++y) {
                for (auto x = 0; x < 19; ++x) {
                    const auto i = 19 * y + x;

                    if (static_cast<float>(player(i)) > 1e-4f) {
                        this->CopySpriteTo(image, SPRITE_WIDTH * x, SPRITE_HEIGHT * y, BLACK_STONE);
                    } else if (static_cast<float>(opponent(i)) > 1e-4f) {
                        this->CopySpriteTo(image, SPRITE_WIDTH * x, SPRITE_HEIGHT * y, WHITE_STONE);
                    }

                    this->Heat(image, SPRITE_WIDTH * x, SPRITE_HEIGHT * y, heat(i), min_heat, max_heat);
                }
            }
        }

    private:
        void CopySpriteTo(
            Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& dst,
            int x,
            int y,
            const float* sprite
        )
        {
            for (auto dy = 0; dy < SPRITE_HEIGHT; ++dy) {
                for (auto dx = 0; dx < SPRITE_WIDTH; ++dx) {
                    for (auto j = 0; j < 3; ++j) {
                        auto dst_index = 3 * (IMAGE_STRIDE * (y + dy) + (x + dx)) + j;
                        auto src_index = 3 * (SPRITE_WIDTH * dy + dx) + j;

                        dst(dst_index) = sprite[src_index];
                    }
                }
            }
        }

        void Heat(
            Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long int>, 16, Eigen::MakePointer>& dst,
            int x,
            int y,
            float heat,
            float min_heat,
            float max_heat
        )
        {
            for (auto dy = 1; dy < SPRITE_HEIGHT; ++dy) {
                for (auto dx = 1; dx < SPRITE_WIDTH; ++dx) {
                    auto r_index = 3 * (IMAGE_STRIDE * (y + dy) + (x + dx)) + 0;
                    auto g_index = 3 * (IMAGE_STRIDE * (y + dy) + (x + dx)) + 1;
                    auto b_index = 3 * (IMAGE_STRIDE * (y + dy) + (x + dx)) + 2;

                    if (heat < -3e-3) {
                        const float alpha = 0.9f - 0.7f * heat / min_heat;

                        dst(r_index) = alpha * dst(r_index) + (1.0f - alpha) * 1.0f;
                        dst(g_index) = alpha * dst(g_index) + (1.0f - alpha) * 0.0f;
                        dst(b_index) = alpha * dst(b_index) + (1.0f - alpha) * 0.0f;
                    } else if (heat > 3e-3) {
                        const float alpha = 0.9f - 0.7f * heat / max_heat;

                        dst(r_index) = alpha * dst(r_index) + (1.0f - alpha) * 0.0f;
                        dst(g_index) = alpha * dst(g_index) + (1.0f - alpha) * 1.0f;
                        dst(b_index) = alpha * dst(b_index) + (1.0f - alpha) * 0.0f;
                    }
                }
            }
        }
};

REGISTER_OP("TensorToHeatImage")
    .Input("player: float16")
    .Input("opponent: float16")
    .Input("heat: float32")
    .Output("image: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->MakeShape(IMAGE_SHAPE));

        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("TensorToHeatImage").Device(DEVICE_CPU), TensorToHeatImageOp);
