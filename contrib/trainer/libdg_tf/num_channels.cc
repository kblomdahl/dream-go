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

#include "example.hh"

/**
 * Op that returns the number of feature channels.
 */
class NumTargetChannelsOp : public OpKernel {
    public:
        explicit NumTargetChannelsOp(OpKernelConstruction* context)
        : OpKernel(context)
        {
            // pass
        }

        void Compute(OpKernelContext* context) override {
            Example example(1, 1);

            // Allocate the output tensors
            Tensor* n_tensor = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output("n", {}, &n_tensor)
            );

            n_tensor->flat<int>()(0) = example.additional_targets_shape_[4];
        }
};

REGISTER_OP("NumTargetChannels")
    .Output("n: int32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Scalar());

        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("NumTargetChannels").Device(DEVICE_CPU), NumTargetChannelsOp);

/**
 * Op that returns the number of feature channels.
 */
class NumMotionChannelsOp : public OpKernel {
    public:
        explicit NumMotionChannelsOp(OpKernelConstruction* context)
        : OpKernel(context)
        {
            // pass
        }

        void Compute(OpKernelContext* context) override {
            Example example(1, 1);

            // Allocate the output tensors
            Tensor* n_tensor = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output("n", {}, &n_tensor)
            );

            n_tensor->flat<int>()(0) = example.motion_features_shape_[4];
        }
};

REGISTER_OP("NumMotionChannels")
    .Output("n: int32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Scalar());

        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("NumMotionChannels").Device(DEVICE_CPU), NumMotionChannelsOp);

/**
 * Op that returns the number of feature channels.
 */
class NumFeatureChannelsOp : public OpKernel {
    public:
        explicit NumFeatureChannelsOp(OpKernelConstruction* context)
        : OpKernel(context)
        {
            // pass
        }

        void Compute(OpKernelContext* context) override {
            Example example(1, 1);

            // Allocate the output tensors
            Tensor* n_tensor = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output("n", {}, &n_tensor)
            );

            n_tensor->flat<int>()(0) = example.features_shape_[4];
        }
};

REGISTER_OP("NumFeatureChannels")
    .Output("n: int32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Scalar());

        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("NumFeatureChannels").Device(DEVICE_CPU), NumFeatureChannelsOp);
