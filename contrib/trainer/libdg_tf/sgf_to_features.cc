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

#include "example.hh"

/**
 * Op that returns the features / labels from a random move in the given SGF.
 */
class SgfToFeaturesOp : public OpKernel {
    public:
        explicit SgfToFeaturesOp(OpKernelConstruction* context)
        : OpKernel(context)
        {
            OP_REQUIRES_OK(context, context->GetAttr("num_examples", &num_examples_));
            OP_REQUIRES(
                context,
                num_examples_ >= 1,
                errors::InvalidArgument(
                    "Input `num_examples` must be at least 1 but received: ",
                    num_examples_
                )
            );

            OP_REQUIRES_OK(context, context->GetAttr("num_unrolls", &num_unrolls_));
            OP_REQUIRES(
                context,
                num_unrolls_ >= 1,
                errors::InvalidArgument(
                    "Input `num_unrolls` must be at least 1 but received: ",
                    num_unrolls_
                )
            );
        }

        void Compute(OpKernelContext* context) override {
            const Tensor* sgf_tensor = nullptr;

            OP_REQUIRES_OK(context, context->input("sgf", &sgf_tensor));
            OP_REQUIRES(
                context,
                TensorShapeUtils::IsScalar(sgf_tensor->shape()),
                errors::InvalidArgument(
                    "Input `sgf` should be a scalar but received shape: ",
                    sgf_tensor->shape().DebugString()
                )
            );

            Example example(num_examples_, num_unrolls_);

            // Allocate the output tensors
            Tensor* features_tensor = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output("features", example.MakeFeaturesShape(), &features_tensor)
            );

            Tensor* motion_features_tensor = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output("motion_features", example.MakeMotionFeaturesShape(), &motion_features_tensor)
            );

            Tensor* lz_features_tensor = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output("lz_features", example.MakeLzFeaturesShape(), &lz_features_tensor)
            );

            Tensor* targets_tensor = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output("targets", example.MakeTargetsShape(), &targets_tensor)
            );

            Tensor* targets_mask_tensor = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output("targets_mask", example.MakeTargetsMaskShape(), &targets_mask_tensor)
            );

            Tensor* value_tensor = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output("value", example.MakeValueShape(), &value_tensor)
            );

            Tensor* policy_tensor = nullptr;
            OP_REQUIRES_OK(
                context,
                context->allocate_output("policy", example.MakePolicyShape(), &policy_tensor)
            );

            // assign dmi pointers in example
            example.features_ = (half*)features_tensor->data();
            example.motion_features_ = (half*)motion_features_tensor->data();
            example.lz_features_ = (half*)lz_features_tensor->data();
            example.additional_targets_ = (float*)targets_tensor->data();
            example.additional_targets_mask_ = (float*)targets_mask_tensor->data();
            example.value_ = (float*)value_tensor->data();
            example.policy_ = (float*)policy_tensor->data();

            // xxx
            const auto sgf = sgf_tensor->flat<tstring>()(0);
            const auto status = parse_sgf_example(
                &example,
                sgf.c_str(),
                sgf.length()
            );

            if (status != 0) {  // something went wrong
                memset(policy_tensor->data(), 0, policy_tensor->TotalBytes());
            }
        }

    private:
        int num_examples_;
        int num_unrolls_;
};

Status set_output_tensor_shape(InferenceContext* c, int idx, TensorShape tensor_shape) {
    ShapeHandle shape;

    TF_RETURN_IF_ERROR(c->MakeShapeFromTensorShape(tensor_shape, &shape));
    c->set_output(idx, shape);
    return Status::OK();
}

REGISTER_OP("SgfToFeatures")
    .Input("sgf: string")
    .Attr("num_examples: int")
    .Attr("num_unrolls: int")
    .Output("features: float16")
    .Output("motion_features: float16")
    .Output("lz_features: float16")
    .Output("targets: float32")
    .Output("targets_mask: float32")
    .Output("policy: float32")
    .Output("value: float32")
    .SetShapeFn([](InferenceContext* c) {
        int num_unrolls = 0;
        TF_RETURN_IF_ERROR(c->GetAttr("num_unrolls", &num_unrolls));

        int num_examples = 0;
        TF_RETURN_IF_ERROR(c->GetAttr("num_examples", &num_examples));

        Example example(num_examples, num_unrolls);
        TF_RETURN_IF_ERROR(set_output_tensor_shape(c, 0, example.MakeFeaturesShape())); // features
        TF_RETURN_IF_ERROR(set_output_tensor_shape(c, 1, example.MakeMotionFeaturesShape())); // motion_features
        TF_RETURN_IF_ERROR(set_output_tensor_shape(c, 2, example.MakeLzFeaturesShape())); // lz_features
        TF_RETURN_IF_ERROR(set_output_tensor_shape(c, 3, example.MakeTargetsShape())); // targets
        TF_RETURN_IF_ERROR(set_output_tensor_shape(c, 4, example.MakeTargetsMaskShape())); // targets_mask
        TF_RETURN_IF_ERROR(set_output_tensor_shape(c, 5, example.MakePolicyShape())); // policy
        TF_RETURN_IF_ERROR(set_output_tensor_shape(c, 6, example.MakeValueShape())); // value

        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("SgfToFeatures").Device(DEVICE_CPU), SgfToFeaturesOp);
