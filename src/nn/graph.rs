// Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use libc::c_void;
use std::collections::HashMap;
use std::ptr;

use nn::ffi::cublas;
use nn::ffi::cuda;
use nn::ffi::cudnn;
use nn::ops::*;
use nn::slots::*;
use nn::{Type, TYPE};

/// The number of features in the neural network architecture.
const NUM_FEATURES: usize = 128;

/// The number of residual blocks in the neural network architecture.
const NUM_LAYERS: usize = 9;

pub trait Graph {
    /// Returns a pointer to the memory on the GPU that should contain the
    /// input values.
    /// 
    /// # Arguments
    /// 
    /// * `size_in_bytes` - the minimum required size of the allocated area
    /// 
    fn get_input(&mut self, size_in_bytes: Option<usize>) -> SlotGuard;

    /// Returns a pointer to a named additional variable. If two variables
    /// shares the same name, then their pointers may alias.
    /// 
    /// # Arguments
    /// 
    /// * `name` - the name of the variable
    /// * `size_in_bytes` - the minimum required size of the allocated area
    fn get_slot(&mut self, name: &'static str, size_in_bytes: usize) -> SlotGuard;

    /// Returns the batch size of this graph.
    fn get_batch_size(&self) -> usize;

    /// Returns the size (in bytes) of the maximum workspace needed.
    fn get_workspace_size(&self) -> usize;
}

pub trait Ops<G: Graph> {
    /// `output = C(weights, input) + offset`
    fn convolution(
        graph: &mut G,
        input: String,
        input_data: *const c_void,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
        weights: String,
        offset: String,
        output: String,
        output_data: *mut c_void,
        workspace: *mut c_void,
        workspace_size: usize
    );

    /// `output_2 = input + C(weights_2, C(weights_1, input) + offset_1) + offset_2`
    fn residual_block(
        graph: &mut G,
        input: String,
        input_data: *const c_void,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
        weights_1: String,
        weights_2: String,
        offset_1: String,
        offset_2: String,
        output_1: String,
        output_data_1: *mut c_void,
        output_2: String,
        output_data_2: *mut c_void,
        workspace: *mut c_void,
        workspace_size: usize
    );

    /// `output = weights * input + offset`
    fn linear(
        graph: &mut G,
        input: String,
        input_data: *const c_void,
        k: i32,
        c: i32,
        weights: String,
        offset: String,
        output: String,
        output_data: *mut c_void,
        workspace: *mut c_void,
        workspace_size: usize
    );

    /// `output = softmax(input)`
    fn softmax(
        graph: &mut G,
        input: String,
        input_data: *const c_void,
        output: String,
        output_data: *mut c_void
    );

    /// `output = relu(input)`
    fn relu(
        graph: &mut G,
        input: String,
        input_data: *const c_void,
        output: String,
        output_data: *mut c_void
    );

    /// `output = tanh(input)`
    fn tanh(
        graph: &mut G,
        input: String,
        input_data: *const c_void,
        output: String,
        output_data: *mut c_void
    );

    /// Returns the additional variable with the given name, if these operations
    /// requires additional slots. If these operations does not require the use
    /// of additional variables then it should return `null` and do nothing.
    /// 
    /// # Arguments
    /// 
    /// * `graph` - the graph to retrieve the slot from
    /// * `name` - the name of the variable
    /// * `size_in_bytes` - the minimum required size for the given slot
    /// 
    fn get_slot(graph: &mut G, name: &'static str, size_in_bytes: usize) -> SlotGuard;
}

pub struct Calibrate;

impl Ops<Builder> for Calibrate {
    fn convolution(
        graph: &mut Builder,
        input: String,
        _input_data: *const c_void,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
        weights: String,
        offset: String,
        output: String,
        _output_data: *mut c_void,
        _workspace: *mut c_void,
        _workspace_size: usize
    )
    {
        Convolution::calibrate(
            &graph.tensors[&input],
            k, c, h, w,
            &graph.tensors[&weights],
            &graph.tensors[&offset],
            &graph.tensors[&output]
        );
    }

    fn residual_block(
        graph: &mut Builder,
        input: String,
        _input_data: *const c_void,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
        weights_1: String,
        weights_2: String,
        offset_1: String,
        offset_2: String,
        output_1: String,
        _output_data_1: *mut c_void,
        output_2: String,
        _output_data_2: *mut c_void,
        _workspace: *mut c_void,
        _workspace_size: usize
    )
    {
        Convolution::calibrate(
            &graph.tensors[&input],
            k, c, h, w,
            &graph.tensors[&weights_1],
            &graph.tensors[&offset_1],
            &graph.tensors[&output_1]
        );

        Convolution::calibrate(
            &graph.tensors[&output_1],
            k, c, h, w,
            &graph.tensors[&weights_2],
            &graph.tensors[&offset_2],
            &graph.tensors[&output_2]
        );
    }

    fn linear(
        graph: &mut Builder,
        input: String,
        _input_data: *const c_void,
        k: i32,
        c: i32,
        weights: String,
        offset: String,
        output: String,
        _output_data: *mut c_void,
        _workspace: *mut c_void,
        _workspace_size: usize
    )
    {
        let output_ = Linear::calibrate(
            &graph.tensors[&input],
            k, c,
            &graph.tensors[&weights],
            &graph.tensors[&offset],
        );

        graph.tensors.insert(output.clone(), output_);
    }

    fn softmax(
        graph: &mut Builder,
        input: String,
        _input_data: *const c_void,
        output: String,
        _output_data: *mut c_void
    )
    {
        let output_ = Softmax::calibrate(&graph.tensors[&input]);

        graph.tensors.insert(output.clone(), output_);
    }

    fn relu(
        graph: &mut Builder,
        input: String,
        _input_data: *const c_void,
        output: String,
        _output_data: *mut c_void
    )
    {
        let output_ = Relu::calibrate(&graph.tensors[&input]);

        graph.tensors.insert(output.clone(), output_);
    }

    fn tanh(
        graph: &mut Builder,
        input: String,
        _input_data: *const c_void,
        output: String,
        _output_data: *mut c_void
    )
    {
        let output_ = Tanh::calibrate(&graph.tensors[&input]);

        graph.tensors.insert(output.clone(), output_);
    }

    fn get_slot(_graph: &mut Builder, _name: &'static str, _size_in_bytes: usize) -> SlotGuard {
        SlotGuard::null()
    }
}

struct Allocate;

impl Ops<Workspace> for Allocate {
    fn convolution(
        workspace: &mut Workspace,
        input: String,
        _input_data: *const c_void,
        _k: i32,
        _c: i32,
        _h: i32,
        _w: i32,
        weights: String,
        offset: String,
        output: String,
        _output_data: *mut c_void,
        _workspace: *mut c_void,
        _workspace_size: usize
    )
    {
        debug_assert!(!workspace.convolutions.contains_key(&output));

        workspace.convolutions.insert(output.clone(), Convolution::new(
            workspace.handle_dnn,
            1.0,
            &workspace.tensors[&input],
            &workspace.tensors[&weights],
            0.0,
            &Tensor::default(),
            &workspace.tensors[&offset],
            &workspace.tensors[&output]
        ));
    }

    fn residual_block(
        workspace: &mut Workspace,
        input: String,
        _input_data: *const c_void,
        _k: i32,
        _c: i32,
        _h: i32,
        _w: i32,
        weights_1: String,
        weights_2: String,
        offset_1: String,
        offset_2: String,
        output_1: String,
        _output_data_1: *mut c_void,
        output_2: String,
        _output_data_2: *mut c_void,
        _workspace: *mut c_void,
        _workspace_size: usize
    )
    {
        debug_assert!(!workspace.convolutions.contains_key(&output_1));
        debug_assert!(!workspace.convolutions.contains_key(&output_2));

        workspace.convolutions.insert(output_1.clone(), Convolution::new(
            workspace.handle_dnn,
            1.0,
            &workspace.tensors[&input],
            &workspace.tensors[&weights_1],
            0.0,
            &Tensor::default(),
            &workspace.tensors[&offset_1],
            &workspace.tensors[&output_1]
        ));

        workspace.convolutions.insert(output_2.clone(), Convolution::new(
            workspace.handle_dnn,
            1.0,
            &workspace.tensors[&output_1],
            &workspace.tensors[&weights_2],
            1.0,
            &workspace.tensors[&input],
            &workspace.tensors[&offset_2],
            &workspace.tensors[&output_2]
        ));
    }

    fn linear(
        workspace: &mut Workspace,
        input: String,
        _input_data: *const c_void,
        _k: i32,
        _c: i32,
        weights: String,
        offset: String,
        output: String,
        _output_data: *mut c_void,
        _workspace: *mut c_void,
        _workspace_size: usize
    )
    {
        workspace.linears.insert(output.clone(), Linear::new(
            &workspace.tensors[&input],
            &workspace.tensors[&weights],
            &workspace.tensors[&offset],
            &workspace.tensors[&output]
        ));
    }

    fn softmax(
        workspace: &mut Workspace,
        input: String,
        _input_data: *const c_void,
        output: String,
        _output_data: *mut c_void
    )
    {
        debug_assert!(!workspace.operators.contains_key(&output));

        let output_ = Softmax::new(
            &workspace.tensors[&input],
            &workspace.tensors[&output]
        );

        workspace.operators.insert(output.clone(), output_);
    }

    fn relu(
        workspace: &mut Workspace,
        _input: String,
        _input_data: *const c_void,
        output: String,
        _output_data: *mut c_void
    )
    {
        debug_assert!(!workspace.operators.contains_key(&output));

        workspace.operators.insert(output, Relu::new());
    }

    fn tanh(
        workspace: &mut Workspace,
        _input: String,
        _input_data: *const c_void,
        output: String,
        _output_data: *mut c_void
    )
    {
        debug_assert!(!workspace.operators.contains_key(&output));

        workspace.operators.insert(output.clone(), Tanh::new());
    }

    fn get_slot(graph: &mut Workspace, name: &'static str, size_in_bytes: usize) -> SlotGuard {
        if size_in_bytes == 0 {
            SlotGuard::null()  // unknown size, need to wait until runtime
        } else {
            graph.get_slot(name, size_in_bytes)
        }
    }
}

pub struct Runtime;

impl Ops<Workspace> for Runtime {
    fn convolution(
        workspace: &mut Workspace,
        input_name: String,
        input_data: *const c_void,
        _k: i32,
        _c: i32,
        _h: i32,
        _w: i32,
        weights_name: String,
        offset_name: String,
        output_name: String,
        output_data: *mut c_void,
        workspace_data: *mut c_void,
        workspace_size: usize
    )
    {
        let op = &workspace.convolutions[&output_name];

        op.forward(
            workspace.handle_dnn,
            &workspace.tensors[&input_name], input_data,
            &workspace.tensors[&weights_name],
            &workspace.tensors[&output_name], output_data,
            &workspace.tensors[&offset_name],
            &workspace.tensors[&output_name], output_data,
            workspace_data, workspace_size,
            workspace.relu
        );

        #[cfg(feature = "trace-cuda")]
        eprintln!("{} <- conv2d({})\n= {:?}", output_name, input_name, workspace.tensors[&output_name].fmt_ptr(output_data));
    }

    fn residual_block(
        workspace: &mut Workspace,
        input_name: String,
        input_data: *const c_void,
        _k: i32,
        _c: i32,
        _h: i32,
        _w: i32,
        weights_name_1: String,
        weights_name_2: String,
        offset_name_1: String,
        offset_name_2: String,
        output_name_1: String,
        output_data_1: *mut c_void,
        output_name_2: String,
        output_data_2: *mut c_void,
        workspace_data: *mut c_void,
        workspace_size: usize
    )
    {
        let op_1 = &workspace.convolutions[&output_name_1];
        let op_2 = &workspace.convolutions[&output_name_2];

        op_1.forward(
            workspace.handle_dnn,
            &workspace.tensors[&input_name], input_data,
            &workspace.tensors[&weights_name_1],
            &workspace.tensors[&output_name_1], output_data_1,
            &workspace.tensors[&offset_name_1],
            &workspace.tensors[&output_name_1], output_data_1,
            workspace_data, workspace_size,
            workspace.relu
        );

        #[cfg(feature = "trace-cuda")]
        eprintln!("{} <- conv2d({})\n= {:?}", output_name_1, input_name, workspace.tensors[&output_name_1].fmt_ptr(output_data_1));

        op_2.forward(
            workspace.handle_dnn,
            &workspace.tensors[&output_name_1], output_data_1,
            &workspace.tensors[&weights_name_2],
            &workspace.tensors[&input_name], input_data,
            &workspace.tensors[&offset_name_2],
            &workspace.tensors[&output_name_2], output_data_2,
            workspace_data, workspace_size,
            workspace.relu
        );

        #[cfg(feature = "trace-cuda")]
        eprintln!("{} <- conv2d({}) + {}\n= {:?}", output_name_2, output_name_2, input_name, workspace.tensors[&output_name_2].fmt_ptr(output_data_2));
    }

    fn linear(
        workspace: &mut Workspace,
        input_name: String,
        input_data: *const c_void,
        k: i32,
        c: i32,
        weights_name: String,
        offset_name: String,
        output_name: String,
        output_data: *mut c_void,
        workspace_data: *mut c_void,
        workspace_size: usize
    )
    {
        let linear = &workspace.linears[&output_name];

        linear.forward(
            workspace.handle_blas,
            workspace.handle_dnn,
            &workspace.tensors[&input_name], input_data,
            workspace.batch_size as i32, k, c,
            &workspace.tensors[&weights_name],
            &workspace.tensors[&offset_name],
            &workspace.tensors[&output_name], output_data,
            workspace_data, workspace_size
        );

        #[cfg(feature = "trace-cuda")]
        eprintln!("{} <- linear({})\n= {:?}", output_name, input_name, workspace.tensors[&output_name].fmt_ptr(output_data));
    }

    fn softmax(
        workspace: &mut Workspace,
        input_name: String,
        input_data: *const c_void,
        output_name: String,
        output_data: *mut c_void
    )
    {
        Softmax::forward(
            &workspace.operators[&output_name],
            workspace.handle_dnn,
            &workspace.tensors[&input_name], input_data,
            &workspace.tensors[&output_name], output_data
        );

        #[cfg(feature = "trace-cuda")]
        eprintln!("{} <- softmax({})\n= {:?}", output_name, input_name, workspace.tensors[&output_name].fmt_ptr(output_data));
    }

    fn relu(
        workspace: &mut Workspace,
        input_name: String,
        input_data: *const c_void,
        output_name: String,
        output_data: *mut c_void
    )
    {
        Relu::forward(
            &workspace.operators[&output_name],
            workspace.handle_dnn, workspace.relu,
            &workspace.tensors[&input_name], input_data,  // input
            &workspace.tensors[&output_name], output_data  // output
        );

        #[cfg(feature = "trace-cuda")]
        eprintln!("{} <- relu({})\n= {:?}", output_name, input_name, workspace.tensors[&output_name].fmt_ptr(output_data));
    }

    fn tanh(
        workspace: &mut Workspace,
        input_name: String,
        input_data: *const c_void,
        output_name: String,
        output_data: *mut c_void
    )
    {
        Tanh::forward(
            &workspace.operators[&output_name],
            workspace.handle_dnn, workspace.tanh,
            &workspace.tensors[&input_name], input_data,
            &workspace.tensors[&output_name], output_data
        );

        #[cfg(feature = "trace-cuda")]
        eprintln!("{} <- tanh({})\n= {:?}", output_name, input_name, workspace.tensors[&output_name].fmt_ptr(output_data));
    }

    fn get_slot(graph: &mut Workspace, name: &'static str, size_in_bytes: usize) -> SlotGuard {
        graph.get_slot(name, size_in_bytes)
    }
}

pub struct Builder {
    pub(super) tensors: HashMap<String, Tensor>,
}

pub struct Workspace {
    pub(super) tensors: HashMap<String, Tensor>,

    // handles for cuBLAS and cuDNN
    pub(super) handle_blas: cublas::Handle,
    pub(super) handle_dnn: cudnn::Handle,

    // operators
    pub(super) convolutions: HashMap<String, Convolution>,
    pub(super) linears: HashMap<String, Linear>,
    pub(super) operators: HashMap<String, Operator>,

    // activation operators
    pub(super) relu: cudnn::ActivationDescriptor,
    pub(super) tanh: cudnn::ActivationDescriptor,

    // workspaces and slots for additional variables
    pub(super) slots: Slots,
    pub(super) slots_c: HashMap<String, SlotGuard>,
    pub(super) workspace_size: usize,
    pub(super) batch_size: usize,

    // streams and events for concurrent execution
    pub(super) tower_stream: cuda::Stream,
    pub(super) policy_stream: cuda::Stream,
    pub(super) value_stream: cuda::Stream,
    pub(super) tower_event: cuda::Event,
}

impl Graph for Builder {
    fn get_input(&mut self, _size_in_bytes: Option<usize>) -> SlotGuard {
        SlotGuard::null()
    }

    fn get_slot(&mut self, _name: &'static str, _size_in_bytes: usize) -> SlotGuard {
        SlotGuard::null()
    }

    fn get_batch_size(&self) -> usize {
        0
    }

    fn get_workspace_size(&self) -> usize {
        0
    }
}

impl Graph for Workspace {
    fn get_input(&mut self, size_in_bytes: Option<usize>) -> SlotGuard {
        self.get_slot("00_input/features:0", size_in_bytes.unwrap_or(0))
    }

    fn get_slot(&mut self, name: &'static str, size_in_bytes: usize) -> SlotGuard {
        let name_str = name.to_string();

        if self.slots_c.contains_key(&name_str) {
            self.slots_c[&name_str].clone()
        } else {
            let slot = self.slots.get_slot(name, size_in_bytes);

            self.slots_c.insert(name_str, slot.clone());
            slot
        }
    }

    fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    fn get_workspace_size(&self) -> usize {
        self.workspace_size
    }
}

impl Builder {
    pub fn new(tensors: HashMap<String, Tensor>) -> Builder {
        let mut g = Builder {
            tensors: tensors
        };

        // add the placeholder tensor that represents the input
        let (data_type, format) = match *TYPE {
            Type::Half => (cudnn::DataType::Half, cudnn::TensorFormat::NCHW),
            Type::Int8 => (cudnn::DataType::Int8, cudnn::TensorFormat::NHWC),
            Type::Single => (cudnn::DataType::Float, cudnn::TensorFormat::NCHW)
        };

        g.tensors.insert("00_input/output:0".to_string(), Tensor::default()
            .set_data_type(data_type, format)
            .set_shape(vec! [0, 36, 19, 19])
            .set_scale(1.0)
            .clone()
        );

        // perform the calibration of every tensor so that we can acquire
        // the correct scaling factors, and types.
        tower::<Calibrate, _>(&mut g, &SlotGuard::null());
        value::<Calibrate, _>(&mut g, &SlotGuard::null());
        policy::<Calibrate, _>(&mut g, &SlotGuard::null());

        // if the entire graph is float-based then force the scale to 1.0
        // since floating types does the scaling internally (better than we
        // can).
        let is_floating_point = g.tensors.values().all(|t| {
            let data_type = t.get_data_type();

            data_type == cudnn::DataType::Half || data_type == cudnn::DataType::Float
        });

        if is_floating_point {
            for t in g.tensors.values() {
                t.set_scale(1.0);
            }
        }

        #[cfg(feature = "trace-cuda")]
        {
            for (key, tensor) in g.tensors.iter() {
                eprintln!("T[{}] = {:?} {:?}, {:?} scale {}",
                    key,
                    tensor.get_shape(),
                    tensor.get_format(),
                    tensor.get_data_type(),
                    tensor.get_scale()
                );
            }
        }

        g.finalize();
        g
    }

    /// Returns a mutable workspace that contains everything you need to
    /// perform a forward pass through the network pre-allocated.
    /// 
    /// # Arguments
    /// 
    /// * `batch_size` - 
    /// 
    pub fn get_workspace(&self, batch_size: usize, slots: Slots) -> Workspace {
        let mut w = Workspace {
            tensors: HashMap::new(),

            handle_blas: ptr::null(),
            handle_dnn: ptr::null(),

            convolutions: HashMap::new(),
            linears: HashMap::new(),
            operators: HashMap::new(),

            relu: ptr::null(),
            tanh: ptr::null(),

            slots: slots,
            slots_c: HashMap::new(),
            workspace_size: 0,
            batch_size: batch_size,

            tower_stream: ptr::null_mut(),
            policy_stream: ptr::null_mut(),
            value_stream: ptr::null_mut(),
            tower_event: ptr::null_mut()
        };

        // adjust the batch size of each tensor (by setting the first dimension), and
        // create the tensor descriptors so that we can use them during the runtime.
        for (key, tensor) in self.tensors.iter() {
            let mut other = tensor.clone();
            let mut other_shape = other.get_shape().clone();

            unsafe {
                if other_shape[0] == 0 {
                    other_shape[0] = batch_size as i32;

                    other.set_shape(other_shape);
                    other.tensor_desc = other.get_tensor_descriptor();
                }
            }

            w.tensors.insert(key.clone(), other);
        }

        // allocate the pre-determined parts of the array
        unsafe {
            check!(cublas::cublasCreate_v2(&mut w.handle_blas));
            check!(cudnn::cudnnCreate(&mut w.handle_dnn));

            #[cfg(feature = "tensor-core")]
            {
                check!(cublas::cublasSetMathMode(
                    w.handle_blas,
                    cublas::Math::TensorOp
                ));
            }

            check!(cudnn::cudnnCreateActivationDescriptor(&mut w.relu));
            check!(cudnn::cudnnSetActivationDescriptor(
                w.relu,
                cudnn::ActivationMode::Relu,
                cudnn::NanPropagation::NotPropagateNan,
                0.0
            ));

            check!(cudnn::cudnnCreateActivationDescriptor(&mut w.tanh));
            check!(cudnn::cudnnSetActivationDescriptor(
                w.tanh,
                cudnn::ActivationMode::Tanh,
                cudnn::NanPropagation::NotPropagateNan,
                0.0
            ));

            check!(cuda::cudaStreamCreate(&mut w.tower_stream));
            check!(cuda::cudaStreamCreate(&mut w.policy_stream));
            check!(cuda::cudaStreamCreate(&mut w.value_stream));
            check!(cuda::cudaEventCreate(&mut w.tower_event));
        }

        tower::<Allocate, _>(&mut w, &SlotGuard::null());
        value::<Allocate, _>(&mut w, &SlotGuard::null());
        policy::<Allocate, _>(&mut w, &SlotGuard::null());

        // determine the maximum observed workspace size
        let conv_workspace = w.convolutions.values()
            .map(|c| c.workspace_size)
            .max().unwrap_or(0);
        let linear_workspace = w.linears.values()
            .map(|l| l.workspace_size)
            .max().unwrap_or(0);

        w.workspace_size = ::std::cmp::max(conv_workspace, linear_workspace);
        w
    }

    /// Finalize all internal data structures and copies them to
    /// the GPU.
    pub fn finalize(&mut self) {
        // copy the weights to the device if that has not been done already.
        for (name, tensor) in self.tensors.iter_mut() {
            unsafe {
                if tensor.has_shape() {
                    tensor.filter_desc = tensor.get_filter_descriptor();
                    tensor.tensor_desc = tensor.get_tensor_descriptor();
                    tensor.copy_to_device();

                    // check that the weights on the GPU are approximately the
                    // same as the CPU weights.
                    tensor.check(&name);
                }
            }
        }
    }
}

impl Workspace {
    /// Release (to the memory pool) any temporary memory that was allocated for
    /// this workspace during the last run.
    pub fn clear(&mut self) {
        self.slots_c.clear();
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        unsafe {
            cuda::cudaEventDestroy(self.tower_event);
            cuda::cudaStreamDestroy(self.value_stream);
            cuda::cudaStreamDestroy(self.policy_stream);
            cuda::cudaStreamDestroy(self.tower_stream);

            cudnn::cudnnDestroyActivationDescriptor(self.tanh);
            cudnn::cudnnDestroyActivationDescriptor(self.relu);

            cudnn::cudnnDestroy(self.handle_dnn);
            cublas::cublasDestroy_v2(self.handle_blas);
        }
    }
}

/// 
pub fn tower<O: Ops<G>, G: Graph>(graph: &mut G, input: &SlotGuard) -> SlotGuard {
    let batch_size = graph.get_batch_size();
    let residual_1 = O::get_slot(graph, "residual_1", 4 * batch_size * NUM_FEATURES * 361);
    let residual_2 = O::get_slot(graph, "residual_2", 4 * batch_size * NUM_FEATURES * 361);
    let workspace_size = graph.get_workspace_size();
    let workspace_1 = O::get_slot(graph, "workspace_1", workspace_size);

    /*
    #[cfg(feature = "trace-cuda")]
    eprintln!("00_input/output:0\n= {:?}", Tensor::default()
        .set_data_type(cudnn::DataType::Int8, cudnn::TensorFormat::NHWC)
        .set_shape(vec! [batch_size as i32, 36, 19, 19])
        .set_scale(1.0)
        .fmt_ptr(input.ptr)
    );
    */

    O::convolution(
        graph,
        "00_input/output:0".to_string(), **input,
        NUM_FEATURES as i32, 36, 3, 3,
        "01_upsample/weights:0".to_string(),
        "01_upsample/offset:0".to_string(),
        "01_upsample/output:0".to_string(), *residual_1,
        *workspace_1, workspace_size
    );

    for i in 2..(NUM_LAYERS + 2) {
        let input_name = if i == 2 {
            "01_upsample/output:0".to_string()
        } else {
            format!("{:02}_residual/output_2:0", i - 1)
        };

        O::residual_block(
            graph,
            input_name, *residual_1,
            NUM_FEATURES as i32, NUM_FEATURES as i32, 3, 3,
            format!("{:02}_residual/weights_1:0", i),
            format!("{:02}_residual/weights_2:0", i),
            format!("{:02}_residual/offset_1:0", i),
            format!("{:02}_residual/offset_2:0", i),
            format!("{:02}_residual/output_1:0", i), *residual_2,
            format!("{:02}_residual/output_2:0", i), *residual_1,
            *workspace_1, workspace_size
        );
    }

    residual_1
}

/// 
pub fn policy<O: Ops<G>, G: Graph>(graph: &mut G, residual_1: &SlotGuard) -> SlotGuard {
    let batch_size = graph.get_batch_size();
    let policy_1 = O::get_slot(graph, "policy_1", 4 * batch_size * 362);
    let policy_out = graph.get_slot("99_output/policy:0", 4 * batch_size * 722);
    let workspace_size = graph.get_workspace_size();
    let workspace_1 = O::get_slot(graph, "workspace_1", workspace_size);

    O::convolution(
        graph,
        format!("{:02}_residual/output_2:0", NUM_LAYERS + 1), **residual_1,
        2, NUM_FEATURES as i32, 1, 1,
        format!("{:02}p_policy/downsample:0", NUM_LAYERS + 2),
        format!("{:02}p_policy/offset:0", NUM_LAYERS + 2),
        format!("{:02}p_policy/output_1:0", NUM_LAYERS + 2), *policy_out,
        *workspace_1, workspace_size
    );

    O::linear(
        graph,
        format!("{:02}p_policy/output_1:0", NUM_LAYERS + 2), *policy_out,
        362, 722,
        format!("{:02}p_policy/weights:0", NUM_LAYERS + 2),
        format!("{:02}p_policy/bias:0", NUM_LAYERS + 2),
        format!("{:02}p_policy/output_2:0", NUM_LAYERS + 2), *policy_1,
        *workspace_1, workspace_size
    );

    O::softmax(
        graph,
        format!("{:02}p_policy/output_2:0", NUM_LAYERS + 2), *policy_1,
        format!("{:02}p_policy/output_3:0", NUM_LAYERS + 2), *policy_out
    );

    policy_out
}

/// 
pub fn value<O: Ops<G>, G: Graph>(graph: &mut G, residual_1: &SlotGuard) -> SlotGuard {
    let batch_size = graph.get_batch_size();
    let value_1 = O::get_slot(graph, "value_1", 4 * batch_size * 361);
    let value_out = graph.get_slot("99_output/value:0", 4 * batch_size * 256);
    let workspace_size = graph.get_workspace_size();
    let workspace_2 = O::get_slot(graph, "workspace_2", workspace_size);

    O::convolution(
        graph,
        format!("{:02}_residual/output_2:0", NUM_LAYERS + 1), **residual_1,
        2, NUM_FEATURES as i32, 1, 1,
        format!("{:02}v_value/downsample:0", NUM_LAYERS + 2),
        format!("{:02}v_value/offset:0", NUM_LAYERS + 2),
        format!("{:02}v_value/output_1:0", NUM_LAYERS + 2), *value_1,
        *workspace_2, workspace_size
    );

    O::linear(
        graph,
        format!("{:02}v_value/output_1:0", NUM_LAYERS + 2), *value_1,
        256, 722,
        format!("{:02}v_value/weights_1:0", NUM_LAYERS + 2),
        format!("{:02}v_value/bias_1:0", NUM_LAYERS + 2),
        format!("{:02}v_value/output_2:0", NUM_LAYERS + 2), *value_out,
        *workspace_2, workspace_size
    );

    O::relu(
        graph,
        format!("{:02}v_value/output_2:0", NUM_LAYERS + 2), *value_out,
        format!("{:02}v_value/output_3:0", NUM_LAYERS + 2), *value_out
    );

    O::linear(
        graph,
        format!("{:02}v_value/output_3:0", NUM_LAYERS + 2), *value_out,
        1, 256,
        format!("{:02}v_value/weights_2:0", NUM_LAYERS + 2),
        format!("{:02}v_value/bias_2:0", NUM_LAYERS + 2),
        format!("{:02}v_value/output_4:0", NUM_LAYERS + 2), *value_1,
        *workspace_2, workspace_size
    );

    O::tanh(
        graph,
        format!("{:02}v_value/output_4:0", NUM_LAYERS + 2), *value_1,
        format!("{:02}v_value/output_5:0", NUM_LAYERS + 2), *value_out
    );

    value_out
}
