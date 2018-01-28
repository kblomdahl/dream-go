// Copyright 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ptr;
use libc::c_void;

use nn::ffi::cuda;
use nn::ffi::cudnn;
use util::types::*;
use util::{max, min};

/// A data structure with interior mutability that store the host,
/// device, and meta information about a tensor.
pub struct Tensor {
    /// The unscaled tensor in host-memory as `f16` if applicable.
    pub host: Option<Box<[f16]>>,

    /// The scaled tensor in device memory as the type given in
    /// `dtype`, or null if not applicable.
    pub ptr: *mut c_void,

    /// The `TensorDescriptor` in device memory that represents
    /// this tensor.
    pub tensor_desc: cudnn::TensorDescriptor,

    /// The `FilterDescriptor` in device memory that represents
    /// this tensor.
    pub filter_desc: cudnn::TensorDescriptor,

    /// The number of references to the allocated device memory.
    pub ref_count: *mut AtomicUsize,

    /// The format of the data in device and host memory.
    pub format: RefCell<cudnn::TensorFormat>,

    /// The type of the data in device memory.
    pub data_type: RefCell<cudnn::DataType>,

    /// The shape of the data in NCHW format.
    pub shape: RefCell<Vec<i32>>,

    /// The appropriate scale, as calculated by the mean and variance.
    pub scale: RefCell<f32>
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        unsafe {
            (*self.ref_count).fetch_add(1, Ordering::SeqCst);
        }

        Tensor {
            host: None,

            ptr: self.ptr,
            tensor_desc: self.tensor_desc,
            filter_desc: self.filter_desc,
            ref_count: self.ref_count,

            format: RefCell::new(self.get_format()),
            data_type: RefCell::new(self.get_data_type()),
            shape: RefCell::new(self.get_shape()),
            scale: RefCell::new(self.get_scale()),
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe {
            if (*self.ref_count).fetch_sub(1, Ordering::SeqCst) == 1 {
                Box::from_raw(self.ref_count);

                check!(cuda::cudaFree(self.ptr));
                check!(cudnn::cudnnDestroyFilterDescriptor(self.filter_desc));
                check!(cudnn::cudnnDestroyTensorDescriptor(self.tensor_desc));
            }
        }
    }
}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor {
            host: None,

            ptr: ptr::null_mut(),
            tensor_desc: ptr::null(),
            filter_desc: ptr::null(),
            ref_count: Box::into_raw(Box::new(AtomicUsize::new(1))),

            format: RefCell::new(cudnn::TensorFormat::NCHW),
            data_type: RefCell::new(cudnn::DataType::Float),
            shape: RefCell::new(vec! []),
            scale: RefCell::new(1.0)
        }
    }
}

impl Tensor {
    pub fn zeros_like(other: &Tensor) -> Tensor {
        Tensor {
            host: None,

            ptr: ptr::null_mut(),
            tensor_desc: ptr::null(),
            filter_desc: ptr::null(),
            ref_count: Box::into_raw(Box::new(AtomicUsize::new(1))),

            format: RefCell::new(other.get_format()),
            data_type: RefCell::new(other.get_data_type()),
            shape: RefCell::new(other.get_shape()),
            scale: RefCell::new(other.get_scale())
        }
    }

    pub fn get_data_type(&self) -> cudnn::DataType {
        *self.data_type.borrow()
    }

    pub fn get_format(&self) -> cudnn::TensorFormat {
        *self.format.borrow()
    }

    pub fn set_data_type(&self, data_type: cudnn::DataType, format: cudnn::TensorFormat) -> &Tensor {
        *self.data_type.borrow_mut() = data_type;
        *self.format.borrow_mut() = format;
        self
    }

    pub fn set_data_type_like(&self, other: &Tensor) -> &Tensor {
        *self.data_type.borrow_mut() = other.get_data_type();
        *self.format.borrow_mut() = other.get_format();
        self
    }

    pub fn set_host(&mut self, host: Box<[f16]>) -> &mut Tensor {
        self.host = Some(host);
        self
    }

    pub fn get_scale(&self) -> f32 {
        *self.scale.borrow()
    }

    pub fn set_scale(&self, scale: f32) -> &Tensor {
        debug_assert!(scale > 0.0);

        *self.scale.borrow_mut() = scale;
        self
    }

    pub fn get_shape(&self) -> Vec<i32> {
        self.shape.borrow().clone()
    }

    pub fn has_shape(&self) -> bool {
        let shape = self.shape.borrow();

        !shape.is_empty() && shape.iter().product::<i32>() > 0
    }

    pub fn set_shape(&self, shape: Vec<i32>) -> &Tensor {
        *self.shape.borrow_mut() = shape;
        self
    }

    pub fn set_shape_like(&self, other: &Tensor) -> &Tensor {
        *self.shape.borrow_mut() = other.get_shape();
        self
    }

    /// Returns the total size of the GPU data in bytes.
    pub fn get_size_in_bytes(&self) -> usize {
        self.get_data_type().size() * self.get_shape().into_iter().product::<i32>() as usize
    }

    /// Returns a cuDNN `FilterDescriptor` with the parameters specified in this
    /// tensor.
    pub unsafe fn get_filter_descriptor(&self) -> cudnn::FilterDescriptor {
        let shape = self.shape.borrow();

        if shape.len() < 4 {  // linear layer weights
            ptr::null()
        } else {
            let mut descr = ptr::null();

            check!(cudnn::cudnnCreateFilterDescriptor(&mut descr));
            check!(cudnn::cudnnSetFilter4dDescriptor(
                descr,
                self.get_data_type(),
                self.get_format(),
                shape[0], shape[1], shape[2], shape[3]
            ));

            descr
        }
    }

    /// Returns a cuDNN `TensorDescriptor` with the parameters specified in this
    /// tensor.
    pub unsafe fn get_tensor_descriptor(&self) -> cudnn::TensorDescriptor {
        let shape = self.shape.borrow();

        if shape.len() < 4 {  // linear layer weights
            ptr::null()
        } else {
            let mut descr = ptr::null();

            check!(cudnn::cudnnCreateTensorDescriptor(&mut descr));
            check!(cudnn::cudnnSetTensor4dDescriptor(
                descr,
                self.get_format(),
                self.get_data_type(),
                shape[0], shape[1], shape[2], shape[3]
            ));

            descr
        }
    }

    /// Allocate the necessary memory on the GPU and copy the scaled version of
    /// this tensor to the GPU.
    pub unsafe fn copy_to_device(&mut self) {
        if let Some(ref host) = self.host {
            let scale = self.get_scale();

            if !self.ptr.is_null() {
                check!(cuda::cudaFree(self.ptr));
            }

            match self.get_data_type() {
                cudnn::DataType::Float => {
                    let elements = host.iter()
                        .map(|&x| f32::from(x) / scale)
                        .collect::<Vec<f32>>();
                    let size = 4 * elements.len();

                    check!(cuda::cudaMalloc(&mut self.ptr, size));
                    check!(cuda::cudaMemcpy(self.ptr, elements.as_ptr() as *const c_void, size, cuda::MemcpyKind::HostToDevice));
                },

                cudnn::DataType::Half => {
                    let elements = host.iter()
                        .map(|&x| f16::from(f32::from(x) / scale))
                        .collect::<Vec<f16>>();
                    let size = 2 * elements.len();

                    check!(cuda::cudaMalloc(&mut self.ptr, size));
                    check!(cuda::cudaMemcpy(self.ptr, elements.as_ptr() as *const c_void, size, cuda::MemcpyKind::HostToDevice));
                },

                cudnn::DataType::Int8 => {
                    let elements = host.iter()
                        .map(|&x| {
                            let x = f32::from(x);

                            if x <= -scale {
                                -128
                            } else if x >= scale {
                                127
                            } else {
                                ((127.0 * x) / scale).round() as i8
                            }
                        })
                        .collect::<Vec<i8>>();
                    let size = 1 * elements.len();

                    check!(cuda::cudaMalloc(&mut self.ptr, size));
                    check!(cuda::cudaMemcpy(self.ptr, elements.as_ptr() as *const c_void, size, cuda::MemcpyKind::HostToDevice));
                },

                _ => {
                    panic!();
                }
            };
        }
    }

    /// Returns the data stored in the GPU for this tensor.
    /// 
    /// # Arguments
    /// 
    /// * `ptr` - 
    /// 
    unsafe fn load(&self, ptr: *const c_void) -> Vec<f32> {
        let size = self.get_shape().iter().product::<i32>();

        if size > 0 && !ptr.is_null() {
            let size = size as usize;
            let scale = self.get_scale();

            match *self.data_type.borrow() {
                cudnn::DataType::Float => {
                    let mut elements = vec! [0.0f32; size];

                    check!(cuda::cudaMemcpy(elements.as_mut_ptr() as *mut c_void, ptr, 4 * size, cuda::MemcpyKind::DeviceToHost));

                    for element in elements.iter_mut() {
                        *element *= scale;
                    }

                    elements
                },

                cudnn::DataType::Half => {
                    let mut elements = vec! [f16::from(0.0); size];

                    check!(cuda::cudaMemcpy(elements.as_mut_ptr() as *mut c_void, ptr, 2 * size, cuda::MemcpyKind::DeviceToHost));

                    elements.into_iter()
                        .map(|x| f32::from(x) * scale)
                        .collect()
                },

                cudnn::DataType::Int8 => {
                    let mut elements = vec! [0i8; size];

                    check!(cuda::cudaMemcpy(elements.as_mut_ptr() as *mut c_void, ptr, 1 * size, cuda::MemcpyKind::DeviceToHost));

                    elements.into_iter()
                        .map(|x| (x as f32 / 127.0) * scale)
                        .collect()
                },

                _ => {
                    panic!();
                }
            }
        } else {
            vec! []
        }
    }

    /// Asserts that the array that is stored on the GPU is the same as the one
    /// that is stored in host memory.
    pub unsafe fn check(&self, name: &str) {
        let elements = self.load(self.ptr);
        let size = elements.len();

        if let Some(ref host) = self.host {
            for i in 0..size {
                let h = if self.get_data_type() == cudnn::DataType::Int8 {
                    let scale = self.get_scale();

                    min(max(f32::from(host[i]), -scale), scale)
                } else {
                    f32::from(host[i])
                };
                let d = (h - elements[i]).abs();

                // compare against a fairly large margin since we use `i8` to
                // represent the value and they can be quite lossy.
                assert!(d < 1e-1, "{}[{}]: {} ~ {}", name, i, h, elements[i]);
            }
        }
    }

    /// Returns a pretty-printable version of this.
    #[cfg(feature = "trace-cuda")]
    pub fn fmt<'a>(&'a self) -> TensorFmt<'a> {
        TensorFmt {
            inner: self,
            ptr: self.ptr
        }
    }

    /// Returns a pretty-printable version of this with custom data.
    /// 
    /// # Arguments
    /// 
    /// * `data` - the data to use instead of the data in this tensor
    /// 
    #[cfg(feature = "trace-cuda")]
    pub fn fmt_ptr<'a>(&'a self, ptr: *const c_void) -> TensorFmt<'a> {
        TensorFmt {
            inner: self,
            ptr: ptr
        }
    }
}

#[cfg(feature = "trace-cuda")]
use std::fmt;

#[cfg(feature = "trace-cuda")]
pub struct TensorFmt<'a> {
    inner: &'a Tensor,
    ptr: *const c_void
}

#[cfg(feature = "trace-cuda")]
impl<'a> fmt::Debug for TensorFmt<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let inner = self.inner;

        // transfer the tensor from device to host memory and then convert it
        // to `f32` to make the rest of the code easier to write.
        let elements: Vec<f32> = if self.ptr.is_null() {
            vec! []
        } else {
            unsafe {
                check!(cuda::cudaDeviceSynchronize());

                inner.load(self.ptr)
            }
        };

        // transpose this tensor to NCHW format so that we can more easily compare
        // the elements against the tensorflow debugger
        /*
        let size = inner.get_shape().iter().product::<i32>() as usize;

        let elements = if inner.shape.len() == 4 && inner.format == cudnn::TensorFormat::NHWC && elements.len() == size && size > 0 {
            // transform from NHWC to NCHW (unless this is a weight array)
            let mut trans = vec! [0.0f32; size];
            let k_size = inner.shape[0] as usize;
            let c_size = inner.shape[1] as usize;
            let h_size = inner.shape[2] as usize;
            let w_size = inner.shape[3] as usize;

            for k in 0..k_size {
                let k_offset = k * w_size * h_size * c_size;

                for c in 0..c_size {
                    for h in 0..h_size {
                        for w in 0..w_size {
                            let out_index = k_offset + c * h_size*w_size + h * w_size + w;
                            let in_index = k_offset + h * w_size*c_size + w * c_size + c;

                            trans[out_index] = elements[in_index];
                        }
                    }
                }
            }

            trans
        } else {
            elements
        };
        */

        // calculate the actual mean and variance of these elements
        let min = elements.iter().fold(::std::f32::INFINITY, |acc, &x| if x < acc { x } else { acc });
        let max = elements.iter().fold(::std::f32::NEG_INFINITY, |acc, &x| if x > acc { x } else { acc });
        let mean = elements.iter().sum::<f32>() / (elements.len() as f32);
        let variance = elements.iter()
            .map(|x| { let d = x - mean; d*d })
            .sum::<f32>() / (elements.len() as f32);

        // pretty-print this tensor as:
        // 
        // E([0.0, 0.0, ..., 0.0, 0.0]) = mean (expect self.mean) +/- std (expect self.std) in [min, max]
        // 
        if elements.len() > 4 {
            let s = elements.len();

            write!(f, "E([{:8.2e}, {:8.2e}, ..., {:8.2e}, {:8.2e}])",
                elements[0], elements[1],
                elements[s-2], elements[s-1]
            )?;
        } else {
            let elements = elements.iter()
                .map(|x| format!("{:8.2e}", x))
                .collect::<Vec<String>>()
                .join(", ");

            write!(f, "E([{}])", elements)?;
        }

        write!(f, " = {:8.2e} +/- {:8.2e} in [{:8.2e}, {:8.2e}]",
            mean, variance,
            min, max
        )
    }
}
