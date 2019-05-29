// Copyright 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

use super::{Error, Stream};
use super::stream::CudaStream;

use libc::{c_uchar, c_void, c_int, c_uint};
use std::ffi::CString;
use std::ptr::{null_mut, null, Unique};

type CudaGraph = *mut c_void;
type CudaGraphExec = *mut c_void;
type CudaGraphNode = *mut c_void;

#[repr(i32)]
enum CudaStreamCaptureMode {
    Global = 0
}

#[link(name = "cuda")]
#[link(name = "cudart")]
extern {
    fn cudaStreamBeginCapture(stream: CudaStream, mode: CudaStreamCaptureMode) -> c_int;
    fn cudaStreamEndCapture(stream: CudaStream, graph: *mut CudaGraph) -> c_int;

    fn cudaGraphDestroy(graph: CudaGraph);
    fn cudaGraphExecDestroy(graph_exec: CudaGraphExec);
    fn cudaGraphInstantiate(graph_exec: &mut CudaGraphExec, graph: CudaGraph, error_node: CudaGraphNode, log_buffer: *const c_uchar, buffer_size: c_uint) -> c_int;
    fn cudaGraphLaunch(graph_exec: CudaGraphExec, stream: CudaStream) -> c_int;
}

pub struct Graph(Unique<c_void>);
pub struct GraphExec(Unique<c_void>);

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { cudaGraphDestroy(self.as_ptr()) };
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        unsafe { cudaGraphExecDestroy(self.as_ptr()) };
    }
}

pub trait StreamGraph {
    fn begin_capture(&self) -> Result<StreamCapture, Error>;
}

impl StreamGraph for Stream {
    fn begin_capture(&self) -> Result<StreamCapture, Error> {
        let success = unsafe { cudaStreamBeginCapture(self.as_ptr(), CudaStreamCaptureMode::Global) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(StreamCapture(self))
        }
    }
}

pub struct StreamCapture<'a>(&'a Stream);

impl<'a> StreamCapture<'a> {
    pub fn end_capture(self) -> Result<Graph, Error> {
        let mut graph = null_mut();
        let success = unsafe { cudaStreamEndCapture(self.0.as_ptr(), &mut graph) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(Graph(Unique::new(graph).unwrap()))
        }
    }
}

impl Graph {
    pub fn as_graph_exec(&self) -> Result<GraphExec, Error> {
        let mut graph_exec = null_mut();
        let success = unsafe { cudaGraphInstantiate(&mut graph_exec, self.as_ptr(), null_mut(), null(), 0) };

        if success != 0 {
            let mut error_message = vec! [0; 10240];
            let other_success = unsafe {
                cudaGraphInstantiate(&mut graph_exec, self.as_ptr(), null_mut(), error_message.as_mut_ptr(), 10240)
            };

            assert_eq!(success, other_success);
            Err(Error::CudaGraphError(success, CString::new(error_message).ok().and_then(|cstr| cstr.into_string().ok()).unwrap()))
        } else {
            Ok(GraphExec(Unique::new(graph_exec).unwrap()))
        }
    }

    pub(super) fn as_ptr(&self) -> CudaGraph {
        self.0.as_ptr()
    }
}

impl GraphExec {
    pub fn launch(&self, stream: &Stream) -> Result<(), Error> {
        let success = unsafe { cudaGraphLaunch(self.as_ptr(), stream.as_ptr()) };

        if success != 0 {
            Err(Error::CudaError(success))
        } else {
            Ok(())
        }
    }

    pub(super) fn as_ptr(&self) -> CudaGraphExec {
        self.0.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    // pass
}