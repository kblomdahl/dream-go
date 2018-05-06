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

use libc::{c_void, c_int, c_uint};

#[repr(i32)]
#[derive(Debug, PartialEq, Eq)]
pub enum DataType {
    R16F = 2,  // real as half
    R32F = 0,  // real as float
    R8I = 3,  // real as signed char
    R32I = 10,  // real as signed int
}

#[repr(i32)]
#[derive(Debug, PartialEq, Eq)]
pub enum DeviceAttr {
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76
}

#[repr(i32)]
#[derive(Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Error {
    Success = 0,
    MissingConfiguration = 1,
    MemoryAllocation = 2,
    InitializationError = 3,
    LaunchFailure = 4,
    LaunchTimeout = 6,
    LaunchOutOfResources = 7,
    InvalidDeviceFunction = 8,
    InvalidConfiguration = 9,
    InvalidDevice = 10,
    InvalidValue = 11,
    InvalidPitchValue = 12,
    InvalidSymbol = 13,
    MapBufferObjectFailed = 14,
    UnmapBufferObjectFailed = 15,
    InvalidHostPointer = 16,
    InvalidDevicePointer = 17,
    InvalidChannelDescriptor = 20,
    InvalidMemcpyDirection = 21,
    AddressOfConstant = 22,
    InvalidFilterSetting = 26,
    InvalidNormSetting = 27,
    CudartUnloading = 29,
    Unknown = 30,
    InvalidResourceHandle = 33,
    NotReady = 34,
    InsufficientDriver = 35,
    InvalidSurface = 37,
    NoDevice = 38,
    ECCUncorrectable = 39,
    SharedObjectSymbolNotFound = 40,
    SharedObjectInitFailed = 41,
    UnsupportedLimit = 42,
    DuplicateVariableName = 43,
    DevicesUnavailable = 46,
    InvalidKernelImage = 47,
    NoKernelImageForDevice = 48,
    IncompatibleDriverContext = 49,
    PeerAccessAlreadyEnabled = 50,
    PeerAccessNotEnabled = 51,
    DeviceAlreadyInUse = 54,
    ProfilerDisabled = 55,
    Assert = 59,
    TooManyPeers = 60,
    HostMemoryAlreadyRegistered = 61,
    HostMemoryNotRegistered = 62,
    OperatingSystem = 63,
    PeerAccessUnsupported = 64,
    LaunchMaxDepthExceeded = 65,
    SyncDepthExceeded = 68,
    LaunchPendingCountExceeded = 69,
    NotPermitted = 70,
    NotSupported = 71,
    HardwareStackError = 72,
    IllegalInstruction = 73,
    MisalignedAddress = 74,
    InvalidAddressSpace = 75,
    InvalidPc = 76,
    IllegalAddress = 77,
    InvalidPtx = 78,
    InvalidGraphicsContext = 79,
    NvlinkUncorrectable = 80,
    StartupFailure = 0x7f
}

impl Error {
    /// Returns whether this _error_ indicates a successful call.
    pub fn is_ok(&self) -> bool {
        *self == Error::Success
    }
}

#[repr(i32)]
#[allow(dead_code)]
pub enum MemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3
}

pub type Event = *const c_void;
pub type Stream = *const c_void;

#[link(name = "cuda")]
#[link(name = "cudart")]
extern {
    pub fn cudaFree(devPtr: *const c_void) -> Error;
    pub fn cudaFreeHost(hostPtr: *const c_void) -> Error;
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> Error;
    pub fn cudaMallocHost(hostPtr: *mut *mut c_void, size: usize) -> Error;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: MemcpyKind) -> Error;
    pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: MemcpyKind, stream: Stream) -> Error;

    pub fn cudaGetDeviceCount(count: *mut c_int) -> Error;
    pub fn cudaGetDevice(device: *mut c_int) -> Error;
    pub fn cudaSetDevice(device: c_int) -> Error;

    #[cfg(feature = "trace-cuda")]
    pub fn cudaDeviceSynchronize() -> Error;
    pub fn cudaRuntimeGetVersion(version: *mut c_int) -> Error;
    pub fn cudaDeviceGetAttribute(value: *mut c_int, attr: DeviceAttr, device: c_int) -> Error;

    pub fn cudaEventCreateWithFlags(event: *mut Event, flags: c_uint) -> Error;
    pub fn cudaEventDestroy(event: Event) -> Error;
    pub fn cudaEventRecord(event: Event, stream: Stream) -> Error;

    pub fn cudaStreamCreateWithFlags(stream: *mut Stream, flags: c_uint) -> Error;
    pub fn cudaStreamDestroy(stream: Stream) -> Error;
    pub fn cudaStreamSynchronize(stream: Stream) -> Error;
    pub fn cudaStreamWaitEvent(stream: Stream, event: Event, flags: u32) -> Error;
}
