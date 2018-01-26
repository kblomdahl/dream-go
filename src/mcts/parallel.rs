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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use threadpool::ThreadPool;

use mcts::param::Param;
use nn::{self, Network, Workspace};
use util::array::*;
use util::singleton::*;
use util::types::*;

pub enum PredictRequest {
    /// Shutdown the server thread.
    Shutdown,

    /// Indicate that new worker entered the pool.
    Increase,

    /// Indicate that a worker left the pool.
    Decrease,

    /// Request to compute the value and policy for some feature.
    Ask(Array, Sender<(Singleton, Array)>),

    /// Indicate that a worker is waiting for some other thread to finish
    /// and should be awaken after the next batch of computations finish.
    Wait(Sender<()>)
}

/// Run the `nn::forward` function for the given features and wrap the
/// results into `Array` elements. This version assumes the neural network
/// use `f32` weights.
/// 
/// # Arguments
/// 
/// * `workspace` - 
/// * `features_list` - 
/// 
fn forward_f32(workspace: &mut Workspace, features_list: Vec<Array>) -> (Vec<Singleton>, Vec<Array>) {
    let (value_list, policy_list) = nn::forward::<f32, _>(
        workspace,
        &features_list.into_iter()
            .map(|feature| {
                match feature {
                    Array::Single(inner) => inner,
                    _ => panic!()
                }
            })
            .collect()
    );

    // wrap the results in `Array` so that we can avoid having to pass
    // generics everywhere
    let value_list = value_list.into_iter()
        .map(|value| Singleton::from_f32(value))
        .collect();
    let policy_list = policy_list.into_iter()
        .map(|policy| Array::from_f32(policy))
        .collect();

    (value_list, policy_list)
}

/// Run the `nn::forward` function for the given features and wrap the
/// results into `Array` elements. This version assumes the neural network
/// use `f16` weights.
/// 
/// # Arguments
/// 
/// * `workspace` - 
/// * `features_list` - 
/// 
fn forward_f16(workspace: &mut Workspace, features_list: Vec<Array>) -> (Vec<Singleton>, Vec<Array>) {
    let (value_list, policy_list) = nn::forward::<f16, _>(
        workspace,
        &features_list.into_iter()
            .map(|feature| {
                match feature {
                    Array::Half(inner) => inner,
                    _ => panic!()
                }
            })
            .collect()
    );

    // wrap the results in `Array` so that we can avoid having to pass
    // generics everywhere
    let value_list = value_list.into_iter()
        .map(|value| Singleton::from_f16(value))
        .collect();
    let policy_list = policy_list.into_iter()
        .map(|policy| Array::from_f16(policy))
        .collect();

    (value_list, policy_list)
}

/// Run the `nn::forward` function for the given features and wrap the
/// results into `Array` elements. This version assumes the neural network
/// use `i8` weights.
/// 
/// # Arguments
/// 
/// * `workspace` - 
/// * `features_list` - 
/// 
fn forward_i8(workspace: &mut Workspace, features_list: Vec<Array>) -> (Vec<Singleton>, Vec<Array>) {
    let (value_list, policy_list) = nn::forward::<q8, f32>(
        workspace,
        &features_list.into_iter()
            .map(|feature| {
                match feature {
                    Array::Int8(inner) => inner,
                    _ => panic!()
                }
            })
            .collect()
    );

    // wrap the results in `Array` so that we can avoid having to pass
    // generics everywhere
    let value_list = value_list.into_iter()
        .map(|value| Singleton::from_f32(value))
        .collect();
    let policy_list = policy_list.into_iter()
        .map(|policy| Array::from_f32(policy))
        .collect();

    (value_list, policy_list)
}

/// Listens for requests over the given channel and delegate neural network
/// inference to a worker thread pool.
/// 
/// # Arguments
/// 
/// * `network` - the network to use for inference
/// * `receiver` - 
/// 
fn server_aux<C: Param>(
    network: Network,
    receiver: Receiver<PredictRequest>,
    is_running: Arc<(Mutex<bool>, Condvar)>
)
{
    // spin-up the pool of worker threads
    let batch_size = C::batch_size();
    let pool_size = C::thread_count() / batch_size;
    let pool = ThreadPool::new(pool_size);

    // start-up all of the neural network worker
    let mut worker_count = 0;
    let waiting_list = Arc::new(Mutex::new(vec! []));
    let pending_count = Arc::new(AtomicUsize::new(0));
    let network = Arc::new(network);

    'a: loop {
        let mut features_list = Vec::with_capacity(batch_size);
        let mut sender_list = Vec::with_capacity(batch_size);

        for msg in receiver.iter() {
            match msg {
                PredictRequest::Shutdown => {
                    debug_assert!(worker_count == 0);

                    break 'a;
                },
                PredictRequest::Increase => {
                    worker_count += 1;
                },
                PredictRequest::Decrease => {
                    worker_count -= 1;
                },
                PredictRequest::Ask(features, sender) => {
                    features_list.push(features);
                    sender_list.push(sender);
                },
                PredictRequest::Wait(sender) => {
                    let mut waiting_list = waiting_list.lock().unwrap();

                    waiting_list.push(sender);
                }
            }

            // if the batch is full, then do not gather more
            if features_list.len() >= batch_size {
                break;
            }

            // if all clients are either waiting, or already in the batch then
            // terminate with a partial batch.
            let accounted_for = {
                let waiting_list = waiting_list.lock().unwrap();

                features_list.len() + waiting_list.len()
            };

            if accounted_for == worker_count {
                break;
            }
        }

        assert_eq!(features_list.len(), sender_list.len());

        // dispatch the gathered batch to a neural network worker in the
        // thread-pool.
        if !features_list.is_empty() {
            let network = network.clone();
            let waiting_list = waiting_list.clone();
            let pending_count = pending_count.clone();

            pending_count.fetch_add(1, Ordering::Release);
            pool.execute(move || {
                let mut workspace = network.get_workspace(features_list.len());
                let (value_list, policy_list) = if workspace.is_int8() {
                    forward_i8(&mut workspace, features_list)
                } else if workspace.is_half() {
                    forward_f16(&mut workspace, features_list)
                } else {
                    forward_f32(&mut workspace, features_list)
                };
                let response_iter = value_list.into_iter().zip(policy_list.into_iter());

                for (sender, response) in sender_list.into_iter().zip(response_iter) {
                    sender.send(response).unwrap();
                }

                // signal that there is new data available, so all sleeping
                // threads should wake up.
                let mut waiting_list = waiting_list.lock().unwrap();
                let count = waiting_list.len();

                for waiting in waiting_list.drain(0..count) {
                    waiting.send(()).unwrap();
                }

                pending_count.fetch_sub(1, Ordering::Release);
            });
        } else if pending_count.load(Ordering::Acquire) == 0 {
            // everyone are asleep? this is probably some race condition between the
            // branches being picked and the channels actually being sent out.
            let mut waiting_list = waiting_list.lock().unwrap();
            let count = waiting_list.len();

            for waiting in waiting_list.drain(0..count) {
                waiting.send(()).unwrap();
            }
        }
    }

    // wait for all of the neural network workers to terminate
    pool.join();

    // signal that the server is no longer running
    let &(ref lock, ref cvar) = &*is_running;
    let mut is_running = lock.lock().unwrap();
    *is_running = false;

    cvar.notify_all();
}

#[derive(Clone)]
pub struct Server {
    handle: Option<Arc<thread::JoinHandle<()>>>,
    sender: Sender<PredictRequest>,
    is_running: Arc<(Mutex<bool>, Condvar)>,
    is_half: bool,
    is_int8: bool
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            if let Ok(handle) = Arc::try_unwrap(handle) {
                self.sender.send(PredictRequest::Shutdown).unwrap();
                handle.join().unwrap();
            }
        }
    }
}

impl Server {
    /// Start-up a server thread that computes the value and policy from
    /// feature lists.
    /// 
    /// # Arguments
    /// 
    /// * `network` -
    /// 
    pub fn new<C: Param>(network: Network) -> Server {
        let (sender, receiver) = channel();
        let is_running = Arc::new((Mutex::new(true), Condvar::new()));
        let is_half = network.is_half();
        let is_int8 = network.is_int8();
        let handle = {
            let is_running = is_running.clone();

            thread::spawn(move || server_aux::<C>(network, receiver, is_running))
        };

        Server {
            handle: Some(Arc::new(handle)),
            sender: sender,
            is_running: is_running,
            is_half: is_half,
            is_int8: is_int8
        }
    }

    /// Waits until this server terminate.
    pub fn join(this: Server) {
        let is_running = this.is_running.clone();

        // get ride of our reference to the server
        drop(this);

        // wait until all other references has been dropped, which should happend
        // _soon_.
        let &(ref lock, ref cvar) = &*is_running;
        let mut is_running = lock.lock().unwrap();

        while *is_running {
            is_running = cvar.wait(is_running).unwrap();
        }
    }

    /// Acquire an exclusive endpoint to this server. Each endpoint **must**
    /// be used or you might encounter a deadlock as the server waits for
    /// a full batch to be available.
    pub fn lock<'a>(&'a self) -> ServerGuard<'a> {
        ServerGuard::new(self)
    }
}

pub struct ServerGuard<'a> {
    sender: Sender<PredictRequest>,
    is_half: bool,
    is_int8: bool,

    lifetime: ::std::marker::PhantomData<&'a usize>
}

impl<'a> Clone for ServerGuard<'a> {
    fn clone(self: &ServerGuard<'a>) -> ServerGuard<'a> {
        self.sender.send(PredictRequest::Increase).unwrap();

        ServerGuard {
            sender: self.sender.clone(),
            is_half: self.is_half,
            is_int8: self.is_int8,

            lifetime: ::std::marker::PhantomData::<&'a usize>::default()
        }
    }
}

impl<'a> Drop for ServerGuard<'a> {
    fn drop(&mut self) {
        self.sender.send(PredictRequest::Decrease).unwrap();
    }
}

impl<'a> ServerGuard<'a> {
    /// Returns a new endpoint to the given server
    pub fn new(server: &'a Server) -> ServerGuard<'a> {
        server.sender.send(PredictRequest::Increase).unwrap();

        ServerGuard {
            sender: server.sender.clone(),
            is_half: server.is_half,
            is_int8: server.is_int8,

            lifetime: ::std::marker::PhantomData::<&'a usize>::default()
        }
    }

    /// Returns true if this server expects input and output in half precision.
    pub fn is_half(&self) -> bool {
        self.is_half
    }

    /// Returns true if this server expects input and output in int8 quantized precision.
    pub fn is_int8(&self) -> bool {
        self.is_int8
    }

    /// Sends a request to the server for computing the value and policy for
    /// some features.
    /// 
    /// # Arguments
    /// 
    /// * `features` -
    /// 
    pub fn send(&self, features: Array) -> (Singleton, Array) {
        let (tx, rx) = channel();

        self.sender.send(PredictRequest::Ask(features, tx)).unwrap();
        rx.recv().unwrap()
    }

    /// Waits for the server to finish some computations.
    pub fn wait(&self) {
        let (tx, rx) = channel();

        self.sender.send(PredictRequest::Wait(tx)).unwrap();
        rx.recv().unwrap();
    }
}
