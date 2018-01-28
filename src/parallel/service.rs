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

use std::thread::{self, JoinHandle};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use parallel::*;

/// The implementation details of a service that is responsible for actually
/// answering the requests. A service can be distributed over multiple threads,
/// where all threads shares the same internal state.
/// 
/// Because all worker threads share the same state you must implement interior
/// mutability of the state using, for example, `Mutex` to ensure no two threads
/// tries to mutate the state at the same time.
pub trait ServiceImpl {
    type State: Send + Sync;
    type Request: Send + Sync;
    type Response: Send + Sync;

    /// Returns the recommended number of threads that this service should
    /// be allocated. This number can be overridden during service creation by
    /// the user.
    fn get_thread_count() -> usize;

    /// Process a single request to this service. The response to the request
    /// should be send over the `resp` channel.
    /// 
    /// # Arguments
    /// 
    /// * `state` - the state of the service
    /// * `req` - the request to process
    /// * `resp` - the channel to send the response to
    /// * `has_more` - whether there are no requests immedietly available after
    ///   this one.
    /// 
    fn process(
        state: &Self::State,
        req: Self::Request,
        resp: OneSender<Self::Response>,
        has_more: bool
    );
}

/// The worker thread that is responsible for receiving requests and dispatching
/// them to the service implementation. This worker will terminate once the
/// variable `is_running` is set to false and there are no pending requests.
/// 
/// # Arguments
/// 
/// * `is_running` - whether the service should be running
/// * `state` - the state of the service
/// * `queue` - the queue of requests
/// 
fn worker_thread<I: ServiceImpl>(
    is_running: Arc<AtomicBool>,
    state: Arc<I::State>,
    queue: Arc<(Mutex<Vec<(I::Request, OneSender<I::Response>)>>, Condvar)>
) {
    let (ref queue, ref cvar) = *queue;
    let mut queue_lock = queue.lock().unwrap();

    loop {
        if let Some((req, tx)) = queue_lock.pop() {
            let has_more = !queue_lock.is_empty();
            drop(queue_lock);

            I::process(&state, req, tx, has_more);

            queue_lock = queue.lock().unwrap();
        } else {
            if !is_running.load(Ordering::Acquire) {
                break
            }

            queue_lock = cvar.wait(queue_lock).unwrap();
        }
    }
}

/// A service that off-load and balance the processing of requests over
/// multiple worker threads.
pub struct Service<I: ServiceImpl> {
    workers: Vec<JoinHandle<()>>,

    state: Arc<I::State>,
    queue: Arc<(Mutex<Vec<(I::Request, OneSender<I::Response>)>>, Condvar)>,
    is_running: Arc<AtomicBool>
}

impl<I: ServiceImpl> ::std::ops::Deref for Service<I> {
    type Target = I::State;

    fn deref(&self) -> &Self::Target {
        &*self.state
    }
}

impl<I: ServiceImpl> Drop for Service<I> {
    fn drop(&mut self) {
        let queue_lock = self.queue.0.lock();

        if self.is_running.compare_and_swap(true, false, Ordering::AcqRel) {
            self.queue.1.notify_all();
            drop(queue_lock);

            let num_workers = self.workers.len();

            for handle in self.workers.drain(0..num_workers) {
                handle.join().unwrap();
            }
        }
    }
}

impl<I: ServiceImpl + 'static> Service<I> {
    /// Returns a new `Service` with the given number of working threads (or
    /// the default given by the service if `None`) and initial state.
    /// 
    /// # Arguments
    /// 
    /// * `num_threads` -
    /// * `initial_state` -
    /// 
    pub fn new(num_threads: Option<usize>, initial_state: I::State) -> Service<I> {
        let state = Arc::new(initial_state);
        let queue = Arc::new((Mutex::new(vec! []), Condvar::new()));
        let is_running = Arc::new(AtomicBool::new(true));
        let num_threads = num_threads.unwrap_or_else(|| I::get_thread_count());

        Service {
            workers: (0..num_threads)
                .map(|_| {
                    let state = state.clone();
                    let queue = queue.clone();
                    let is_running = is_running.clone();

                    thread::spawn(move || worker_thread::<I>(is_running, state, queue))
                })
                .collect(),

            state: state,
            queue: queue,
            is_running: is_running
        }
    }

    /// Acquire an endpoint that can be used to communicate with the
    /// service.
    pub fn lock<'a>(&'a self) -> ServiceGuard<'a, I> {
        ServiceGuard {
            _owner: ::std::marker::PhantomData::default(),

            state: self.state.clone(),
            queue: self.queue.clone(),
            is_running: self.is_running.clone()
        }
    }
}

#[derive(Clone)]
pub struct ServiceGuard<'a, I: ServiceImpl> {
    _owner: ::std::marker::PhantomData<&'a ()>,

    state: Arc<I::State>,
    queue: Arc<(Mutex<Vec<(I::Request, OneSender<I::Response>)>>, Condvar)>,
    is_running: Arc<AtomicBool>
}

impl<'a, I: ServiceImpl + 'static> ServiceGuard<'a, I> {
    /// Returns the current state of this service.
    pub fn get_state<'b>(&'b self) -> &'b I::State {
        &*self.state
    }

    /// Sends a request to the service and returns the response.
    /// 
    /// # Arguments
    /// 
    /// * `req` -
    /// 
    pub fn send(&self, req: I::Request) -> I::Response {
        let (tx, rx) = one_channel();

        if let Ok(mut queue_lock) = self.queue.0.lock() {
            queue_lock.push((req, tx));
            self.queue.1.notify_one();

            // get ride of the lock so that one of the service workers
            // can acquire it
            drop(queue_lock);

            // wait for a response
            OneReceiver::recv(rx).unwrap()
        } else {
            panic!("Service is unavailable");
        }
    }

    /// Returns a clone of this guard with a `'static` lifetime. This
    /// is useful for transferring a guard across the thread boundary.
    pub fn clone_static(&self) -> ServiceGuard<'static, I> {
        ServiceGuard {
            _owner: ::std::marker::PhantomData::default(),

            state: self.state.clone(),
            queue: self.queue.clone(),
            is_running: self.is_running.clone()
        }
    }
}
