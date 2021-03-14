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

use std::sync::{Arc, Mutex, MutexGuard};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use crossbeam_channel::{bounded, Sender};
use crossbeam_queue::SegQueue;

/// The implementation details of a service that is responsible for actually
/// answering the requests. A service can be distributed over multiple threads,
/// where all threads shares the same internal state.
/// 
/// Because all worker threads share the same state you must implement interior
/// mutability of the state using, for example, `Mutex` to ensure no two threads
/// tries to mutate the state at the same time.
pub trait ServiceImpl {
    type State: Send;
    type Request: Send + Sync;
    type Response: Send + Sync;

    /// Returns the recommended number of threads that this service should
    /// be allocated. This number can be overridden during service creation by
    /// the user.
    fn get_thread_count() -> usize;

    /// Setup the initial state of the current thread. This is the first
    /// function called for each worker thread.
    /// 
    /// # Arguments
    /// 
    /// * `index` -
    /// 
    fn setup_thread(index: usize);

    /// Process a single request to this service. The response to the request
    /// should be send over the `resp` channel.
    /// 
    /// # Arguments
    /// 
    /// * `state` - the state of the service
    /// * `state_lock` - the state of the service (acquired lock)
    /// * `req` - the request to process
    /// * `resp` - the channel to send the response to
    /// * `has_more` - whether there are no requests immedietly available after
    ///   this one. This number is only valid for the lifetime of `state_lock`.
    /// 
    fn process(
        state: &Mutex<Self::State>,
        state_lock: MutexGuard<Self::State>,
        req: Self::Request,
        resp: Sender<Self::Response>,
        has_more: bool
    );
}

/// The interior state of a service
pub struct ServiceState<I: ServiceImpl> {
    /// Whether this service should still be running.
    is_running: AtomicBool,

    /// The number of requests currently being processed.
    num_process: AtomicUsize,

    /// The queue of pending requests, and the channel to send the response
    /// over.
    queue: SegQueue<(I::Request, Sender<I::Response>)>
}

/// The worker thread that is responsible for receiving requests and dispatching
/// them to the service implementation. This worker will terminate once the
/// variable `is_running` is set to false and there are no pending requests.
/// 
/// # Arguments
/// 
/// * `num_running` - the number of requests currently being processed
/// * `is_running` - whether the service should be running
/// * `state` - the state of the service
/// * `queue` - the queue of requests
/// 
fn worker_thread<I: ServiceImpl>(
    state: Arc<Mutex<I::State>>,
    inner: Arc<ServiceState<I>>
) {
    loop {
        if let Some((req, tx)) = inner.queue.pop() {
            let state_lock = state.lock().unwrap();
            let has_more = !inner.queue.is_empty();
            inner.num_process.fetch_add(1, Ordering::AcqRel);

            I::process(&state, state_lock, req, tx, has_more);

            inner.num_process.fetch_sub(1, Ordering::AcqRel);
        } else if !inner.is_running.load(Ordering::Acquire) {
            break
        }
    }
}

/// A service that off-load and balance the processing of requests over
/// multiple worker threads.
pub struct Service<I: ServiceImpl> {
    workers: Vec<JoinHandle<()>>,

    state: Arc<Mutex<I::State>>,
    inner: Arc<ServiceState<I>>
}

impl<I: ServiceImpl> ::std::ops::Deref for Service<I> {
    type Target = Mutex<I::State>;

    fn deref(&self) -> &Self::Target {
        &*self.state
    }
}

impl<I: ServiceImpl> Drop for Service<I> {
    fn drop(&mut self) {
        if let Ok(_) = self.inner.is_running.compare_exchange_weak(true, false, Ordering::SeqCst, Ordering::Relaxed) {
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
        let state = Arc::new(Mutex::new(initial_state));
        let inner = Arc::new(ServiceState {
            num_process: AtomicUsize::new(0),
            queue: SegQueue::new(),
            is_running: AtomicBool::new(true)
        });
        let num_threads = num_threads.unwrap_or_else(I::get_thread_count);

        Service {
            workers: (0..num_threads)
                .map(|i| {
                    let inner = inner.clone();
                    let state = state.clone();

                    thread::Builder::new()
                        .name("service_worker".into())
                        .spawn(move || {
                            I::setup_thread(i);

                            worker_thread::<I>(state, inner)
                        })
                        .unwrap()
                })
                .collect(),

            state: state,
            inner: inner
        }
    }

    /// Acquire an endpoint that can be used to communicate with the
    /// service.
    pub fn lock(&self) -> ServiceGuard<I> {
        ServiceGuard {
            _owner: ::std::marker::PhantomData::default(),

            state: self.state.clone(),
            inner: self.inner.clone()
        }
    }
}

pub struct ServiceGuard<'a, I: ServiceImpl> {
    _owner: ::std::marker::PhantomData<&'a ()>,

    state: Arc<Mutex<I::State>>,
    inner: Arc<ServiceState<I>>
}

impl<'a, I: ServiceImpl + 'static> ServiceGuard<'a, I> {
    /// Returns the current state of this service.
    pub fn get_state(&self) -> MutexGuard<I::State> {
        self.state.lock().unwrap()
    }

    /// Sends a request to the service and returns the response.
    /// 
    /// # Arguments
    /// 
    /// * `req` -
    /// 
    pub fn send(&self, req: I::Request) -> Option<I::Response> {
        let (tx, rx) = bounded(1);

        if self.inner.is_running.load(Ordering::Acquire) {
            self.inner.queue.push((req, tx));

            // wait for a response
            rx.recv().ok()
        } else {
            None
        }
    }

    /// Send all of the given requests to the service and returns the
    /// responses.
    /// 
    /// # Arguments
    /// 
    /// * `reqs` -
    /// 
    pub fn send_all<E: Iterator<Item=I::Request>>(&self, reqs: E) -> Option<Vec<I::Response>> {
        if self.inner.is_running.load(Ordering::Acquire) {
            let responses = reqs.map(|req| {
                let (tx, rx) = bounded(1);

                self.inner.queue.push((req, tx));
                rx
            }).collect::<Vec<_>>();

            // wait for all of the responses
            responses.into_iter().map(|rx| rx.recv().ok()).collect()
        } else {
            None
        }
    }

    /// Returns a clone of this guard with a `'static` lifetime. This
    /// is useful for transferring a guard across the thread boundary.
    pub fn clone_to_static(&self) -> ServiceGuard<'static, I> {
        ServiceGuard {
            _owner: ::std::marker::PhantomData::default(),

            state: self.state.clone(),
            inner: self.inner.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use test::Bencher;
    use super::*;

    struct FakeServiceImpl;

    impl ServiceImpl for FakeServiceImpl {
        type State = ();
        type Request = i32;
        type Response = i32;

        fn get_thread_count() -> usize {
            4
        }

        fn setup_thread(_index: usize) {
            // pass
        }

        fn process(_state: &Mutex<Self::State>, _state_lock: MutexGuard<Self::State>, req: Self::Request, resp: Sender<Self::Response>, _has_more: bool) {
            resp.send(2 * req).unwrap();
        }
    }

    #[test]
    fn check_double_service() {
        let double: Service<FakeServiceImpl> = Service::new(None, ());
        let double_lock = double.lock();

        assert_eq!(double_lock.send( 0), Some( 0));
        assert_eq!(double_lock.send( 3), Some( 6));
        assert_eq!(double_lock.send(10), Some(20));
        assert_eq!(double_lock.send(15), Some(30));
        assert_eq!(double_lock.send(33), Some(66));
    }

    struct NoServiceImpl;

    impl ServiceImpl for NoServiceImpl {
        type State = ();
        type Request = ();
        type Response = ();

        fn get_thread_count() -> usize {
            1
        }

        fn setup_thread(_index: usize) {
            // pass
        }

        fn process(_state: &Mutex<Self::State>, _state_lock: MutexGuard<Self::State>, _req: Self::Request, resp: Sender<Self::Response>, _has_more: bool) {
            resp.send(()).unwrap();
        }
    }

    #[bench]
    fn bench_service(b: &mut Bencher) {
        let no: Service<NoServiceImpl> = Service::new(None, ());
        let no_lock = no.lock();

        b.iter(move || {
            no_lock.send(()).unwrap();
        })
    }
}
