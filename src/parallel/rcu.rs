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

use super::filo_queue::FiloQueue;
use super::one_shot_channel::{one_channel, OneSender, OneReceiver};

use std::sync::atomic::{Ordering, fence};
use std::collections::VecDeque;
use std::thread;
use std::sync::Arc;

trait FnBox {
    fn call_box(self: Box<Self>);
}

impl<F: FnOnce()> FnBox for F {
    fn call_box(self: Box<F>) {
        (*self)()
    }
}

enum RcuSignal {
    Register(OneSender<bool>),
    Unregister,
    Quiescent(OneSender<bool>),
    Update(Box<FnBox + Send + 'static>, OneSender<bool>)
}

lazy_static! {
    static ref SENDER: Arc<FiloQueue<RcuSignal>> = {
        let q = Arc::new(FiloQueue::default());
        let q_ = q.clone();

        thread::Builder::new()
            .name("rcu_worker".into())
            .spawn(|| run_worker_thread(q_))
            .unwrap();

        q
    };
}

fn run_worker_thread(q: Arc<FiloQueue<RcuSignal>>) {
    let mut pending_updates: VecDeque<Box<FnBox + Send + 'static>> = VecDeque::new();
    let mut quiescent_threads = vec! [];
    let mut num_threads = 0;

    loop {
        if let Some(m) = q.pop() {
            match m {
                RcuSignal::Register(reply_channel) => {
                    quiescent_threads.push(reply_channel);
                    num_threads += 1;
                },
                RcuSignal::Unregister => {
                    num_threads -= 1;
                },
                RcuSignal::Quiescent(reply_channel) => {
                    quiescent_threads.push(reply_channel);
                },
                RcuSignal::Update(f, reply_channel) => {
                    pending_updates.push_back(f);
                    quiescent_threads.push(reply_channel);
                }
            };

            // allow update operations if there are no registered read-lock threads
            let changed = if quiescent_threads.len() == num_threads || num_threads == 0 {
                let has_updates = !pending_updates.is_empty();

                if has_updates {
                    fence(Ordering::SeqCst);
                }

                for f in pending_updates.drain(0..) {
                    f.call_box();
                }

                has_updates
            } else {
                false
            };

            if pending_updates.is_empty() {
                for rc in quiescent_threads.drain(0..) {
                    rc.send(changed);
                }
            }
        } else {
            thread::yield_now();
        }
    }
}

/// Execute the given function inside of an RCU _write-lock_.
pub fn update<F: FnOnce() + Send + 'static>(f: F) {
    let (tx, rx) = one_channel();

    SENDER.push(RcuSignal::Update(Box::new(f), tx));
    if OneReceiver::recv(rx).unwrap() {
        fence(Ordering::SeqCst);
    }
}

/// Signal to the RCU implementation that this is a _safe_ spot to execute
/// update operations, as the current thread does not hold any protected
/// data.
pub fn quiescent_state() {
    // signal to helper thread that we are in a quiescent state, and then wait
    // for the _good to go_ signal
    let (tx, rx) = one_channel();

    SENDER.push(RcuSignal::Quiescent(tx));
    if OneReceiver::recv(rx).unwrap() {
        fence(Ordering::SeqCst);
    }
}

/// Signal to the RCU implementation that this thread will access RCU
/// protected fields.
pub fn register_thread() {
    // register, and wait for any running update to finish
    let (tx, rx) = one_channel();

    SENDER.push(RcuSignal::Register(tx));
    if OneReceiver::recv(rx).unwrap() {
        fence(Ordering::SeqCst);
    }
}

/// Signal to the RCU implementation that this thread will no longer access
/// RCU protected fields.
pub fn unregister_thread() {
    SENDER.push(RcuSignal::Unregister);
}
