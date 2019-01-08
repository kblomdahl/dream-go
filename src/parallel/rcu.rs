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

use super::one_shot_channel::{one_channel, OneSender, OneReceiver};

use std::sync::atomic::{Ordering, fence};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::sync::Mutex;
use std::collections::VecDeque;
use std::thread;

trait FnBox {
    fn call_box(self: Box<Self>);
}

impl<F: FnOnce()> FnBox for F {
    fn call_box(self: Box<F>) {
        (*self)()
    }
}

enum RcuSignal {
    Register(OneSender<()>),
    Unregister,
    Quiescent(OneSender<()>),
    Update(Box<FnBox + Send + 'static>, OneSender<()>)
}

lazy_static! {
    static ref SENDER: Mutex<Sender<RcuSignal>> = {
        let (sx, rx) = channel();

        thread::Builder::new()
            .name("rcu_worker".into())
            .spawn(move || run_worker_thread(rx))
            .unwrap();

        Mutex::new(sx)
    };
}

fn run_worker_thread(rx: Receiver<RcuSignal>) {
    let mut pending_updates: VecDeque<Box<FnBox + Send + 'static>> = VecDeque::new();
    let mut quiescent_threads = vec! [];
    let mut num_threads = 0;

    while let Ok(m) = rx.recv() {
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
        if quiescent_threads.len() == num_threads || num_threads == 0 {
            for f in pending_updates.drain(0..) {
                f.call_box();
            }
        }

        if pending_updates.is_empty() {
            for rc in quiescent_threads.drain(0..) {
                rc.send(());
            }
        }
    }
}

/// Execute the given function inside of an RCU _write-lock_.
pub fn update<F: FnOnce() + Send + 'static>(f: F) {
    fence(Ordering::SeqCst);

    let (tx, rx) = one_channel();

    SENDER.lock().unwrap().send(RcuSignal::Update(Box::new(f), tx)).unwrap();
    OneReceiver::recv(rx).unwrap()
}

/// Signal to the RCU implementation that this is a _safe_ spot to execute
/// update operations, as the current thread does not hold any protected
/// data.
pub fn quiescent_state() {
    fence(Ordering::SeqCst);

    // signal to helper thread that we are in a quiescent state, and then wait
    // for the _good to go_ signal
    let (tx, rx) = one_channel();

    SENDER.lock().unwrap().send(RcuSignal::Quiescent(tx)).unwrap();
    OneReceiver::recv(rx).unwrap()
}

/// Signal to the RCU implementation that this thread will access RCU
/// protected fields.
pub fn register_thread() {
    fence(Ordering::SeqCst);

    // register, and wait for any running update to finish
    let (tx, rx) = one_channel();

    SENDER.lock().unwrap().send(RcuSignal::Register(tx)).unwrap();
    OneReceiver::recv(rx).unwrap()
}

/// Signal to the RCU implementation that this thread will no longer access
/// RCU protected fields.
pub fn unregister_thread() {
    SENDER.lock().unwrap().send(RcuSignal::Unregister).unwrap();
}
