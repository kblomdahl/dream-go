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

use std::ptr;
use std::sync::{Arc, Mutex, Condvar};

/// The sender part of a one-shot channel.
pub struct OneSender<T> {
    inner: Arc<(Mutex<*mut T>, Condvar)>
}

unsafe impl<T: Send> Send for OneSender<T> {}

impl<T> OneSender<T> {
    fn new(inner: &Arc<(Mutex<*mut T>, Condvar)>) -> OneSender<T> {
        OneSender {
            inner: inner.clone()
        }
    }

    /// Send a message over this channel. This function will panic if no
    /// receiver exist to receive this message.
    /// 
    /// # Arguments
    /// 
    /// * `value` - the message
    /// 
    pub fn send(&self, value: T) {
        let mut ptr = self.inner.0.lock().unwrap();

        debug_assert!((*ptr).is_null());

        if Arc::strong_count(&self.inner) == 1 {
            // no receiver available on the other end, this will
            // always fail
            panic!("No receiver available on the other end");
        } else {
            *ptr = Box::into_raw(Box::new(value));

            // wake up the receiver end of this channel
            self.inner.1.notify_one();
        }
    }
}

/// The receiver part of a one-shot channel.
pub struct OneReceiver<T> {
    inner: Arc<(Mutex<*mut T>, Condvar)>
}

impl<T> OneReceiver<T> {
    fn new(inner: &Arc<(Mutex<*mut T>, Condvar)>) -> OneReceiver<T> {
        OneReceiver {
            inner: inner.clone()
        }
    }

    /// Waits until a message arrives on this channel and then returns that
    /// message. If the sender end of this channel has been dropped, or is while
    /// we are waiting, then `None` is returned.
    /// 
    /// # Arguments
    /// 
    /// * `this` -
    /// 
    pub fn recv(this: OneReceiver<T>) -> Option<T> {
        // wait for the value to become available
        let (ref lock, ref cvar) = *this.inner;
        let mut value = lock.lock().unwrap();

        while value.is_null() {
            if Arc::strong_count(&this.inner) == 1 {
                // we can never receive an answer because the sender has been dropped
                break
            }

            value = cvar.wait(value).unwrap();
        }

        // take ownership of the interior value
        let p = ::std::mem::replace(&mut *value, ptr::null_mut());

        if p.is_null() {
            None
        } else {
            Some(unsafe { *Box::from_raw(p) })
        }
    }
}

/// Returns the sender and receiver end-points for a one-shot channel. This
/// channel can only be used once, but is significantly faster than the standard
/// implementation because of this limitation.
/// 
/// Neither the sender nor the receiver can be cloned, but the sender can be
/// sent over thread boundaries.
pub fn one_channel<T>() -> (OneSender<T>, OneReceiver<T>) {
    let inner = {
        let lock = Mutex::new(ptr::null_mut());
        let cvar = Condvar::new();

        Arc::new((lock, cvar))
    };

    (OneSender::new(&inner), OneReceiver::new(&inner))
}

#[cfg(test)]
mod tests {
    use parallel::*;

    #[test]
    fn send_recv() {
        let (tx, rx) = one_channel();

        tx.send(3.14f32);

        assert_eq!(OneReceiver::recv(rx), Some(3.14f32));
    }

    #[test]
    fn no_deadlock() {
        let (tx, rx) = one_channel::<()>();

        drop(tx);
        assert_eq!(OneReceiver::recv(rx), None);
    }
}