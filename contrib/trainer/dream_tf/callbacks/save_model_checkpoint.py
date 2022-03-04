# Copyright (c) 2021 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf

class CustomSaveModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model_dir,
        *,
        monitor='val_loss',
        save_best_only=True
    ):
        super(CustomSaveModelCheckpoint, self).__init__()

        self.model_dir = model_dir
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_monitor_value = None

    def on_epoch_end(self, epoch, logs):
        if self.save_best_only:
            if self.best_monitor_value is not None and logs[self.monitor] > self.best_monitor_value:
                return

            self.best_monitor_value = logs[self.monitor]

        self.model.save_weights(f'{self.model_dir}/weights.{epoch:03d}.h5', save_format='h5')
