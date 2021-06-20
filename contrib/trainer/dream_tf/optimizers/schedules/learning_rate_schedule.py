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

class WarmupExponentialDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Learning rate is first warmed up from 0 to 0.01, and then decayed by
    0.97 every 2.4 epochs. A learning rate schedule inspired by
    EfficientNetV2 [1].

    [1] https://arxiv.org/pdf/2104.00298.pdf """

    def __init__(
        self,
        initial_learning_rate=1e-4,
        max_learning_rate=0.01,
        num_warmup_steps=2500,
        num_decay_steps=240,
        decay_rate=0.97,
    ):
        super(WarmupExponentialDecaySchedule, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.max_learning_rate = max_learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_decay_steps = num_decay_steps
        self.decay_rate = decay_rate

    def warmup_learning_rate(self, step):
        alpha = step / self.num_warmup_steps

        return (1 - alpha) * self.initial_learning_rate + alpha * self.max_learning_rate

    def decay_learning_rate(self, step):
        return self.max_learning_rate * self.decay_rate ** (step / self.num_decay_steps)

    def __call__(self, step):
        return tf.cond(
            step < self.num_warmup_steps,
            true_fn=lambda: self.warmup_learning_rate(step),
            false_fn=lambda: self.decay_learning_rate(step - self.num_warmup_steps)
        )
