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

import unittest

from .learning_rate_schedule import WarmupExponentialDecaySchedule
from ...test_common import TestUtils

class WarmupExponentialDecayScheduleTest(unittest.TestCase, TestUtils):
    def setUp(self):
        self.initial_learning_rate = 1e-3
        self.max_learning_rate = 0.256
        self.num_warmup_steps = 2500
        self.learning_rate = WarmupExponentialDecaySchedule(
            initial_learning_rate=self.initial_learning_rate,
            max_learning_rate=self.max_learning_rate,
            num_warmup_steps=self.num_warmup_steps,
            num_decay_steps=240,
            decay_rate=0.97
        )

    def test_warmup(self):
        for step in range(self.num_warmup_steps - 1):
            lr = self.learning_rate(step)
            next_lr = self.learning_rate(step + 1)

            self.assertLess(lr, next_lr)
            self.assertGreaterEqual(lr, self.initial_learning_rate)
            self.assertLessEqual(lr, self.max_learning_rate)

    def test_decay(self):
        for step in range(1000):
            lr = self.learning_rate(self.num_warmup_steps + step)
            next_lr = self.learning_rate(self.num_warmup_steps + step + 1)

            self.assertGreater(lr, next_lr)
            self.assertLessEqual(lr, self.max_learning_rate)

if __name__ == '__main__':
    unittest.main()
