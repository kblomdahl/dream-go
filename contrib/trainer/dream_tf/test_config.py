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

from .config import Config
from .test_common import TestUtils

class ConfigTest(unittest.TestCase, TestUtils):
    def test_is_start(self):
        self.assertEqual(Config(['--start'], exit_on_error=False).is_start(), True)

    def test_is_resume(self):
        self.assertEqual(Config(['--resume'], exit_on_error=False).is_resume(), True)

    def test_is_verify(self):
        self.assertEqual(Config(['--verify'], exit_on_error=False).is_verify(), True)

    def test_is_dump(self):
        self.assertEqual(Config(['--dump'], exit_on_error=False).is_dump(), True)

    def test_model(self):
        self.assertEqual(Config(['--start', '--model', 'x'], exit_on_error=False).get_model_dir(), 'x')

    def test_model_default(self):
        self.assertIsNotNone(Config(['--start'], exit_on_error=False).get_model_dir())

    def test_name(self):
        self.assertIn('primex', Config(['--start', '--name', 'primex'], exit_on_error=False).get_model_dir())

    def test_warm_start(self):
        self.assertEqual(Config(['--start', '--warm-start', 'x'], exit_on_error=False).warm_start, 'x')

    def test_hparams(self):
        self.assertIsNotNone(Config(['--start']).hparams)

if __name__ == '__main__':
    unittest.main()
