# Copyright (c) 2019 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
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

import json
import subprocess
import threading

from queue import Queue

class AgentPool:
    def __init__(self, config_file):
        with open(config_file) as f:
            config = json.load(f)

        self.setup_hooks = [hook['exec'] for hook in config['setup']]
        self.teardown_hooks = [hook['exec'] for hook in config['teardown']]
        self.worker_configs = config['workers']

        for setup_hook in self.setup_hooks:
            subprocess.call(setup_hook, shell=True)

    def __del__(self):
        for teardown_hook in self.teardown_hooks:
            subprocess.call(teardown_hook, shell=True)

    def iter(self, weights):
        workers = [_start_worker(worker_config, weights) for worker_config in self.worker_configs]

        return AgentPoolIter(workers)


class AgentPoolIter:
    def __init__(self, workers):
        self.workers = workers

    def __iter__(self):
        examples = Queue()

        # Consume the worker output, and produce queue entries in background threads
        producers = list([_start_producer(worker, examples) for worker in self.workers])
        remaining = len(self.workers)

        while remaining > 0:
            example = examples.get()

            if example is not None:
                yield example
            else:
                remaining -= 1

        # wait for all threads to terminate (which they should have already done)
        for producer in producers:
            producer.join()


def _start_worker(config, weights):
    from .agent import Agent

    return Agent(config['exec'], weights)


def _start_producer(worker, examples):
    def _run():
        for example in worker:
            if example is not None:
                examples.put(example)
        examples.put(None)

    t = threading.Thread(target=_run)
    t.start()

    return t
