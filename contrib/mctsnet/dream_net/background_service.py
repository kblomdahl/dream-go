from threading import Thread, Condition

class OneShotChannel:
    def __init__(self):
        self.cond_var = Condition()
        self.value = None

    def store(self, value):
        with self.cond_var:
            self.value = value
            self.cond_var.notify()

    def wait(self):
        with self.cond_var:
            while self.value is None:
                self.cond_var.wait()

            return self.value

class BackgroundService:
    def __init__(self, func, batch_size):
        self.batch_size = batch_size
        self.func = func
        self.tasks = []
        self.cond_var = Condition()
        self.thread = Thread(target=lambda: self._run(), daemon=True)
        self.thread.start()

    def _get_batch(self):
        with self.cond_var:
            while len(self.tasks) < self.batch_size:
                self.cond_var.wait()

                print(self.func, len(self.tasks))

            batch = self.tasks[:self.batch_size]
            self.tasks = self.tasks[self.batch_size:]

            return batch

    def _run(self):
        while True:
            batch = self._get_batch()
            results = self.func([features for _ch, features in batch])

            for i, (ch, _features) in enumerate(batch):
                ch.store(results[i])

    def __call__(self, task):
        channel = OneShotChannel()

        with self.cond_var:
            self.tasks.append((channel, task))
            self.cond_var.notify()

        return channel.wait()
