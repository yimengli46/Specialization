import torch.utils.data as data
import torch
from random import Random

from itertools import islice
from timeit import default_timer as timer


class RandomIterator:
    """Class to implement an iterator
    of powers of two"""

    def __init__(self, num_elems=0, seed=0):
        self.num_elems = num_elems
        self.random = Random(seed)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.num_elems:
            result = self.random.randint(0, 10000)
            self.n += 1
            return result
        else:
            raise StopIteration

# ==================================================================================================


class MyIterableDataset(data.IterableDataset):
    def __init__(self, seed=0):
        super(MyIterableDataset).__init__()
        self.random = Random(seed)
        self.seed = seed
        self.stream = self.get_stream(self.seed)

    def get_stream(self, seed=0):
        randIter = RandomIterator(num_elems=1000, seed=seed)
        return iter(randIter)  # map(self.process_data, iter(randIter))

    def __iter__(self):
        worker_info = data.get_worker_info()
        dataset = worker_info.dataset
        worker_id = worker_info.id
        n_workers = worker_info.num_workers
        # return iter(self.stream)
        return islice(self.stream, worker_id, None, n_workers)


NUM_WORKERS = 1
ds = MyIterableDataset(seed=0)
loader = data.DataLoader(ds, batch_size=5, num_workers=NUM_WORKERS)

t0 = timer()
for batch in islice(loader, 100):
    print(batch)
t1 = timer()
print(f'build map time = {t1 - t0}')


'''
# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id):
	worker_info = data.get_worker_info()
	dataset = worker_info.dataset  # the dataset copy in this worker process

# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
ds = MyIterableDataset()

# Mult-process loading with the custom `worker_init_fn`
# Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
print(list(data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))

# With even more workers
print(list(data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
'''
