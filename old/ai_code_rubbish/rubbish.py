import lmdb
import time
from tqdm import tqdm
lmdb_path = "media/lmdb/offline_with_hybrid"

env = lmdb.open(
    lmdb_path,
    readonly=True,
    lock=False,
    readahead=False,
    max_readers=512
)

txn = env.begin()

N = 5000

start = time.time()

for i in tqdm(range(N)):
    key = f"{i:08d}".encode()
    _ = txn.get(key)

print("LMDB raw get time:", time.time() - start)
print("per sample:", (time.time() - start) / N)