import pycuda.autoinit
import numpy as np
import requests
import hashlib
import time
from pycuda.compiler import SourceModule

# GPU kernel function to calculate hash
kernel_code = """
__global__ void mine_kernel(char *data, unsigned long long *nonce, char *target, int *found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (*found) return;

    char hash[64];
    unsigned long long local_nonce = *nonce + idx;

    // Simulate hashing (replace with actual hash logic)
    for (int i = 0; i < 64; i++) {
        hash[i] = data[i] ^ (local_nonce & 0xFF);
    }

    // Compare hash to target (simplified)
    if (hash[0] == target[0] && hash[1] == target[1]) {
        *found = 1;
        *nonce = local_nonce;
    }
}
"""

# Initialize CUDA module
mod = SourceModule(kernel_code)
mine_kernel = mod.get_function("mine_kernel")

def calculate_hash(data, nonce):
    data_with_nonce = data + str(nonce).encode()
    return hashlib.sha256(data_with_nonce).hexdigest()

def gpu_mine(data, target):
    data_gpu = cuda.mem_alloc(len(data))
    nonce_gpu = cuda.mem_alloc(8)
    target_gpu = cuda.mem_alloc(len(target))
    found_gpu = cuda.mem_alloc(4)

    cuda.memcpy_htod(data_gpu, data)
    cuda.memcpy_htod(target_gpu, target)

    block_size = 256
    grid_size = 1024

    nonce = np.zeros(1, dtype=np.uint64)
    found = np.zeros(1, dtype=np.int32)

    while not found[0]:
        cuda.memcpy_htod(nonce_gpu, nonce)
        mine_kernel(data_gpu, nonce_gpu, target_gpu, found_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))
        cuda.memcpy_dtoh(found, found_gpu)
        nonce[0] += block_size * grid_size

    cuda.memcpy_dtoh(nonce, nonce_gpu)
    return nonce[0]

def send_mined_block(api_url, miner_address, nonce):
    response = requests.post(f"{api_url}/mine/{miner_address}", json={"nonce": nonce})
    return response.json()

if __name__ == "__main__":
    api_url = "http://127.0.0.1:8080"
    miner_address = "f14a04aa277028a8da6cf506832a252c9c0f8e6d1f7dc0e6534110d36693d5de"

    print("Starting GPU miner...")

    while True:
        data = b"block_data_placeholder"
        target = b"00"  # Example target

        nonce = gpu_mine(data, target)
        result = send_mined_block(api_url, miner_address, nonce)

        print(f"Block mined: {result}")
        time.sleep(1)
