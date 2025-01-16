"""
Copyright © 2025 Marc All rights reserved.
This code is part of the "Blockchain-Project" and is protected by international copyright laws.
Unauthorized use, reproduction, or distribution of this code is strictly prohibited.
"""
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from blockchain import Blockchain
import time
from utils import sha256_hash

class Miner:
    def __init__(self, difficulty):
        self.difficulty = difficulty

    def mine_block(self, block):
        target = "0" * self.difficulty
        while not block.hash.startswith(target):
            block.nonce += 1
            block.hash = sha256_hash(block.__dict__)
        print(f"Block mined! Hash: {block.hash}")
        return block

# GPU Kernel для Proof-of-Work
mod = SourceModule("""
__global__ void mine_block(char *block_data, unsigned long long *nonce, int difficulty, int *found) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    char hash_result[64];
    int target = difficulty;

    for (unsigned long long i = idx; i < idx + 1000000000; i++) {
        // Создаем хэш-функцию (упрощенная версия)
        sprintf(hash_result, "%llu%s", i, block_data);

        // Проверяем на сложность
        int zero_count = 0;
        for (int j = 0; j < 64; j++) {
            if (hash_result[j] == '0') {
                zero_count++;
            } else {
                break;
            }
        }

        if (zero_count >= target) {
            *nonce = i;
            *found = 1;
            break;
        }
    }
}
""")

mine_block = mod.get_function("mine_block")

# Функция для майнинга на GPU
def gpu_mine(block_data, difficulty):
    nonce = np.zeros(1, dtype=np.uint64)
    found = np.zeros(1, dtype=np.int32)

    block_data_gpu = cuda.mem_alloc(len(block_data))
    nonce_gpu = cuda.mem_alloc(nonce.nbytes)
    found_gpu = cuda.mem_alloc(found.nbytes)

    cuda.memcpy_htod(block_data_gpu, block_data.encode())
    cuda.memcpy_htod(nonce_gpu, nonce)
    cuda.memcpy_htod(found_gpu, found)

    mine_block(block_data_gpu, nonce_gpu, np.int32(difficulty), found_gpu, block=(256, 1, 1), grid=(1024, 1))

    cuda.memcpy_dtoh(nonce, nonce_gpu)
    cuda.memcpy_dtoh(found, found_gpu)

    if found[0]:
        print(f"✅ Block mined! Nonce: {nonce[0]}")
        return nonce[0]
    else:
        print("❌ Mining failed.")
        return None

# Тест
if __name__ == "__main__":
    blockchain = Blockchain()
    block_data = "Marc's Block"
    difficulty = 5  # Количество нулей в начале хэша
    gpu_mine(block_data, difficulty)
