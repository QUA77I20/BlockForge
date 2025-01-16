"""
Copyright © 2025 Marc All rights reserved.
This code is part of the "Blockchain-Project" and is protected by international copyright laws.
Unauthorized use, reproduction, or distribution of this code is strictly prohibited.
"""

class Consensus:
    def proof_of_work(self, block, difficulty):
        target = "0" * difficulty
        while not block.hash.startswith(target):
            block.nonce += 1
            block.hash = block.calculate_hash()
        return block.hash
