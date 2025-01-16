"""
Copyright © 2025 Marc All rights reserved.
This code is part of the "Blockchain-Project" and is protected by international copyright laws.
Unauthorized use, reproduction, or distribution of this code is strictly prohibited.
"""
from utils import sha256_hash
import hashlib
import time

MINING_REWARD = 50  # 💰 Reward for mining a block

# Block class to represent each block in the blockchain
class Block:
    def __init__(self, index, previous_hash, transactions, timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.transactions}{self.timestamp}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

# Blockchain class to manage the entire chain of blocks
# Blockchain class to manage the chain of blocks
class Blockchain:
    def __init__(self, difficulty=4):
        self.chain = [self.create_genesis_block()]
        self.unconfirmed_transactions = []
        self.balances = {}
        self.difficulty = difficulty

    def create_genesis_block(self):
        return Block(0, "0", [], time.time())

    def get_latest_block(self):
        return self.chain[-1]

    def add_transaction(self, transaction):
        sender = transaction.get("sender")
        receiver = transaction.get("receiver")
        amount = transaction.get("amount")

        if sender != "MINING_REWARD":
            if self.balances.get(sender, 0) < amount:
                raise ValueError("Insufficient balance for transaction.")

        self.unconfirmed_transactions.append(transaction)
        self.balances[receiver] = self.balances.get(receiver, 0) + amount
        if sender != "MINING_REWARD":
            self.balances[sender] -= amount

    def mine(self, miner_address):
        if not self.unconfirmed_transactions:
            return {"message": "No transactions to mine."}

        last_block = self.get_latest_block()

        # Добавляем награду за майнинг
        reward_transaction = {
            "sender": "NETWORK_REWARD",
            "receiver": miner_address,
            "amount": 50  # фиксированная награда за майнинг
        }
        self.unconfirmed_transactions.insert(0, reward_transaction)  # награду ставим первой

        # Создаем новый блок
        new_block = Block(
            index=last_block.index + 1,
            previous_hash=last_block.hash,
            transactions=self.unconfirmed_transactions
        )

        # Выполняем PoW (Proof of Work)
        proof = self.proof_of_work(new_block)
        new_block.nonce = proof

        # Добавляем блок в цепочку
        self.chain.append(new_block)

        # Обновляем баланс майнера
        if miner_address not in self.balances:
            self.balances[miner_address] = 0
        self.balances[miner_address] += 50  # обновляем баланс

        # Очищаем список неподтвержденных транзакций
        self.unconfirmed_transactions = []

        return {"message": f"Block #{new_block.index} mined successfully!", "block_index": new_block.index}

    def mine_block(self, block):
        target = "0" * self.difficulty
        while not block.hash.startswith(target):
            block.nonce += 1
            block.hash = block.calculate_hash()

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                print(f"❌ Block {current_block.index} has an invalid hash!")
                return False

            if current_block.previous_hash != previous_block.hash:
                print(f"❌ Block {current_block.index} has an invalid previous hash!")
                return False

        print("✅ Blockchain is valid!")
        return True

    def proof_of_work(self, block):
        block.nonce = 0
        computed_hash = block.calculate_hash()
        while not computed_hash.startswith('0' * self.difficulty):
            block.nonce += 1
            computed_hash = block.calculate_hash()
        return computed_hash

    def issue_initial_balance(self, address, amount):
        if address not in self.balances:
            self.balances[address] = 0
        self.balances[address] += amount
        print(f"💰 Initial balance of {amount} coins issued to {address}")