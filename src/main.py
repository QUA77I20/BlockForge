"""
Copyright © 2025 Marc All rights reserved.
This code is part of the "Blockchain-Project" and is protected by international copyright laws.
Unauthorized use, reproduction, or distribution of this code is strictly prohibited.
"""

from blockchain import Blockchain
from transaction import Transaction
from fastapi.staticfiles import StaticFiles

# Подключаем статические файлы (интерфейс)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the blockchain
my_blockchain = Blockchain(difficulty=4)

# Add transactions
my_blockchain.add_transaction(Transaction("Marc", "Chris", 100).to_dict())
my_blockchain.add_transaction(Transaction("Marc", "Sonya", 50).to_dict())

# Mine the block with unconfirmed transactions
print("⛏️ Mining block...")
my_blockchain.mine()

# Display the blockchain
print("\n📋 Current Blockchain:")
for block in my_blockchain.chain:
    print(f"Index: {block.index}, Hash: {block.hash}, Prev Hash: {block.previous_hash}, Data: {block.data}")

# Validate the blockchain
if my_blockchain.is_chain_valid():
    print("\n✅ Blockchain is valid!")
else:
    print("\n❌ Blockchain is NOT valid!")
