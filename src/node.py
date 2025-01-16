"""
Copyright © 2025 Marc All rights reserved.
Unauthorized use, reproduction, or distribution of this code is strictly prohibited.
"""

from fastapi import FastAPI
from blockchain import Blockchain
from utils import sha256_hash
from wallet import Wallet
from fastapi.responses import HTMLResponse
from cryptography.hazmat.primitives import serialization  # 💥 ВАЖНО: добавили импорт
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
blockchain = Blockchain()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/wallet")
def create_wallet():
    wallet = Wallet()
    return {
        "address": wallet.address,
        "public_key": wallet.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
    }

@app.get("/balance")
def get_balance(address: str):
    balance = blockchain.balances.get(address, 0)
    return {"address": address, "balance": balance}

@app.get("/mine")
def mine_block():
    index = blockchain.mine()
    return {"message": f"Block #{index} mined successfully!"}

# Route to get the entire blockchain
@app.get("/chain")
def get_chain():
    chain_data = [block.__dict__ for block in blockchain.chain]
    return {"length": len(chain_data), "chain": chain_data}

# Route to add a transaction
@app.post("/transaction")
def add_transaction(transaction: dict):
    try:
        blockchain.add_transaction(transaction)
        return {"message": "Transaction added successfully!"}
    except ValueError as e:
        return {"error": str(e)}


# Route to check if the blockchain is valid
@app.get("/validate")
def validate_chain():
    is_valid = blockchain.is_chain_valid()
    return {"valid": is_valid}

# Route to add a signed transaction
@app.post("/transaction/signed")
def add_signed_transaction(transaction: dict):
    sender_wallet = Wallet()
    transaction_data = f"{transaction['sender']} sends {transaction['amount']} coins to {transaction['receiver']}"
    signature = transaction.get("signature")

    if sender_wallet.verify_signature(transaction_data, signature):
        blockchain.add_transaction(transaction)
        return {"message": "Signed transaction added successfully!"}
    else:
        return {"error": "Invalid signature"}


# Route to mine a new block
@app.get("/mine/{miner_address}")
def mine_block(miner_address: str):
    if not blockchain.unconfirmed_transactions:
        return {"message": "No transactions to mine."}

    block_index = blockchain.mine(miner_address)
    return {"message": f"Block #{block_index} mined successfully!"}


# Route to check the balance of a wallet
@app.get("/balance/{address}")
def get_balance(address: str):
    balance = blockchain.balances.get(address, 0)
    return {"address": address, "balance": balance}

# Route to get the transaction history of a wallet
@app.get("/transactions/{address}")
def get_transactions(address: str):
    transactions = [
        tx for block in blockchain.chain for tx in block.transactions
        if tx["sender"] == address or tx["receiver"] == address
    ]
    return {"address": address, "transactions": transactions}


# Модель для входных данных
class IssueRequest(BaseModel):
    address: str
    amount: int


# Новый маршрут для выдачи средств
@app.post("/issue")
def issue_balance(request: IssueRequest):
    if request.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than 0")

    blockchain.issue_initial_balance(request.address, request.amount)
    return {"message": f"Successfully issued {request.amount} coins to {request.address}"}
