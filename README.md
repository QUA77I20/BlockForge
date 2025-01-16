# 🧱 Blockchain Project

This project is a fully functioning **Proof-of-Work (PoW) blockchain** written in Python, complete with mining, transaction validation, and wallet creation. The blockchain is integrated with a **FastAPI** backend to provide a user-friendly API and a simple **HTML interface** to interact with the blockchain.

---

## ⚙️ Key Features

- **Proof-of-Work (PoW)**: Ensures security and decentralization by requiring computational work to mine new blocks.
- **Mining Rewards**: Miners receive a fixed reward for successfully mining a new block.
- **Wallets & Transactions**: Generate wallets with public/private keys and send/receive transactions between addresses.
- **Blockchain Validation**: Ensures the integrity of the entire chain by validating hashes and transactions.
- **API Integration**: Use FastAPI to interact with the blockchain via RESTful endpoints.
- **Static HTML Interface**: Simple web interface for basic blockchain interactions.

---

## 📂 Project Structure

```
Blockchain-Project
├── .venv/            # Virtual environment
├── src/              # Source code folder
│   ├── static/       # Static HTML files for the web interface
│   ├── blockchain.py # Core blockchain logic
│   ├── node.py       # FastAPI server for the blockchain
│   ├── wallet.py     # Wallet generation and signature verification
│   ├── transaction.py# Transaction-related functions
│   ├── miner.py      # Mining functionality
│   └── utils.py      # Utility functions (e.g., hashing)
├── README.md         # Project documentation
├── LICENSE           # License file
└── requirements.txt  # Dependencies
```

---

## 🚀 How It Works

1. **Blockchain**: The blockchain consists of a list of blocks, each containing transactions, a timestamp, a nonce, and a hash of the previous block.
2. **Proof-of-Work**: Each block requires solving a computational puzzle (finding the correct nonce) before it can be added to the chain.
3. **Transactions**: Users can create transactions by specifying a sender, receiver, and amount. These transactions are added to a pool of unconfirmed transactions.
4. **Mining**: Miners take the pool of unconfirmed transactions and attempt to solve the PoW puzzle. Once solved, the block is added to the chain, and the miner receives a reward.

---

## 📋 API Endpoints (via FastAPI)

| Method | Endpoint             | Description                      |
|--------|----------------------|----------------------------------|
| GET    | `/chain`             | Get the entire blockchain        |
| GET    | `/wallet`            | Create a new wallet              |
| GET    | `/balance/{address}` | Check balance of a wallet        |
| POST   | `/transaction`       | Create a new transaction         |
| GET    | `/mine/{miner}`      | Mine a new block                 |
| GET    | `/validate`          | Check if the blockchain is valid |

---

## 🖥️ Static HTML Interface

The project includes a **static HTML interface** to simplify interaction with the blockchain. The interface allows users to:

- View the current blockchain
- Create new wallets
- Send transactions
- Mine new blocks
- Check wallet balances

Before accessing the interface, ensure you have the necessary permissions and context for interacting with the project. If permissions are in place, you can access the interface by running the project and navigating to:

```
http://127.0.0.1:8080/static/index.html
```

---

## 📦 Dependencies

Dependencies used in this project:

- **FastAPI**: For building the RESTful API
- **cryptography**: For generating and verifying digital signatures
- **uvicorn**: For running the FastAPI server

---

## 🛠️ How to Interact with the Blockchain

Once the project is deployed and running, you can interact with the blockchain using either the provided **API endpoints** or the **HTML interface**.

### Option 1: API Interaction

Use the following endpoints to interact with the blockchain:

- **Create Wallet**: `POST /wallet`
- **Check Balance**: `GET /balance/{address}`
- **Send Transaction**: `POST /transaction`
- **Mine Block**: `GET /mine/{miner_address}`
- **Validate Blockchain**: `GET /validate`

Example using `curl` to check balance:

```bash
curl -X GET http://127.0.0.1:8080/balance/YOUR_WALLET_ADDRESS
```

### Option 2: HTML Interface

Navigate to the static HTML interface to easily interact with the blockchain:

```
http://127.0.0.1:8080/static/index.html
```

The interface allows you to view the blockchain, create wallets, send transactions, mine blocks, and check balances without needing to use the command line.

---

## 🔐 Security Features

- **Digital Signatures**: Transactions are signed using the sender's private key to ensure authenticity and prevent tampering.
- **Hashing**: Each block contains the hash of the previous block, making it tamper-proof.
- **Proof-of-Work**: Adds security by requiring computational effort to add new blocks.

---

## 📚 How the Blockchain Works

### 1. Creating a Wallet

A wallet is created using a pair of public and private keys. The public key acts as the wallet address, while the private key is used to sign transactions.

### 2. Sending Transactions

Transactions include a sender, receiver, and amount. They are signed by the sender's private key and verified by the network before being added to the blockchain.

### 3. Mining Blocks

Miners collect unconfirmed transactions, solve a Proof-of-Work puzzle, and add a new block to the chain. The miner receives a reward for their work.

### 4. Validating the Blockchain

The blockchain can be validated by checking the hashes of each block and ensuring that all transactions are properly signed and verified.

---

## 🔄 Future Enhancements

- **GPU Miner**: Implement GPU-based mining to significantly increase mining efficiency and reduce computation time.
- **Perceptron Integration**: Integrate a perceptron-based AI model to optimize transaction validation, predict block hashes, and enhance security.
- **Mobile Interface**: Develop a mobile-friendly interface for interacting with the blockchain on the go.
- **Smart Contracts**: Add functionality for creating and executing smart contracts directly on the blockchain.
- **Decentralized Nodes**: Allow multiple nodes to run simultaneously, creating a decentralized blockchain network.

---

## ⚖️ License

```
Copyright © 2025 Marc. All rights reserved.
Unauthorized use, reproduction, or distribution of this code is strictly prohibited.
```

