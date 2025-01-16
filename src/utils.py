"""
Copyright © 2025 Marc All rights reserved.
This code is part of the "Blockchain-Project" and is protected by international copyright laws.
Unauthorized use, reproduction, or distribution of this code is strictly prohibited.
"""

import hashlib
import json

# Function to hash data using SHA-256
def sha256_hash(data):
    json_data = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_data.encode()).hexdigest()

# Function to serialize a block to JSON
def serialize_block(block):
    return json.dumps(block.__dict__, sort_keys=True)

# Function to deserialize JSON back to a Block object
def deserialize_block(json_data):
    data = json.loads(json_data)
    return Block(
        data['index'],
        data['previous_hash'],
        data['data'],
        data['timestamp']
    )
