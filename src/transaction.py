"""
Copyright © 2025 Marc All rights reserved.
This code is part of the "Blockchain-Project" and is protected by international copyright laws.
Unauthorized use, reproduction, or distribution of this code is strictly prohibited.
"""

class Transaction:
    def __init__(self, sender, receiver, amount):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount

    def to_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount
        }
