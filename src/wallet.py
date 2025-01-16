"""
Copyright © 2025 Marc All rights reserved.
Unauthorized use, reproduction, or distribution of this code is strictly prohibited.
"""

import os
import hashlib
import base64
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import os

class Wallet:
    def __init__(self):
        self.private_key = ec.generate_private_key(ec.SECP256K1())
        self.public_key = self.private_key.public_key()
        self.address = self.generate_address()
        self.encrypted_private_key = self.encrypt_private_key()

    def generate_address(self):
        public_key_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha256(public_key_bytes).hexdigest()

    def encrypt_private_key(self):
        key = os.urandom(32)
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
        encryptor = cipher.encryptor()
        encrypted_key = encryptor.update(self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )) + encryptor.finalize()
        return encrypted_key

    def decrypt_private_key(self, encrypted_key, key, iv):
        cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted_key) + decryptor.finalize()

    def sign_transaction(self, transaction_data):
        """
        Подписываем транзакцию приватным ключом.
        """
        signature = self.private_key.sign(
            transaction_data.encode(),
            ec.ECDSA(SHA256())
        )
        return base64.b64encode(signature).decode()

    def verify_signature(self, transaction_data, signature):
        """
        Проверяем подпись транзакции с помощью публичного ключа.
        """
        signature_bytes = base64.b64decode(signature)
        try:
            self.public_key.verify(
                signature_bytes,
                transaction_data.encode(),
                ec.ECDSA(SHA256())
            )
            return True
        except:
            return False

# Тестирование кошелька
if __name__ == "__main__":
    wallet = Wallet()
    print("🔐 Wallet Address:", wallet.address)

    # Тестовая транзакция
    transaction = "Marc sends 100 coins to Chris"
    signature = wallet.sign_transaction(transaction)
    print("📜 Signature:", signature)

    # Проверка подписи
    is_valid = wallet.verify_signature(transaction, signature)
    print("✅ Is signature valid?", is_valid)

