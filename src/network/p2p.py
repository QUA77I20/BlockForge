"""
Copyright © 2025 Marc All rights reserved.
This code is part of the "Blockchain-Project" and is protected by international copyright laws.
Unauthorized use, reproduction, or distribution of this code is strictly prohibited.
"""

import socket

class P2PNetwork:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.peers = []

    def connect_to_peer(self, peer_host, peer_port):
        self.peers.append((peer_host, peer_port))
        print(f"Connected to peer: {peer_host}:{peer_port}")
