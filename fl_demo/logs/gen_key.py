import os

key = os.urandom(32)  # 256-bit key
iv = os.urandom(16)   # 128-bit IV

with open("aes_key_iv.bin", "wb") as f:
    f.write(key + iv)
print("AES key and IV generated at logs/aes_key_iv.bin")