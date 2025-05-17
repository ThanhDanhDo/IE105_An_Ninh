## Cấu trúc thư mục

```text
Do_An_An_Ninh/
├── client/
│   ├── aes_utils.py              # Client-side AES utility functions
│   ├── client_aes.py             # Client AES encryption/decryption implementation
│   ├── pycache/                  # Python bytecode cache (auto-generated)
│   └── data/                     # Data storage for client
├── logs/
│   ├── aes_key_iv.bin            # AES key and IV file
│   ├── ciphertext_logs.txt       # Logs for encrypted data (ciphertext)
│   ├── gen_key.py                # Script to generate key
│   ├── plaintext_logs.txt        # Logs for original data (plaintext)
│   ├── requirements.txt          # Python dependencies list
│   └── test_result_log.txt       # Test result logs
├── server/
│   ├── aes_utils.py              # Server-side AES utility functions
│   ├── evaluate.py               # Evaluation script (accuracy/loss)
│   ├── server_aes.py             # Server AES encryption/decryption implementation
│   ├── pycache/                  # Python bytecode cache (auto-generated)
│   └── data/                     # Data storage for server
├── fl_env/                       # Virtual environment (ignored by Git)
    ├── pyvenv.cfg                # Virtual environment configuration
    ├── Include/                  # Include files for virtual env
    ├── Lib/                      # Library files (ignored)
    └── Scripts/                  # Activation/deactivation scripts
```
