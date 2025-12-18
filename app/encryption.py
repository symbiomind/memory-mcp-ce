"""
BITEY Cheese Encryption Module

AES-256-GCM authenticated encryption with Argon2id key derivation.
Provides optional encryption for memory content storage.

Storage format in BYTEA:
[16 bytes salt][12 bytes nonce][N bytes ciphertext][16 bytes auth_tag]
"""

import os
import logging
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from argon2.low_level import hash_secret_raw, Type

logger = logging.getLogger(__name__)

# Constants
SALT_SIZE = 16  # 16 bytes for Argon2id salt
NONCE_SIZE = 12  # 12 bytes for AES-GCM nonce
KEY_SIZE = 32  # 256-bit key for AES-256
TAG_SIZE = 16  # AES-GCM auth tag is 16 bytes (included in ciphertext by AESGCM)

# Argon2id parameters (secure defaults for homelabber use)
ARGON2_TIME_COST = 3  # iterations
ARGON2_MEMORY_COST = 65536  # 64 MB in KB
ARGON2_PARALLELISM = 4


def get_encryption_key() -> Optional[str]:
    """
    Get the encryption key from environment.
    
    Returns:
        The encryption key string if set and non-empty, None otherwise.
    """
    from app.config import ENCRYPTION_KEY
    if ENCRYPTION_KEY and ENCRYPTION_KEY.strip():
        return ENCRYPTION_KEY.strip()
    return None


def is_encryption_enabled() -> bool:
    """
    Check if encryption is enabled (ENCRYPTION_KEY is set and non-empty).
    
    Returns:
        True if encryption is enabled, False otherwise.
    """
    return get_encryption_key() is not None


def derive_key(password: str, salt: bytes) -> bytes:
    """
    Derive a 256-bit key from password using Argon2id.
    
    Args:
        password: The encryption key/password string
        salt: Random salt bytes (16 bytes)
    
    Returns:
        32-byte derived key suitable for AES-256
    """
    return hash_secret_raw(
        secret=password.encode('utf-8'),
        salt=salt,
        time_cost=ARGON2_TIME_COST,
        memory_cost=ARGON2_MEMORY_COST,
        parallelism=ARGON2_PARALLELISM,
        hash_len=KEY_SIZE,
        type=Type.ID  # Argon2id
    )


def encrypt_content(plaintext: str) -> Optional[bytes]:
    """
    Encrypt content using AES-256-GCM with Argon2id key derivation.
    
    Args:
        plaintext: The content string to encrypt
    
    Returns:
        Encrypted blob as bytes: [salt][nonce][ciphertext+tag]
        Returns None if encryption is not enabled.
    """
    key_str = get_encryption_key()
    if not key_str:
        return None
    
    try:
        # Generate random salt and nonce
        salt = os.urandom(SALT_SIZE)
        nonce = os.urandom(NONCE_SIZE)
        
        # Derive key using Argon2id
        key = derive_key(key_str, salt)
        
        # Encrypt with AES-256-GCM
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), None)
        
        # Combine: salt + nonce + ciphertext (includes auth tag)
        encrypted_blob = salt + nonce + ciphertext
        
        return encrypted_blob
        
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        return None


def decrypt_content(encrypted_blob: bytes) -> Optional[str]:
    """
    Decrypt content encrypted with encrypt_content().
    
    Args:
        encrypted_blob: The encrypted bytes [salt][nonce][ciphertext+tag]
    
    Returns:
        Decrypted plaintext string, or None if decryption fails.
    """
    key_str = get_encryption_key()
    if not key_str:
        logger.debug("Cannot decrypt: no encryption key configured")
        return None
    
    try:
        # Minimum size: salt + nonce + at least 1 byte ciphertext + tag
        min_size = SALT_SIZE + NONCE_SIZE + 1 + TAG_SIZE
        if len(encrypted_blob) < min_size:
            logger.warning(f"Encrypted blob too small: {len(encrypted_blob)} bytes")
            return None
        
        # Extract components
        salt = encrypted_blob[:SALT_SIZE]
        nonce = encrypted_blob[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
        ciphertext = encrypted_blob[SALT_SIZE + NONCE_SIZE:]
        
        # Derive key using Argon2id with extracted salt
        key = derive_key(key_str, salt)
        
        # Decrypt with AES-256-GCM
        aesgcm = AESGCM(key)
        plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
        
        return plaintext_bytes.decode('utf-8')
        
    except Exception as e:
        # This catches wrong key, corrupted data, tampered data, etc.
        logger.warning(f"Decryption failed (wrong key or corrupted data): {e}")
        return None


def decode_or_decrypt_content(content_bytes: bytes, is_encrypted: bool) -> Optional[str]:
    """
    Decode or decrypt content based on the enc flag.
    
    Args:
        content_bytes: The raw bytes from database
        is_encrypted: The enc flag from database
    
    Returns:
        Plaintext string, or None if decryption fails or key not available.
    """
    if is_encrypted:
        # Encrypted content - need key to decrypt
        return decrypt_content(content_bytes)
    else:
        # Plain UTF-8 encoded content
        try:
            return content_bytes.decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to decode UTF-8 content: {e}")
            return None


def should_include_memory(is_encrypted: bool) -> bool:
    """
    Determine if a memory should be included in results based on encryption state.
    
    Args:
        is_encrypted: The enc flag from database
    
    Returns:
        True if the memory can be accessed, False otherwise.
    """
    if not is_encrypted:
        # Unencrypted memories are always accessible
        return True
    
    # Encrypted memories require the encryption key
    return is_encryption_enabled()
