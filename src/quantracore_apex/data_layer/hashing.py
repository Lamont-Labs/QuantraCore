"""
Data Hashing module for QuantraCore Apex.

Provides cryptographic hashing for data integrity verification.
"""

import hashlib
from typing import Union


def compute_data_hash(data: Union[str, bytes]) -> str:
    """
    Compute SHA-256 hash of data.
    
    Args:
        data: String or bytes to hash
        
    Returns:
        Hex digest of SHA-256 hash
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    
    return hashlib.sha256(data).hexdigest()


def compute_file_hash(filepath: str) -> str:
    """
    Compute SHA-256 hash of a file.
    
    Args:
        filepath: Path to file
        
    Returns:
        Hex digest of SHA-256 hash
    """
    sha256 = hashlib.sha256()
    
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def verify_hash(data: Union[str, bytes], expected_hash: str) -> bool:
    """
    Verify that data matches expected hash.
    
    Args:
        data: Data to verify
        expected_hash: Expected SHA-256 hash
        
    Returns:
        True if hashes match
    """
    return compute_data_hash(data) == expected_hash


def compute_window_hash(bars_json: str) -> str:
    """
    Compute a short hash for window identification.
    
    Returns first 16 characters of SHA-256.
    """
    full_hash = compute_data_hash(bars_json)
    return full_hash[:16]
