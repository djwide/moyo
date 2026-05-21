"""Shared storage utilities for sente and moyo projects.

Provides a uniform interface for local and S3-compatible backends.
"""

import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Union


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists
        
    Returns:
        Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_files(directory: Union[str, Path], 
               pattern: str = "*", 
               recursive: bool = False) -> List[Path]:
    """List files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def copy_directory(source: Union[str, Path], 
                   destination: Union[str, Path],
                   overwrite: bool = False) -> None:
    """Copy a directory and its contents.
    
    Args:
        source: Source directory
        destination: Destination directory
        overwrite: Whether to overwrite existing files
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    
    shutil.copytree(source, destination, dirs_exist_ok=overwrite)


def safe_filename(filename: str) -> str:
    """Convert a filename to a safe version for filesystem.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Replace problematic characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Ensure it's not empty
    if not filename:
        filename = "unnamed"
    
    return filename


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    file_path = Path(file_path)
    if file_path.exists():
        return file_path.stat().st_size
    return 0


def get_directory_size(directory: Union[str, Path]) -> int:
    """Get total size of directory in bytes.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    total_size = 0
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size


def backup_file(file_path: Union[str, Path], 
                backup_suffix: str = ".backup") -> Optional[Path]:
    """Create a backup of a file.
    
    Args:
        file_path: Path to the file to backup
        backup_suffix: Suffix for the backup file
        
    Returns:
        Path to the backup file, or None if backup failed
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    
    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
    
    try:
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception:
        return None


# Storage abstraction

class Storage(Protocol):
    """Abstract storage interface."""

    def read_bytes(self, key: str) -> bytes: ...
    def write_bytes(self, key: str, data: bytes) -> None: ...
    def open_reader(self, key: str) -> io.BufferedReader: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str = "") -> List[str]: ...
    def presigned_url(self, key: str, expires_in: int = 3600) -> Optional[str]: ...


@dataclass
class LocalStorage:
    """Local filesystem storage backend scoped to a root directory."""

    root_dir: Union[str, Path]

    def _full_path(self, key: str) -> Path:
        if key.startswith("/"):
            key = key[1:]
        return Path(self.root_dir) / key

    def read_bytes(self, key: str) -> bytes:
        return self._full_path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        full = self._full_path(key)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)

    def open_reader(self, key: str) -> io.BufferedReader:
        return open(self._full_path(key), "rb")

    def exists(self, key: str) -> bool:
        return self._full_path(key).exists()

    def delete(self, key: str) -> None:
        try:
            self._full_path(key).unlink()
        except FileNotFoundError:
            pass

    def list(self, prefix: str = "") -> List[str]:
        base = self._full_path(prefix)
        if not base.exists():
            return []
        keys: List[str] = []
        for p in base.rglob("*"):
            if p.is_file():
                rel = p.relative_to(self._full_path(""))
                keys.append(str(rel).replace("\\", "/"))
        return keys

    def presigned_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        return None


@dataclass
class S3Storage:
    """S3-compatible storage backend using boto3."""

    bucket: str
    prefix: str = ""
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    addressing_style: str = "auto"

    def __post_init__(self) -> None:
        import boto3
        from botocore.config import Config as BotoConfig

        cfg = BotoConfig(s3={"addressing_style": self.addressing_style})
        self._s3 = boto3.client("s3", region_name=self.region, endpoint_url=self.endpoint_url, config=cfg)

    def _key(self, key: str) -> str:
        key = key.lstrip("/")
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{key}"
        return key

    def read_bytes(self, key: str) -> bytes:
        obj = self._s3.get_object(Bucket=self.bucket, Key=self._key(key))
        return obj["Body"].read()

    def write_bytes(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def open_reader(self, key: str) -> io.BufferedReader:
        import tempfile
        data = self.read_bytes(key)
        tmp = tempfile.TemporaryFile()
        tmp.write(data)
        tmp.seek(0)
        return tmp  # type: ignore[return-value]

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=self._key(key))

    def list(self, prefix: str = "") -> List[str]:
        keys: List[str] = []
        token: Optional[str] = None
        list_prefix = self._key(prefix)
        while True:
            kwargs = dict(Bucket=self.bucket, Prefix=list_prefix)
            if token:
                kwargs["ContinuationToken"] = token
            resp = self._s3.list_objects_v2(**kwargs)
            for item in resp.get("Contents", []) or []:
                key = item["Key"]
                if self.prefix and key.startswith(self.prefix):
                    key = key[len(self.prefix):].lstrip("/")
                keys.append(key)
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return keys

    def presigned_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        try:
            return self._s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.bucket, "Key": self._key(key)},
                ExpiresIn=expires_in,
            )
        except Exception:
            return None


def get_storage(config: Optional[dict] = None) -> Storage:
    """Factory for Storage from env/config.

    Env vars supported:
    - STORAGE_BACKEND=local|s3
    - LOCAL_ROOT=./data
    - S3_BUCKET, S3_PREFIX, S3_REGION, S3_ENDPOINT_URL, S3_ADDRESSING
    """
    cfg = {**{
        "backend": os.environ.get("STORAGE_BACKEND", "local"),
        "local_root": os.environ.get("LOCAL_ROOT", "./data"),
        "s3_bucket": os.environ.get("S3_BUCKET"),
        "s3_prefix": os.environ.get("S3_PREFIX", ""),
        "s3_region": os.environ.get("S3_REGION"),
        "s3_endpoint_url": os.environ.get("S3_ENDPOINT_URL"),
        "s3_addressing": os.environ.get("S3_ADDRESSING", "auto"),
    }, **(config or {})}

    if cfg["backend"] == "s3":
        if not cfg.get("s3_bucket"):
            raise ValueError("S3 backend requires S3_BUCKET")
        return S3Storage(
            bucket=cfg["s3_bucket"],
            prefix=cfg.get("s3_prefix", ""),
            region=cfg.get("s3_region"),
            endpoint_url=cfg.get("s3_endpoint_url"),
            addressing_style=cfg.get("s3_addressing", "auto"),
        )
    # default local
    return LocalStorage(cfg["local_root"]) 

