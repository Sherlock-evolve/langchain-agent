"""Authenticated encryption and tenant path validation for persisted state."""

from __future__ import annotations

import base64
import binascii
import os
import re
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ENCRYPTED_MAGIC = b"WASE1"
NONCE_BYTES = 12
TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class StorageEncryptionError(RuntimeError):
    """Persisted state could not be encrypted or authenticated."""


class SnapshotCipher:
    """AES-256-GCM envelope encryption for snapshots and approvals."""

    def __init__(
        self,
        key: bytes,
        *,
        nonce_factory=None,
    ) -> None:
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("snapshot encryption key must contain 32 bytes")
        if nonce_factory is not None and not callable(nonce_factory):
            raise TypeError("nonce_factory must be callable")
        self._aead = AESGCM(key)
        self._nonce_factory = nonce_factory or (
            lambda: os.urandom(NONCE_BYTES)
        )

    @classmethod
    def from_base64(cls, encoded_key: str) -> "SnapshotCipher":
        if not isinstance(encoded_key, str) or not encoded_key.strip():
            raise ValueError("snapshot encryption key is required")
        try:
            key = base64.b64decode(
                encoded_key.strip().encode("ascii"),
                altchars=b"-_",
                validate=True,
            )
        except (UnicodeEncodeError, binascii.Error, ValueError):
            raise ValueError(
                "snapshot encryption key must be URL-safe base64"
            ) from None
        return cls(key)

    @staticmethod
    def generate_key() -> str:
        return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")

    def encrypt(self, plaintext: bytes, *, aad: bytes) -> bytes:
        if not isinstance(plaintext, bytes) or not isinstance(aad, bytes):
            raise TypeError("plaintext and aad must be bytes")
        nonce = self._nonce_factory()
        if not isinstance(nonce, bytes) or len(nonce) != NONCE_BYTES:
            raise StorageEncryptionError(
                "snapshot nonce factory returned an invalid nonce"
            )
        try:
            ciphertext = self._aead.encrypt(nonce, plaintext, aad)
        except Exception:
            raise StorageEncryptionError(
                "snapshot encryption failed"
            ) from None
        return ENCRYPTED_MAGIC + nonce + ciphertext

    def decrypt(self, payload: bytes, *, aad: bytes) -> bytes:
        if not isinstance(payload, bytes) or not isinstance(aad, bytes):
            raise TypeError("payload and aad must be bytes")
        if not payload.startswith(ENCRYPTED_MAGIC):
            raise StorageEncryptionError(
                "encrypted snapshot envelope is missing"
            )
        nonce_start = len(ENCRYPTED_MAGIC)
        nonce_end = nonce_start + NONCE_BYTES
        if len(payload) <= nonce_end:
            raise StorageEncryptionError(
                "encrypted snapshot envelope is truncated"
            )
        try:
            return self._aead.decrypt(
                payload[nonce_start:nonce_end],
                payload[nonce_end:],
                aad,
            )
        except (InvalidTag, ValueError):
            raise StorageEncryptionError(
                "encrypted snapshot authentication failed"
            ) from None


@dataclass(frozen=True)
class TenantPaths:
    """Validated, non-overlapping storage roots for one user/workspace."""

    service_root: Path
    user_id: str
    workspace_id: str

    def __post_init__(self) -> None:
        root = Path(self.service_root)
        if root.is_symlink():
            raise ValueError("service root cannot be a symbolic link")
        for name, value in (
            ("user_id", self.user_id),
            ("workspace_id", self.workspace_id),
        ):
            if (
                not isinstance(value, str)
                or TENANT_ID_PATTERN.fullmatch(value) is None
            ):
                raise ValueError(f"{name} is invalid")
        object.__setattr__(self, "service_root", root.resolve())

    @property
    def root(self) -> Path:
        return (
            self.service_root
            / "users"
            / self.user_id
            / "workspaces"
            / self.workspace_id
        )

    @property
    def workspace(self) -> Path:
        return self.root / "files"

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def knowledge(self) -> Path:
        return self.root / "knowledge"

    @property
    def audit(self) -> Path:
        return self.root / "audit"

    def prepare(self) -> None:
        for path in (
            self.workspace,
            self.sessions,
            self.knowledge,
            self.audit,
        ):
            if path.is_symlink():
                raise ValueError("tenant storage path cannot be a symlink")
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(path, 0o700)
