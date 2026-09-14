"""Small S3-compatible artifact client used by Ledger ML workloads."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


class ArtifactStoreError(RuntimeError):
    """Raised when an artifact cannot be transferred."""


class S3ArtifactStore:
    def __init__(self, bucket: str, *, env: Mapping[str, str] | None = None) -> None:
        values = os.environ if env is None else env
        if not bucket.strip():
            raise ArtifactStoreError("artifact bucket must not be empty")
        self.bucket = bucket
        self._client = self._build_client(values)

    @staticmethod
    def _build_client(env: Mapping[str, str]):
        try:
            import boto3
        except ImportError as err:
            raise ArtifactStoreError("boto3 is required for ObjectStore artifacts") from err
        kwargs: dict[str, str] = {
            "region_name": env.get("LEDGERML_S3_REGION", "us-east-1"),
        }
        endpoint = (env.get("LEDGERML_S3_ENDPOINT_URL") or env.get("AWS_ENDPOINT_URL", "")).strip()
        if endpoint:
            kwargs["endpoint_url"] = endpoint
        addressing_style = env.get("LEDGERML_S3_ADDRESSING_STYLE", "").strip()
        if addressing_style:
            kwargs["config"] = boto3.session.Config(s3={"addressing_style": addressing_style})
        return boto3.client("s3", **kwargs)

    def key(self, output_path: str, version: str, filename: str) -> str:
        parts = [output_path.strip("/"), version.strip("/"), filename]
        return "/".join(part for part in parts if part)

    def upload_file(self, source: str | Path, key: str) -> None:
        try:
            self._client.upload_file(str(source), self.bucket, key)
        except Exception as err:
            raise ArtifactStoreError(f"unable to upload artifact {key!r}: {err}") from err

    def download_file(self, key: str, destination: str | Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self.bucket, key, str(target))
        except Exception as err:
            raise ArtifactStoreError(f"unable to download artifact {key!r}: {err}") from err
        return target


def artifact_store_from_env(env: Mapping[str, str] | None = None) -> S3ArtifactStore:
    values = os.environ if env is None else env
    return S3ArtifactStore(values.get("LEDGERML_OUTPUT_NAME", ""), env=values)
