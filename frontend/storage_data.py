from __future__ import annotations

import gc
import gzip
import json
import multiprocessing
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import ijson
import requests


INDEX_OBJECT = "index.json.gz"
RAW_PAGES_OBJECT = "raw_pages.json.gz"
DOWNLOAD_TIMEOUT = (10, 120)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


class StorageDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class StorageSettings:
    url: str
    secret_key: str
    bucket: str

    @property
    def configured(self) -> bool:
        return bool(self.url and self.secret_key and self.bucket)


@dataclass
class PageContentState:
    bodies: dict[int, str] = field(default_factory=dict)
    progress: float = 0.0
    source: str = ""
    warning: str = ""
    error: str = ""
    ready: bool = False
    page_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, progress: float, page_count: int | None = None) -> None:
        with self._lock:
            self.progress = max(self.progress, min(0.99, progress))
            if page_count is not None:
                self.page_count = page_count

    def add_bodies(self, bodies: dict[int, str]) -> None:
        with self._lock:
            self.bodies.update(bodies)

    def finish(self, source: str, warning: str, page_count: int) -> None:
        with self._lock:
            self.progress = 1.0
            self.source = source
            self.warning = warning
            self.page_count = page_count
            self.ready = True

    def fail(self, message: str) -> None:
        with self._lock:
            self.error = message

    def body_for(self, doc_id: int) -> str:
        with self._lock:
            return self.bodies.get(doc_id, "")

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "progress": self.progress,
                "source": self.source,
                "warning": self.warning,
                "error": self.error,
                "ready": self.ready,
                "page_count": self.page_count,
            }

    def load(
        self,
        local_path: str,
        remote_object: str,
        settings: StorageSettings,
        yield_seconds: float,
    ) -> None:
        if yield_seconds and hasattr(os, "nice"):
            os.nice(10)
        _load_page_content(self, Path(local_path), remote_object, settings, yield_seconds)


class PageContentManager(BaseManager):
    pass


PageContentManager.register("PageContentState", PageContentState)


def create_page_content_manager() -> PageContentManager | None:
    if os.name == "nt":
        return None
    manager = PageContentManager(ctx=multiprocessing.get_context("forkserver"))
    manager.start()
    return manager


def read_storage_settings(secrets: object | None = None) -> StorageSettings:
    if os.getenv("MSE_LOCAL_ONLY", "").strip() == "1":
        return StorageSettings(url="", secret_key="", bucket="")

    values = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL", "").strip(),
        "SUPABASE_SECRET_KEY": os.getenv("SUPABASE_SECRET_KEY", "").strip(),
        "SUPABASE_BUCKET": os.getenv("SUPABASE_BUCKET", "").strip(),
    }

    if secrets is not None:
        try:
            for key in values:
                values[key] = values[key] or str(secrets.get(key, "")).strip()
        except Exception:
            pass

    return StorageSettings(
        url=values["SUPABASE_URL"].rstrip("/"),
        secret_key=values["SUPABASE_SECRET_KEY"],
        bucket=values["SUPABASE_BUCKET"],
    )


def _storage_headers(settings: StorageSettings) -> dict[str, str]:
    return {
        "apikey": settings.secret_key,
        "Authorization": f"Bearer {settings.secret_key}",
        "User-Agent": "MSE-Tuebingen-Search/1.0",
    }


def _read_local_json(path: Path) -> dict:
    try:
        with path.open("rb") as raw_stream:
            stream = gzip.GzipFile(fileobj=raw_stream) if path.suffix == ".gz" else raw_stream
            payload = next(ijson.items(stream, "", use_float=True))
    except (OSError, StopIteration, ValueError, ijson.JSONError) as exc:
        raise StorageDataError(f"Stored object {path.name} is not valid JSON data.") from exc

    if not isinstance(payload, dict):
        raise StorageDataError("Stored JSON must contain a top-level object.")
    return payload


def _remote_objects(settings: StorageSettings, object_name: str) -> list[dict]:
    bucket = quote(settings.bucket, safe="")
    response = requests.post(
        f"{settings.url}/storage/v1/object/list/{bucket}",
        headers={**_storage_headers(settings), "Content-Type": "application/json"},
        json={"prefix": "", "limit": 100, "offset": 0},
        timeout=DOWNLOAD_TIMEOUT,
    )
    if response.status_code != 200:
        raise StorageDataError(
            f"Supabase Storage could not list {object_name} (HTTP {response.status_code})."
        )

    entries = response.json()
    part_pattern = re.compile(rf"^{re.escape(object_name)}\.part(\d{{3}})$")
    parts = [entry for entry in entries if part_pattern.match(entry.get("name", ""))]
    if parts:
        return sorted(parts, key=lambda entry: entry["name"])

    exact = [entry for entry in entries if entry.get("name") == object_name]
    if not exact:
        raise StorageDataError(f"Supabase Storage does not contain {object_name}.")
    return exact


def _object_signature(objects: list[dict]) -> str:
    values = []
    for entry in objects:
        metadata = entry.get("metadata") or {}
        values.append(
            [
                entry.get("name", ""),
                entry.get("updated_at", ""),
                metadata.get("size", 0),
                metadata.get("eTag", metadata.get("etag", "")),
            ]
        )
    return json.dumps(values, separators=(",", ":"))


def _cache_directory() -> Path:
    configured = os.getenv("MSE_DATA_CACHE_DIR", "").strip()
    cache_dir = Path(configured) if configured else Path(tempfile.gettempdir()) / "mse-search-data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_remote_file(
    settings: StorageSettings,
    object_name: str,
    progress_callback: Callable[[float], None] | None = None,
) -> Path:
    try:
        objects = _remote_objects(settings, object_name)
    except requests.RequestException as exc:
        raise StorageDataError(f"Could not reach Supabase Storage for {object_name}.") from exc

    destination = _cache_directory() / object_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    signature_path = Path(f"{destination}.meta")
    signature = _object_signature(objects)

    if destination.exists() and signature_path.exists():
        if signature_path.read_text(encoding="utf-8") == signature:
            if progress_callback:
                progress_callback(1.0)
            return destination

    total_bytes = sum(int((entry.get("metadata") or {}).get("size", 0)) for entry in objects)
    downloaded_bytes = 0
    temporary_path = Path(f"{destination}.tmp")
    bucket = quote(settings.bucket, safe="")

    try:
        with temporary_path.open("wb") as output:
            for entry in objects:
                name = entry["name"]
                object_path = quote(name, safe="/")
                with requests.get(
                    f"{settings.url}/storage/v1/object/authenticated/{bucket}/{object_path}",
                    headers=_storage_headers(settings),
                    stream=True,
                    timeout=DOWNLOAD_TIMEOUT,
                ) as response:
                    if response.status_code != 200:
                        raise StorageDataError(
                            f"Supabase Storage could not load {name} (HTTP {response.status_code})."
                        )
                    for chunk in response.iter_content(DOWNLOAD_CHUNK_SIZE):
                        if not chunk:
                            continue
                        output.write(chunk)
                        downloaded_bytes += len(chunk)
                        if progress_callback and total_bytes:
                            progress_callback(downloaded_bytes / total_bytes)
        temporary_path.replace(destination)
        signature_path.write_text(signature, encoding="utf-8")
    except requests.RequestException as exc:
        temporary_path.unlink(missing_ok=True)
        raise StorageDataError(f"Could not reach Supabase Storage for {object_name}.") from exc
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def _materialize_json_source(
    local_path: Path,
    remote_object: str,
    settings: StorageSettings,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[Path, str, str]:
    remote_warning = ""

    if settings.configured:
        try:
            path = _download_remote_file(settings, remote_object, progress_callback)
            return path, "Supabase Storage", ""
        except StorageDataError as exc:
            remote_warning = str(exc)

    for candidate in (local_path, Path(f"{local_path}.gz")):
        if candidate.exists():
            if progress_callback:
                progress_callback(1.0)
            source = f"local {candidate.name}"
            warning = f"{remote_warning} Using {source} instead." if remote_warning else ""
            return candidate, source, warning

    if remote_warning:
        raise StorageDataError(remote_warning)
    raise StorageDataError(
        f"No local {local_path.name} found and Supabase Storage is not configured."
    )


def load_json_source(
    local_path: Path,
    remote_object: str,
    secrets: object | None,
) -> tuple[dict, str, str]:
    """Load an unchanged JSON source without creating a second decoded copy in memory."""
    settings = read_storage_settings(secrets)
    path, source, warning = _materialize_json_source(local_path, remote_object, settings)
    return _read_local_json(path), source, warning


def _uncompressed_size(path: Path) -> int:
    if path.suffix != ".gz":
        return path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(-4, os.SEEK_END)
        return int.from_bytes(stream.read(4), "little")


def _load_page_content(
    state: PageContentState,
    local_path: Path,
    remote_object: str,
    settings: StorageSettings,
    yield_seconds: float,
) -> None:
    try:
        path, source, warning = _materialize_json_source(
            local_path,
            remote_object,
            settings,
            lambda progress: state.update(progress * 0.15),
        )
        total_size = max(1, _uncompressed_size(path))
        page_count = 0

        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            with path.open("rb") as raw_stream:
                stream = gzip.GzipFile(fileobj=raw_stream) if path.suffix == ".gz" else raw_stream
                body_batch = {}
                for page in ijson.items(stream, "pages.item", use_float=True):
                    doc_id = page.get("doc_id")
                    if doc_id is not None:
                        body_batch[int(doc_id)] = page.get("body", "")
                    page_count += 1
                    if yield_seconds:
                        time.sleep(yield_seconds)
                    if page_count % 25 == 0:
                        state.add_bodies(body_batch)
                        body_batch = {}
                        position = stream.tell() if path.suffix == ".gz" else raw_stream.tell()
                        state.update(0.15 + 0.84 * min(1.0, position / total_size), page_count)
                if body_batch:
                    state.add_bodies(body_batch)
        finally:
            if gc_was_enabled:
                gc.collect(0)
                gc.enable()

        state.finish(source, warning, page_count)
    except Exception as exc:
        if isinstance(exc, StorageDataError):
            state.fail(str(exc))
        else:
            state.fail(f"Page content loading failed ({type(exc).__name__}).")


def start_page_content_loading(
    local_path: Path,
    remote_object: str,
    secrets: object | None,
    manager: PageContentManager | None = None,
) -> PageContentState:
    state = manager.PageContentState() if manager else PageContentState()
    settings = read_storage_settings(secrets)
    worker = threading.Thread(
        target=state.load,
        args=(str(local_path), remote_object, settings, 0.005 if manager else 0.0),
        name="page-content-loader",
        daemon=True,
    )
    worker.start()
    return state
