"""Local report upload uses the same GCS layout as Cloud Run jobs."""

from __future__ import annotations

from pathlib import Path

import pytest

from moyo.order_storage import order_storage_folder
from moyo.report_storage import (
    infer_local_run_id,
    local_order_id,
    project_slug_from_path,
    publish_local_report,
    reports_bucket_name,
    should_upload_local_report,
)


class _FakeBlob:
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploads: list[tuple[str, str | None]] = []
        self.text: str | None = None

    def upload_from_filename(self, path: str, content_type: str | None = None) -> None:
        self.uploads.append((path, content_type))

    def upload_from_string(self, data: str, content_type: str | None = None) -> None:
        self.text = data
        self.uploads.append(("string", content_type))


class _FakeBucket:
    def __init__(self, name: str = "senteguard-website-moyo-reports") -> None:
        self.name = name
        self.blobs: dict[str, _FakeBlob] = {}

    def blob(self, path: str) -> _FakeBlob:
        blob = self.blobs.get(path) or _FakeBlob(path)
        self.blobs[path] = blob
        return blob


def test_local_order_id_is_stable_per_run():
    oid = local_order_id("tell_me_about_senteguard_founder_david_weidman")
    assert oid == "ord_local_tell_me_about_senteguard_founder_david_weidman"
    folder = order_storage_folder(
        oid,
        ["Tell me about SenteGuard founder David Weidman"],
    )
    assert folder == "senteguard_dweidman"
    assert not folder.startswith("ord_")


def test_infer_run_id_uses_project_slug(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    projects = tmp_path / "projects"
    expl = (
        projects
        / "tell_me_about_senteguard_founder_david_weidman"
        / "public_sources"
        / "exploration.md"
    )
    expl.parent.mkdir(parents=True)
    expl.write_text("# Topic exploration: Tell me about SenteGuard\n", encoding="utf-8")
    monkeypatch.setenv("MOYO_PROJECTS_DIR", str(projects))
    assert project_slug_from_path(expl) == "tell_me_about_senteguard_founder_david_weidman"
    assert infer_local_run_id(expl) == "tell_me_about_senteguard_founder_david_weidman"


def test_infer_run_id_keeps_explicit_fallback():
    assert infer_local_run_id(None, fallback="custom_run") == "custom_run"


def test_should_upload_defaults_on_for_local_render(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MOYO_REPORTS_SKIP_UPLOAD", raising=False)
    monkeypatch.delenv("MOYO_CLOUD_RUNTIME", raising=False)
    monkeypatch.delenv("MOYO_TEST_MODE", raising=False)
    monkeypatch.delenv("CLOUD_RUN_JOB", raising=False)
    monkeypatch.delenv("CLOUD_RUN_EXECUTION", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    assert should_upload_local_report() is True
    assert should_upload_local_report(upload=False) is False
    assert should_upload_local_report(test_mode=True) is False
    assert should_upload_local_report(graphics_only=True) is False
    assert should_upload_local_report(rendered=False) is False


def test_should_upload_skips_llm_test_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MOYO_REPORTS_SKIP_UPLOAD", raising=False)
    monkeypatch.delenv("MOYO_CLOUD_RUNTIME", raising=False)
    monkeypatch.setenv("MOYO_TEST_MODE", "1")
    assert should_upload_local_report() is False


def test_should_upload_skips_cloud_runtime(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MOYO_REPORTS_SKIP_UPLOAD", raising=False)
    monkeypatch.delenv("MOYO_TEST_MODE", raising=False)
    monkeypatch.setenv("MOYO_CLOUD_RUNTIME", "1")
    assert should_upload_local_report() is False


def test_should_upload_honors_skip_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOYO_REPORTS_SKIP_UPLOAD", "1")
    assert should_upload_local_report() is False


def test_reports_bucket_name_normalizes_gs_uri(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOYO_REPORTS_STORAGE_BUCKET", "gs://senteguard-website-moyo-reports/")
    assert reports_bucket_name() == "senteguard-website-moyo-reports"


def test_publish_local_report_uses_cloud_object_names(tmp_path: Path):
    run_dir = tmp_path / "run"
    output = run_dir / "output"
    assets = run_dir / "assets"
    output.mkdir(parents=True)
    assets.mkdir()
    (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (output / "report.pdf").write_bytes(b"%PDF")
    (output / "report.html").write_text("<html/>", encoding="utf-8")
    (run_dir / "report_data.json").write_text("{}", encoding="utf-8")
    (assets / "exposure-radar.svg").write_text("<svg/>", encoding="utf-8")
    expl = tmp_path / "exploration.md"
    expl.write_text("# Topic exploration: Tell me about SenteGuard\n", encoding="utf-8")

    bucket = _FakeBucket()
    result = publish_local_report(
        run_dir=run_dir,
        run_id="tell_me_about_senteguard_founder_david_weidman",
        product="snapshot",
        exploration=expl,
        bucket=bucket,
    )
    assert result.storage_folder == "senteguard_dweidman"
    assert result.prefix == "reports/senteguard_dweidman"
    assert result.gcs_prefix == (
        "gs://senteguard-website-moyo-reports/reports/senteguard_dweidman/"
    )
    assert f"{result.prefix}/report.pdf" in bucket.blobs
    assert f"{result.prefix}/report.md" in bucket.blobs
    assert f"{result.prefix}/report.html" in bucket.blobs
    assert f"{result.prefix}/report.json" in bucket.blobs
    assert f"{result.prefix}/manifest.json" in bucket.blobs
    assert f"{result.prefix}/exploration.md" in bucket.blobs
    assert f"{result.prefix}/assets/exposure-radar.svg" in bucket.blobs
    assert (run_dir / "gcs.prefix").read_text(encoding="utf-8").strip() == result.gcs_prefix
    # Single-prompt orders stay flat (no 01_slug/ subfolder), matching Cloud Run.
    nested = [name for name in bucket.blobs if "/01_" in name]
    assert nested == []
