"""Black-box paths should not load the native FAISS library at import time."""

import importlib
import sys

import shared_utils.faiss_index as faiss_index_module


def test_llm_fuzzer_import_does_not_load_faiss(monkeypatch):
    monkeypatch.setattr(faiss_index_module, "_faiss_module", None, raising=False)
    monkeypatch.setattr(faiss_index_module, "_faiss_unavailable", False, raising=False)
    sys.modules.pop("faiss", None)

    from moyo.publicside.barrierprobe import llm_fuzzer  # noqa: F401

    assert faiss_index_module._faiss_module is None
    assert "faiss" not in sys.modules


def test_llm_fuzzer_import_does_not_load_embeddings():
    sys.modules.pop("sentence_transformers", None)
    sys.modules.pop("torch", None)
    sys.modules.pop("shared_utils.embeddings", None)

    import moyo.publicside.barrierprobe.llm_fuzzer as llm_fuzzer

    importlib.reload(llm_fuzzer)

    assert "sentence_transformers" not in sys.modules
    assert "torch" not in sys.modules
    assert "shared_utils.embeddings" not in sys.modules


def test_queryseed_import_does_not_load_numpy_analyzer():
    """Cloud model-rerun imports QuerySeed; that must not pull BarrierAnalyzer."""
    sys.modules.pop("moyo.publicside.barrierprobe.barrier_analyzer", None)
    sys.modules.pop("moyo.publicside.barrierprobe.distribution", None)

    import moyo.publicside.barrierprobe as barrierprobe

    importlib.reload(barrierprobe)
    from moyo.publicside.barrierprobe.llm_fuzzer import QuerySeed

    assert QuerySeed is not None
    assert "moyo.publicside.barrierprobe.barrier_analyzer" not in sys.modules
    assert "moyo.publicside.barrierprobe.distribution" not in sys.modules


def test_faiss_index_load_still_imports_faiss(monkeypatch, tmp_path):
    monkeypatch.setattr(faiss_index_module, "_faiss_module", None, raising=False)
    monkeypatch.setattr(faiss_index_module, "_faiss_unavailable", False, raising=False)
    sys.modules.pop("faiss", None)

    from shared_utils.faiss_index import FAISSIndex

    dim = 8
    index = FAISSIndex(dimension=dim, index_type="flat")
    index.add_vectors([[0.0] * dim])
    index.save(tmp_path, name="probe")

    sys.modules.pop("faiss", None)
    monkeypatch.setattr(faiss_index_module, "_faiss_module", None, raising=False)

    FAISSIndex.load(tmp_path)
    assert faiss_index_module._faiss_module is not None
    assert "faiss" in sys.modules
