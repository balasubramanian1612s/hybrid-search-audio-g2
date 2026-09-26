import pytest

from eval import first_hit_rank, is_hit, reciprocal_rank, span_recall


def res(file, start_ms, end_ms):
    return {"file": file, "start_ms": start_ms, "end_ms": end_ms}


def span(file, start, end):
    return {"file": file, "start": start, "end": end}


def test_is_hit_needs_same_file_and_overlap():
    s = span("a.wav", 10.0, 20.0)
    assert is_hit(res("a.wav", 15000, 25000), s)
    assert not is_hit(res("b.wav", 15000, 25000), s)       # same time, other file
    assert not is_hit(res("a.wav", 20000, 30000), s)       # touching edge is not overlap
    assert not is_hit(res("a.wav", 0, 10000), s)


def test_span_recall_counts_spans_not_chunks():
    spans = [span("a.wav", 0, 10), span("a.wav", 100, 110)]
    # three chunks all inside the first span -> only 1 of 2 spans found
    results = [res("a.wav", 0, 3000), res("a.wav", 3000, 6000), res("a.wav", 6000, 9000)]
    assert span_recall(results, spans, k=5) == 0.5


def test_span_recall_only_looks_at_top_k():
    spans = [span("a.wav", 100, 110)]
    results = [res("a.wav", 0, 1000)] * 5 + [res("a.wav", 100000, 105000)]
    assert span_recall(results, spans, k=5) == 0.0
    assert span_recall(results, spans, k=6) == 1.0


def test_first_hit_rank_and_rr():
    spans = [span("a.wav", 50, 60)]
    results = [res("a.wav", 0, 1000), res("a.wav", 55000, 58000)]
    assert first_hit_rank(results, spans) == 2
    assert reciprocal_rank(results, spans) == pytest.approx(0.5)
    assert first_hit_rank([], spans) is None
    assert reciprocal_rank([], spans) == 0.0
