from align import align, assign_speaker, filter_spans, group_into_turns


def span(speaker, start, end):
    return {"speaker": speaker, "start": start, "end": end}


def word(text, start, end):
    return {"word": text, "start": start, "end": end}


def test_filter_spans_drops_short():
    spans = [span("A", 0.0, 0.2), span("B", 1.0, 1.25), span("A", 2.0, 3.0)]
    assert filter_spans(spans) == [span("B", 1.0, 1.25), span("A", 2.0, 3.0)]


def test_assign_uses_max_overlap_not_midpoint():
    # Word midpoint (0.5) is in neither span; A overlaps 0.45s, B only 0.3s.
    spans = [span("A", 0.0, 0.45), span("B", 0.7, 2.0)]
    assert assign_speaker(word("hi", 0.0, 1.0), spans) == ("A", False)


def test_assign_no_overlap_goes_to_nearest_and_is_low():
    spans = [span("A", 0.0, 1.0), span("B", 5.0, 6.0)]
    assert assign_speaker(word("hi", 4.0, 4.5), spans) == ("B", True)


def test_assign_zero_duration_word_inside_span_is_not_low():
    spans = [span("A", 0.0, 1.0), span("B", 1.0, 2.0)]
    assert assign_speaker(word("uh", 1.5, 1.5), spans) == ("B", False)


def test_group_merges_small_gap_and_splits_large_gap_and_speaker_change():
    words = [
        {**word("one", 0.0, 0.5), "speaker": "A", "low": False},
        {**word("two", 2.5, 3.0), "speaker": "A", "low": True},    # gap 2.0 -> merge
        {**word("three", 5.1, 5.5), "speaker": "A", "low": False},  # gap 2.1 -> new turn
        {**word("four", 5.6, 6.0), "speaker": "B", "low": False},   # speaker change
    ]
    turns = group_into_turns(words)
    assert [(t["speaker"], t["text"], t["n_low"]) for t in turns] == [
        ("A", "one two", 1),
        ("A", "three", 0),
        ("B", "four", 0),
    ]
    assert turns[0]["start"] == 0.0 and turns[0]["end"] == 3.0


def test_n_words_counts_tokens_not_pieces():
    words = [
        {**word("e.g. that", 0.0, 0.5), "speaker": "A", "low": False},
        {**word("works", 0.6, 1.0), "speaker": "A", "low": False},
    ]
    assert group_into_turns(words)[0]["n_words"] == 3


def test_align_end_to_end_preserves_tokens():
    transcript = {
        "file": "x.wav",
        "segments": [
            {"words": [word("hello", 0.0, 0.4), word("there", 0.5, 0.9)]},
            {"words": [word("hi", 1.2, 1.5), word("back", 1.6, 2.0)]},
        ],
    }
    diarization = {"turns": [span("A", 0.0, 1.0), span("B", 1.1, 2.1), span("A", 1.0, 1.05)]}
    out = align(transcript, diarization)
    assert out["file"] == "x.wav"
    assert [(t["speaker"], t["text"]) for t in out["turns"]] == [
        ("A", "hello there"),
        ("B", "hi back"),
    ]
    assert sum(t["n_words"] for t in out["turns"]) == 4
