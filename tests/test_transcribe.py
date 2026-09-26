from types import SimpleNamespace

from pipeline.transcribe import segment_to_dict


def test_segment_to_dict_strips_and_rounds():
    seg = SimpleNamespace(
        id=1,
        text="  Hello there. ",
        start=0.12345,
        end=1.98765,
        words=[
            SimpleNamespace(word=" Hello", start=0.12345, end=0.5),
            SimpleNamespace(word=" there.", start=0.6, end=1.98765),
        ],
    )
    d = segment_to_dict(seg)
    assert d == {
        "id": 1,
        "text": "Hello there.",
        "start": 0.123,
        "end": 1.988,
        "words": [
            {"word": "Hello", "start": 0.123, "end": 0.5},
            {"word": "there.", "start": 0.6, "end": 1.988},
        ],
    }


def test_segment_to_dict_handles_missing_words():
    seg = SimpleNamespace(id=0, text="x", start=0.0, end=1.0, words=None)
    assert segment_to_dict(seg)["words"] == []
