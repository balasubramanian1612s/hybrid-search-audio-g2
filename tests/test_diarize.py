from pyannote.core import Annotation, Segment

from diarize import annotation_to_turns, speaker_totals


def make_annotation():
    ann = Annotation()
    ann[Segment(5.0, 7.5)] = "SPEAKER_01"
    ann[Segment(0.0, 4.0)] = "SPEAKER_00"
    ann[Segment(7.0, 9.0)] = "SPEAKER_00"  # overlaps SPEAKER_01
    return ann


def test_annotation_to_turns_sorted_by_start():
    turns = annotation_to_turns(make_annotation())
    assert turns == [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 4.0},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 7.5},
        {"speaker": "SPEAKER_00", "start": 7.0, "end": 9.0},
    ]


def test_speaker_totals():
    totals = speaker_totals(annotation_to_turns(make_annotation()))
    assert totals == {"SPEAKER_00": 6.0, "SPEAKER_01": 2.5}
