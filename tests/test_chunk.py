from chunk import TARGET_MAX, chunk, is_backchannel, merge_small, split_turn


def make_words(text, start=0.0, step=0.5):
    words = []
    t = start
    for tok in text.split():
        words.append({"word": tok, "start": t, "end": t + 0.4})
        t += step
    return words


def make_turn(speaker, text, start=0.0):
    words = make_words(text, start)
    return {"speaker": speaker, "text": text, "words": words}


def sentence(n, label="w"):
    """A sentence of n words ending with a period."""
    return " ".join(f"{label}{i}" for i in range(n - 1)) + " end."


def test_is_backchannel():
    assert is_backchannel({"text": "Yeah."})
    assert is_backchannel({"text": "Mm-hmm, right, right."})
    assert not is_backchannel({"text": "Yeah, I think so."})


def test_split_turn_packs_whole_sentences_under_target():
    words = make_words(" ".join([sentence(30), sentence(30), sentence(30)]))
    pieces = split_turn(words, target_max=80)
    assert [len(p) for p in pieces] == [60, 30]
    assert pieces[0][-1]["word"] == "end."


def test_split_turn_hard_cuts_a_sentence_longer_than_target():
    words = make_words(sentence(170))
    assert [len(p) for p in split_turn(words, target_max=80)] == [80, 80, 10]


def test_merge_small_prefers_previous_same_speaker():
    a = {"speaker": "A", "words": make_words(sentence(40)), "context": "q"}
    b = {"speaker": "A", "words": make_words(sentence(5)), "context": "q"}
    out = merge_small([a, b])
    assert len(out) == 1 and len(out[0]["words"]) == 45


def test_merge_small_falls_forward_then_keeps_if_alone():
    small_a = {"speaker": "A", "words": make_words(sentence(5)), "context": ""}
    next_a = {"speaker": "A", "words": make_words(sentence(40)), "context": ""}
    lone_b = {"speaker": "B", "words": make_words(sentence(5)), "context": "x"}
    out = merge_small([small_a, next_a, lone_b])
    assert [(c["speaker"], len(c["words"])) for c in out] == [("A", 45), ("B", 5)]


def test_chunk_end_to_end():
    question = "Why is the moon so important to us?"
    # two sentences that together exceed TARGET_MAX, so they must split
    half = TARGET_MAX // 2 + 5
    answer = " ".join([sentence(half, "a"), sentence(half, "b")])
    turns_data = {
        "file": "x.wav",
        "turns": [
            make_turn("A", question, start=0.0),
            make_turn("B", "Yeah.", start=5.0),
            make_turn("B", answer, start=10.0),
        ],
    }
    chunks = chunk(turns_data)

    # question kept alone (no same-speaker neighbour), backchannel dropped,
    # answer split into two chunks at the sentence boundary
    assert [(c["speaker"], c["n_words"]) for c in chunks] == [("A", 8), ("B", half), ("B", half)]

    # both answer chunks carry the question as context; text stays clean
    for c in chunks[1:]:
        assert c["embed_text"] == f"{question} {c['text']}"
        assert question not in c["text"]
    assert chunks[0]["embed_text"] == question

    # exact times from word timings: second answer chunk starts at word `half`
    assert chunks[1]["start_ms"] == 10000
    assert chunks[2]["start_ms"] == 10000 + half * 500
