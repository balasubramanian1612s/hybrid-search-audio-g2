"""Stage 7: evaluate keyword, vector and hybrid search against golden.json.

golden.json labels are time spans, not chunk ids (so re-chunking never breaks them):
    [{query, type, spans: [{file, start, end, why}], note?, expect_fail?, fail_reason?}]
    start/end in seconds; expect_fail marks queries we know the system gets wrong

A chunk hits a span if same file AND chunk.start_ms < span.end*1000 AND chunk.end_ms > span.start*1000.

- Recall@5 counts spans: spans hit by any top-5 chunk / spans in the query
- MRR: 1 / rank of the first chunk hitting any span, looking at the top MRR_DEPTH
- model + DB connection are loaded once for the whole run

Writes results/result_<timestamp>.md and prints the same report.
"""

import json
from datetime import datetime
from pathlib import Path

from chunk import CONTEXT_WORDS, MIN_WORDS, TARGET_MAX
from ingest import EMBED_MODEL
from search import POOL, RRF_K, connect, load_model, search

GOLDEN_PATH = Path("golden.json")
RESULTS_DIR = Path("results")

RECALL_K = 5
MRR_DEPTH = 10
MODES = ["keyword", "vector", "hybrid"]
TYPES = ["lexical", "semantic", "mixed"]

# Success criteria from design.md, checked on hybrid (the mode users get).
LEXICAL_TARGET = 0.80
SEMANTIC_TARGET = 0.70


# ---------- scoring ----------

def is_hit(result, span):
    return (
        result["file"] == span["file"]
        and result["start_ms"] < span["end"] * 1000
        and result["end_ms"] > span["start"] * 1000
    )


def span_recall(results, spans, k=RECALL_K):
    top = results[:k]
    n_hit = sum(1 for span in spans if any(is_hit(r, span) for r in top))
    return n_hit / len(spans)


def first_hit_rank(results, spans):
    """1-based rank of the first result that hits any span, or None."""
    for rank, r in enumerate(results, start=1):
        if any(is_hit(r, span) for span in spans):
            return rank
    return None


def reciprocal_rank(results, spans):
    rank = first_hit_rank(results, spans)
    return 0.0 if rank is None else 1 / rank


def mean(values):
    return sum(values) / len(values) if values else 0.0


# ---------- running ----------

def evaluate(conn, model, golden):
    rows = []
    for q in golden:
        for mode in MODES:
            results = search(conn, model, q["query"], MRR_DEPTH, mode)
            rows.append({
                "query": q["query"],
                "type": q["type"],
                "mode": mode,
                "recall": span_recall(results, q["spans"]),
                "rr": reciprocal_rank(results, q["spans"]),
                "first_rank": first_hit_rank(results, q["spans"]),
                "n_results": len(results),
            })
    return rows


def summarize(rows):
    """{(type, mode): {"n", "recall", "mrr"}}, with type "all" for every query."""
    summary = {}
    for qtype in TYPES + ["all"]:
        for mode in MODES:
            subset = [r for r in rows if r["mode"] == mode and (qtype == "all" or r["type"] == qtype)]
            summary[(qtype, mode)] = {
                "n": len(subset),
                "recall": mean([r["recall"] for r in subset]),
                "mrr": mean([r["rr"] for r in subset]),
            }
    return summary


def check_criteria(summary):
    lexical = summary[("lexical", "hybrid")]["recall"]
    semantic = summary[("semantic", "hybrid")]["recall"]
    criteria = [
        (f"lexical R@{RECALL_K} (hybrid)", f">= {LEXICAL_TARGET:.2f}", f"{lexical:.2f}", lexical >= LEXICAL_TARGET),
        (f"semantic R@{RECALL_K} (hybrid)", f">= {SEMANTIC_TARGET:.2f}", f"{semantic:.2f}", semantic >= SEMANTIC_TARGET),
    ]
    # Fusion must never lose a moment that the better single arm found.
    for qtype in TYPES:
        hybrid = summary[(qtype, "hybrid")]["recall"]
        best_single = max(summary[(qtype, "keyword")]["recall"], summary[(qtype, "vector")]["recall"])
        criteria.append((
            f"{qtype} R@{RECALL_K}: hybrid >= best single arm",
            f">= {best_single:.2f}",
            f"{hybrid:.2f}",
            hybrid >= best_single,
        ))
    return criteria


# ---------- report ----------

def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def fmt_rank(rank):
    return "-" if rank is None else str(rank)


def render_report(golden, rows, summary, criteria, n_chunks, timestamp):
    out = []
    out.append(f"# Eval {timestamp}")
    out.append("")
    out.append(
        f"{len(golden)} queries | {n_chunks} chunks | model {EMBED_MODEL} | "
        f"chunk TARGET_MAX={TARGET_MAX} MIN_WORDS={MIN_WORDS} CONTEXT_WORDS={CONTEXT_WORDS} | "
        f"RRF k={RRF_K} pool={POOL} | R@{RECALL_K}, MRR over top {MRR_DEPTH}"
    )

    out.append("")
    out.append("## Success criteria")
    out.append("")
    out.append(md_table(
        ["criterion", "target", "actual", "pass"],
        [(name, target, actual, "PASS" if ok else "FAIL") for name, target, actual, ok in criteria],
    ))

    out.append("")
    out.append("## Summary by query type")
    out.append("")
    headers = ["type", "n"]
    for mode in MODES:
        headers += [f"{mode} R@{RECALL_K}", f"{mode} MRR"]
    table_rows = []
    for qtype in TYPES + ["all"]:
        row = [qtype, summary[(qtype, MODES[0])]["n"]]
        for mode in MODES:
            s = summary[(qtype, mode)]
            row += [f"{s['recall']:.2f}", f"{s['mrr']:.2f}"]
        table_rows.append(row)
    out.append(md_table(headers, table_rows))

    n_kw_empty = sum(1 for r in rows if r["mode"] == "keyword" and r["n_results"] == 0)
    out.append("")
    out.append(f"Keyword search returned 0 results for {n_kw_empty} of {len(golden)} queries.")

    by_key = {(r["query"], r["mode"]): r for r in rows}

    expected_fails = [q for q in golden if q.get("expect_fail")]
    if expected_fails:
        out.append("")
        out.append("## Known failure cases (expect_fail in golden.json, included in the numbers above)")
        out.append("")
        table_rows = []
        for q in expected_fails:
            hybrid = by_key[(q["query"], "hybrid")]
            if hybrid["recall"] == 0:
                status = "still fails"
            elif hybrid["recall"] < 1:
                status = "partial"
            else:
                status = "now passes"
            table_rows.append([
                q["type"],
                q["query"].replace("|", "\\|"),
                f"{hybrid['recall']:.2f}",
                fmt_rank(hybrid["first_rank"]),
                status,
                q.get("fail_reason", "").replace("|", "\\|"),
            ])
        out.append(md_table(
            ["type", "query", f"hybrid R@{RECALL_K}", "hybrid rank", "status", "reason"],
            table_rows,
        ))

    out.append("")
    out.append(f"## Per query (R@{RECALL_K} and first-hit rank; '-' = no hit in top {MRR_DEPTH})")
    out.append("")
    headers = ["#", "type", "query"]
    headers += [f"{m} R@{RECALL_K}" for m in MODES]
    headers += [f"{m} rank" for m in MODES]
    table_rows = []
    for i, q in enumerate(golden, start=1):
        label = q["query"].replace("|", "\\|")
        if q.get("expect_fail"):
            label += " (expect fail)"
        row = [i, q["type"], label]
        row += [f"{by_key[(q['query'], m)]['recall']:.2f}" for m in MODES]
        row += [fmt_rank(by_key[(q["query"], m)]["first_rank"]) for m in MODES]
        table_rows.append(row)
    out.append(md_table(headers, table_rows))
    out.append("")
    return "\n".join(out)


def main():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    # Load once for the whole run.
    model = load_model()
    with connect() as conn:
        n_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        rows = evaluate(conn, model, golden)

    summary = summarize(rows)
    criteria = check_criteria(summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = render_report(golden, rows, summary, criteria, n_chunks, timestamp)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"result_{timestamp}.md"
    out_path.write_text(report)
    print(report)
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
