"""Compare six PDSM rules on one saved score table, without loading audio models."""

import argparse
import csv
import hashlib
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import posixpath
import tempfile

import numpy as np

from .process.pdsm import select_indices


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str
    value: float


RULES = (
    Rule("top_5pct", "percent", 0.05),
    Rule("top_10pct", "percent", 0.10),
    Rule("top_20pct", "percent", 0.20),
    Rule("adaptive_0.5", "adaptive", 0.5),
    Rule("adaptive_1.0", "adaptive", 1.0),
    Rule("adaptive_1.5", "adaptive", 1.5),
)
REQUIRED = {
    "file_path", "wav_channel", "wav_start", "wav_end", "dimension",
    "phoneme_index", "start_frame", "end_frame", "start_seconds",
    "end_seconds", "score",
}
COMPARISON_FIELDS = [
    "method", "rule", "k", "adaptive_lambda", "split", "dimension",
    "n_files", "n_groups", "n_phonemes", "n_selected", "selection_rate",
    "n_labelled", "n_positive", "n_negative", "tp", "fp", "fn", "tn",
    "precision", "recall", "f1", "fpr",
]
SELECTION_FIELDS = [
    "method", "method_rule", "method_k", "method_lambda",
    "method_cutoff", "method_selected", "method_rank",
]


def _number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value, name):
    result = _number(value, name)
    if not result.is_integer() or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(result)


def load_scores(input_csv):
    """Validate complete utterance/dimension groups and file-level splits."""
    with Path(input_csv).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        if len(fields) != len(set(fields)):
            raise ValueError("CSV contains duplicate column names")
        missing = REQUIRED - set(fields)
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        if set(fields) & set(SELECTION_FIELDS):
            raise ValueError("Input must be phoneme_scores.csv, not selections.csv")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("CSV contains no phoneme rows")
    groups, entries, file_splits = {}, [], {}
    for row_number, raw in enumerate(raw_rows, 2):
        if None in raw:
            raise ValueError(f"CSV row {row_number} has too many values")
        # Preserve the exported text, including blank/whitespace phoneme symbols.
        # Only normalize values used for validation/grouping below.
        raw = {key: (value or "") for key, value in raw.items()}
        file_path = posixpath.normpath(raw["file_path"].replace("\\", "/"))
        if not raw["file_path"].strip() or file_path == ".":
            raise ValueError("file_path cannot be empty")
        dataset = raw.get("dataset", "").strip()
        file_key = (dataset, file_path)
        channel = _integer(raw["wav_channel"], "wav_channel")
        start = _integer(raw["wav_start"], "wav_start")
        end = _integer(raw["wav_end"], "wav_end")
        if end <= start:
            raise ValueError("wav_end must be after wav_start")
        dimension = raw["dimension"].strip().lower()
        if dimension not in {"mos", "noi", "dis", "col", "loud"}:
            raise ValueError(f"Invalid quality dimension: {dimension}")
        group_key = (*file_key, channel, start, end, dimension)
        index = _integer(raw["phoneme_index"], "phoneme_index")
        frame_start = _integer(raw["start_frame"], "start_frame")
        frame_end = _integer(raw["end_frame"], "end_frame")
        if frame_end < frame_start:
            raise ValueError("end_frame is inclusive and must be >= start_frame")
        seconds_start = _number(raw["start_seconds"], "start_seconds")
        seconds_end = _number(raw["end_seconds"], "end_seconds")
        if not 0 <= seconds_start < seconds_end:
            raise ValueError("Phoneme time interval must have positive duration")
        label_text = raw.get("label", "").strip()
        split = raw.get("split", "").strip().lower()
        if bool(label_text) != bool(split):
            raise ValueError("label and split must either both be filled or both be empty")
        label = None
        if label_text:
            label = _number(label_text, "label")
            if label not in {0, 1} or split not in {"dev", "test"}:
                raise ValueError("label must be 0/1 and split must be dev/test")
            label = int(label)
            previous = file_splits.setdefault(file_key, split)
            if previous != split:
                raise ValueError(f"File appears in both dev and test: {file_path}")
        entry = {
            "row_id": len(entries), "raw": raw, "file_key": file_key,
            "group_key": group_key, "dimension": dimension, "index": index,
            "start_frame": frame_start, "end_frame": frame_end,
            "start_seconds": seconds_start, "end_seconds": seconds_end,
            "score": _number(raw["score"], "score"), "label": label,
            "split": split or "unlabelled",
        }
        groups.setdefault(group_key, []).append(entry)
        entries.append(entry)
    for group_key, group in groups.items():
        group.sort(key=lambda row: (row["start_frame"], row["index"]))
        if [row["index"] for row in group] != list(range(len(group))):
            raise ValueError(f"Missing, duplicate, or nonchronological phoneme indices: {group_key}")
        if len({row["split"] for row in group}) != 1:
            raise ValueError("Each utterance/dimension group must be fully labelled or fully unlabelled")
        if group[0]["start_frame"] != 0 or any(
            current["start_frame"] != previous["end_frame"] + 1
            for previous, current in zip(group, group[1:])
        ):
            raise ValueError("Phoneme intervals must cover contiguous local frames")
        if any(current["start_seconds"] < previous["end_seconds"] - 1e-9
               for previous, current in zip(group, group[1:])):
            raise ValueError("Phoneme intervals overlap within one utterance/dimension group")
        for row in group:
            count = row["raw"].get("phoneme_count", "").strip()
            if count and _integer(count, "phoneme_count") != len(group):
                raise ValueError("Incomplete group: phoneme_count does not match the CSV rows")
    return fields, entries, groups


def binary_metrics(labels, selected):
    labels = np.asarray(labels)
    selected = np.asarray(selected)
    if labels.ndim != 1 or labels.shape != selected.shape or not np.isin(labels, [0, 1]).all() or not np.isin(selected, [0, 1]).all():
        raise ValueError("Binary labels and selections must have matching one-dimensional shapes")
    labels = labels.astype(int)
    selected = selected.astype(bool)
    tp = int(np.sum((labels == 1) & selected))
    fp = int(np.sum((labels == 0) & selected))
    fn = int(np.sum((labels == 1) & ~selected))
    tn = int(np.sum((labels == 0) & ~selected))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else None}


def _atomic_write(path, text):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    os.replace(temporary, path)


def _csv_text(fields, rows):
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def compare(input_csv, output_dir):
    fields, entries, groups = load_scores(input_csv)
    comparisons, selections = [], []
    buckets = {}
    for entry in entries:
        for dimension in (entry["dimension"], "ALL"):
            buckets.setdefault((entry["split"], dimension), []).append(entry)
    for rule in RULES:
        ranks, cutoffs = {}, {}
        for key in sorted(groups):
            group = groups[key]
            scores = [entry["score"] for entry in group]
            chosen, cutoff = select_indices(
                scores, rule.kind, k=rule.value,
                adaptive_lambda=rule.value if rule.kind == "adaptive" else 1.0)
            for rank, position in enumerate(chosen, 1):
                ranks[group[int(position)]["row_id"]] = rank
            cutoffs[key] = cutoff
        for entry in entries:
            rank = ranks.get(entry["row_id"], "")
            selections.append({
                **entry["raw"], "method": rule.name, "method_rule": rule.kind,
                "method_k": rule.value if rule.kind == "percent" else "",
                "method_lambda": rule.value if rule.kind == "adaptive" else "",
                "method_cutoff": cutoffs[entry["group_key"]],
                "method_selected": int(rank != ""), "method_rank": rank,
            })
        for (split, dimension), group in sorted(buckets.items()):
            selected = [entry["row_id"] in ranks for entry in group]
            n_labelled = sum(entry["label"] is not None for entry in group)
            row = {
                "method": rule.name, "rule": rule.kind,
                "k": rule.value if rule.kind == "percent" else "",
                "adaptive_lambda": rule.value if rule.kind == "adaptive" else "",
                "split": split, "dimension": dimension,
                "n_files": len({entry["file_key"] for entry in group}),
                "n_groups": len({entry["group_key"] for entry in group}),
                "n_phonemes": len(group), "n_selected": sum(selected),
                "selection_rate": sum(selected) / len(group), "n_labelled": n_labelled,
            }
            if n_labelled:
                labels = [entry["label"] for entry in group]
                row.update(binary_metrics(labels, selected))
                row["n_positive"] = sum(labels)
                row["n_negative"] = len(labels) - sum(labels)
            else:
                row.update({key: None for key in [
                    "n_positive", "n_negative", "tp", "fp", "fn", "tn",
                    "precision", "recall", "f1", "fpr"]})
            comparisons.append(row)
    dev = [row for row in comparisons if row["split"] == "dev" and row["dimension"] == "ALL"]
    selection = {
        "input_csv": str(Path(input_csv).resolve()),
        "input_sha256": hashlib.sha256(Path(input_csv).read_bytes()).hexdigest(),
        "adaptive_std_ddof": 0, "adaptive_comparison": ">", "percent_rounding": "floor",
        "status": "not_selected", "reason": "No labelled dev groups",
        "selection_split": "dev", "ranking": ["micro_f1", "precision", "recall", "fixed_method_order"],
        "method_order": [rule.name for rule in RULES],
        "best_adaptive": None, "best_percent": None, "best_overall": None,
    }
    if dev and dev[0]["n_positive"]:
        def choose(candidates):
            best = max(candidates, key=lambda row: (row["f1"], row["precision"], row["recall"]))
            test = next((row for row in comparisons
                         if row["method"] == best["method"] and row["split"] == "test"
                         and row["dimension"] == "ALL"), None)
            return {"method": best["method"], "dev": best, "test": test}
        selection.update(
            status="selected", reason="Chosen only by labelled dev micro metrics",
            best_adaptive=choose([row for row in dev if row["rule"] == "adaptive"]),
            best_percent=choose([row for row in dev if row["rule"] == "percent"]),
            best_overall=choose(dev),
        )
    elif dev:
        selection["reason"] = "Dev has no positive error labels"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_dir / "comparison.csv", _csv_text(COMPARISON_FIELDS, comparisons))
    _atomic_write(output_dir / "selections.csv", _csv_text(fields + SELECTION_FIELDS, selections))
    _atomic_write(output_dir / "selected_on_dev.json",
                  json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"comparison": comparisons, "selection": selection, "n_rows": len(entries),
            "n_groups": len(groups), "n_files": len({entry["file_key"] for entry in entries})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compare(args.input_csv, args.output_dir)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Comparison failed: {error}\n")
    print(f"Compared 6 rules on {result['n_rows']} phonemes, "
          f"{result['n_groups']} groups, {result['n_files']} files.")
    print(f"Dev selection: {result['selection']['status']} - {result['selection']['reason']}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
