"""CSV schema and time conversion for every scored phoneme (standard library only)."""

PHONEME_SCORE_FIELDS = [
    "dataset", "file_path", "file_num", "wav_channel", "wav_segment",
    "wav_start", "wav_end", "file_fs", "dimension", "quality_score",
    "phoneme_index", "phoneme_count", "phoneme_id", "phoneme",
    "start_frame", "end_frame", "start_seconds", "end_seconds",
    "score", "selected", "selection_rank", "k_method", "k",
    "adaptive_lambda", "cutoff", "saliency_method", "pdsm_preprocess",
    "pdsm_pool", "label", "split",
]


def build_export_rows(row, dimension, records, config, frame_hop=0.01):
    """Convert inclusive local frame bounds to [start,end) on the original WAV."""
    sample_rate = float(row["file_fs"])
    if sample_rate <= 0 or frame_hop <= 0:
        raise ValueError("Sample rate and frame hop must be positive")
    base_time = int(row["wav_start"]) / sample_rate
    segment_end = int(row["wav_end"]) / sample_rate
    method = config["pdsm"]["k_method"]
    context = {
        "dataset": config["dataset_name"], "file_path": str(row["file_path"]),
        "file_num": int(row["file_num"]), "wav_channel": int(row["wav_channel"]),
        "wav_segment": int(row["wav_segment"]), "wav_start": int(row["wav_start"]),
        "wav_end": int(row["wav_end"]), "file_fs": int(sample_rate),
        "dimension": dimension, "quality_score": float(row[f"sq_{dimension}"]),
        "k_method": method,
        "k": config["pdsm"]["k"] if method != "adaptive" else "",
        "adaptive_lambda": config["pdsm"].get("adaptive_lambda", 1.0) if method == "adaptive" else "",
        "saliency_method": config["saliency"]["saliency_method"],
        "pdsm_preprocess": config["pdsm"]["preprocess"],
        "pdsm_pool": config["pdsm"]["pool"], "label": "", "split": "",
    }
    output = []
    for record in records:
        start = base_time + record["start_frame"] * frame_hop
        end = min(segment_end, base_time + (record["end_frame"] + 1) * frame_hop)
        if not base_time <= start < end <= segment_end:
            raise ValueError("Phoneme frame interval lies outside its WAV segment")
        output.append({**context, **record, "start_seconds": start, "end_seconds": end})
    return output
