import numpy as np


THRESHOLD_VALUE = 0.4
COUNT_AS_DOUBLE_PAD = 5 # 50 ms


def thresh_abs(x):
    out_x = np.abs(x)
    out_x[out_x < THRESHOLD_VALUE] = 0.0
    return out_x


def l2_norm(x):
    return np.power(np.sum(np.power(x, 2)), 1/2)


def l1_norm(x):
    return np.sum(np.abs(x))


def safe_max(x):
    if len(x.flatten()) > 0:
        if (x.flatten() > 0).any():
            return np.max(x.flatten())
    return 0


def get_preprocess_and_pool(config_file):
    """
    Returns the preprocess and pooling functions wanted from the config file

    Parameters
    ----------
    config_file (dict) : input config file detailing the options to run the evaluation with

    Returns
    ----------
    preprocess_fn : Function determining the preprocess function as detailed in paper
    pool_fn : Function determining the pooling function as detailed in paper
    """
    
    preprocess = {"abs": np.abs, "thresh_abs": thresh_abs}
    pools = {"mean": np.mean, "median": np.median, "max": safe_max,
             "sum": np.sum, "l1_norm": l1_norm, "l2_norm": l2_norm}
    try:
        return preprocess[config_file["pdsm"]["preprocess"]], pools[config_file["pdsm"]["pool"]]
    except KeyError as error:
        raise ValueError("Unknown PDSM preprocessing or pooling method") from error


def select_indices(scores, k_method="percent", k=0.1, adaptive_lambda=1.0):
    """Select pooled phoneme scores, breaking ties by input/time order.

    Adaptive uses population std (ddof=0) and strict score > mean+lambda*std.
    No minimum count is imposed. The returned cutoff is defined only for
    adaptive selection; percent/count selection is rank based.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Phoneme scores must be a finite one-dimensional array")
    if k_method not in {"percent", "threshold", "adaptive"}:
        raise ValueError(f"Unknown PDSM k_method: {k_method}")
    if k_method == "adaptive":
        if isinstance(adaptive_lambda, (bool, np.bool_)):
            raise ValueError("adaptive_lambda must be a finite non-negative number")
        scale = float(adaptive_lambda)
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("adaptive_lambda must be a finite non-negative number")
        if not len(values):
            return np.array([], dtype=int), None
        cutoff = float(values.mean() + scale * values.std(ddof=0))
        candidates = np.flatnonzero(values > cutoff)
        order = np.argsort(-values[candidates], kind="stable")
        return candidates[order], cutoff
    if isinstance(k, (bool, np.bool_)):
        raise ValueError("PDSM k must be numeric, not boolean")
    number = float(k)
    if not np.isfinite(number):
        raise ValueError("PDSM k must be finite")
    if k_method == "percent":
        if not 0 <= number <= 1:
            raise ValueError("Percent k must be in [0, 1]")
        count = int(np.floor(len(values) * number))
    else:
        if number < 0 or not number.is_integer():
            raise ValueError("Threshold k must be a non-negative integer count")
        count = min(len(values), int(number))
    return np.argsort(-values, kind="stable")[:count], None


def score_phonemes(saliency_map, ppg, ppg_dict, preprocess_fn, pool_fn):
    """Pool every contiguous phoneme, including its inclusive final frame."""
    saliency = np.asarray(saliency_map)
    posterior = np.asarray(ppg)
    if saliency.ndim != 2 or posterior.ndim != 2:
        raise ValueError("Saliency and PPG must be two-dimensional")
    if saliency.shape[1] != posterior.shape[1]:
        raise ValueError("Saliency and PPG must have the same time length")
    if not np.isfinite(saliency).all() or not np.isfinite(posterior).all():
        raise ValueError("Saliency and PPG must be finite")
    if not saliency.shape[1]:
        return [], np.array([], dtype=np.float64)
    if not saliency.shape[0] or not posterior.shape[0]:
        raise ValueError("Saliency and PPG need non-empty frequency/phoneme axes")
    processed = np.asarray(preprocess_fn(saliency))
    if processed.shape != saliency.shape or not np.isfinite(processed).all():
        raise ValueError("PDSM preprocessing must preserve shape and finite values")
    labels = np.argmax(posterior, axis=0)
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes - 1, len(labels) - 1]
    phonemes, energies = [], []
    for start, end in zip(starts, ends):
        phoneme_id = int(labels[start])
        key = str(phoneme_id) if str(phoneme_id) in ppg_dict else phoneme_id
        if key not in ppg_dict:
            raise ValueError(f"Missing phoneme label for id {phoneme_id}")
        phonemes.append((phoneme_id, ppg_dict[key], int(start), int(end)))
        energies.append(float(pool_fn(processed[:, start:end + 1])))
    energies = np.asarray(energies, dtype=np.float64)
    if not np.isfinite(energies).all():
        raise ValueError("PDSM pooling produced non-finite scores")
    return phonemes, energies


def PDSM(saliency_map, ppg, ppg_dict, preprocess_fn, pool_fn, k_method, k,
         adaptive_lambda=1.0, return_scores=False):
    """
    Runs PDSM algorithm as described in [1]S. Gupta, M. Ravanelli, P. Germain, and C. Subakan, “Phoneme Discretized Saliency Maps for Explainable Detection of AI-Generated Voice,” Sep. 2024, [Online]. Available: http://arxiv.org/abs/2406.10422

    Parameters
    ----------
    saliency_map (numpy.array) : Saliency map for a given input to the SQ_AST model
    ppg (numpy.array) : Phoneme PosteriorGram showing the Confidence of phonemes over time
    ppg_dict (dict) : Dictionary showing the translation of indeces to phonemes
    preprocess_fn : Function determining the preprocess function as detailed in paper
    pool_fn : Function determining the pooling function as detailed in paper
    k_method (str) : "threshold" for a count, "percent" for a fraction,
        or "adaptive" for score > mean + adaptive_lambda * population std
    k (float) : count/fraction; unused by adaptive selection
    adaptive_lambda (float) : finite non-negative adaptive cutoff scale
    return_scores (bool) : also return records for all phonemes when true

    Returns
    ----------
    m_out (numpy.array) : Output mask of most important phonemes
    phonemes_return (list) : List of dicts with information on each important phoneme
    records (list, optional) : all scores and selection metadata
    """

    phonemes, phoneme_energy = score_phonemes(
        saliency_map, ppg, ppg_dict, preprocess_fn, pool_fn)
    max_indices, cutoff = select_indices(
        phoneme_energy, k_method, k, adaptive_lambda)
    m_out = np.zeros_like(saliency_map, dtype=np.float32)
    phonemes_return = []
    for max_indices_index, phoneme_index in enumerate(max_indices):
        _, _, start, end = phonemes[phoneme_index]
        m_out[:, start:end + 1] = 0.5 - 0.2*(max_indices_index/len(max_indices))
        phonemes_return.append(phonemes[phoneme_index])
    if return_scores:
        ranks = {int(index): rank + 1 for rank, index in enumerate(max_indices)}
        records = [
            {"phoneme_index": index, "phoneme_count": len(phonemes),
             "phoneme_id": int(phoneme[0]), "phoneme": str(phoneme[1]),
             "start_frame": phoneme[2], "end_frame": phoneme[3],
             "score": float(phoneme_energy[index]), "selected": int(index in ranks),
             "selection_rank": ranks.get(index, ""), "cutoff": cutoff}
            for index, phoneme in enumerate(phonemes)
        ]
        return m_out, phonemes_return, records
    return m_out, phonemes_return


def get_bulk_hists_for_system(df, dim):
    """
    Gets the system-wide histograms for most important phonemes

    Parameters
    ----------
    df (pandas.dataframe) : Input dataframe to gather hist data
    dim (str) : Dimension to run analysis on
    
    Returns
    ----------
    single_phoneme_hist (dict) : Histogram data for single phonemes (returns None if empty)
    double_phoneme_hist (dict) : Histogram data for phonemes-pairs (returns None if empty)
    """

    single_phoneme_hist = {}
    double_phoneme_hist = {}

    for index, row in df.iterrows():
        phoneme_info = row[f"pdsm_sq_{dim}"]
        # sort in terms of start times
        phoneme_info = sorted(phoneme_info, key=lambda x: x[2])
        # iterate over list of phonemes
        for index in range(len(phoneme_info)):
            phoneme = phoneme_info[index]
            if phoneme[1] not in single_phoneme_hist.keys():
                single_phoneme_hist[phoneme[1]] = 1
            else:
                single_phoneme_hist[phoneme[1]] += 1

            if index < len(phoneme_info) - 1:
                next_phoneme = phoneme_info[index+1]
                if phoneme[3] == next_phoneme[2] - COUNT_AS_DOUBLE_PAD:
                    phoneme_transition = phoneme[1] + "-" + next_phoneme[1]
                    if phoneme_transition not in double_phoneme_hist.keys():
                        double_phoneme_hist[phoneme_transition] = 1
                    else:
                        double_phoneme_hist[phoneme_transition] += 1

    # get proportions rather than quantities
    single_phoneme_hist_total = sum(single_phoneme_hist.values())
    if single_phoneme_hist_total > 0:
        single_phoneme_hist = {k: v / single_phoneme_hist_total for k, v in single_phoneme_hist.items()}
        single_phoneme_hist = dict(sorted(single_phoneme_hist.items(), key=lambda item: item[1], reverse=True))
    else:
        single_phoneme_hist = None

    double_phoneme_hist_total = sum(double_phoneme_hist.values())
    if double_phoneme_hist_total > 0:
        double_phoneme_hist = {k: v / double_phoneme_hist_total for k, v in double_phoneme_hist.items()}
        double_phoneme_hist = dict(sorted(double_phoneme_hist.items(), key=lambda item: item[1], reverse=True))
    else:
        double_phoneme_hist = None

    return single_phoneme_hist, double_phoneme_hist

