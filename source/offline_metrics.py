from collections import Counter

import numpy as np


def _percentile_interval(values, confidence=0.95):
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return [round(float(lower), 6), round(float(upper), 6)]


def _bootstrap_mean(values, samples, rng):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or samples < 1:
        value = float(values.mean()) if len(values) else 0.0
        return [round(value, 6), round(value, 6)]
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    return _percentile_interval(estimates)


def _stratified_macro_recall(
    rows,
    label_key,
    samples,
    rng,
    cluster_key=None,
):
    strata = {}
    for row in rows:
        strata.setdefault(row[label_key], []).append(float(row["correct"]))
    recalls = [float(np.mean(values)) for values in strata.values()]
    score = float(np.mean(recalls)) if recalls else 0.0
    if samples < 1:
        rounded = round(score, 6)
        return score, [rounded, rounded]

    if cluster_key is not None:
        clusters = {}
        for row in rows:
            clusters.setdefault(row[cluster_key], []).append(row)
        if len(clusters) < 2:
            rounded = round(score, 6)
            return score, [rounded, rounded]
        cluster_rows = list(clusters.values())
        estimates = np.empty(samples, dtype=float)
        for sample_index in range(samples):
            sampled_indices = rng.integers(
                0,
                len(cluster_rows),
                size=len(cluster_rows),
            )
            sampled_rows = [
                row
                for cluster_index in sampled_indices
                for row in cluster_rows[cluster_index]
            ]
            sampled_recalls = [
                float(
                    np.mean(
                        [
                            bool(row["correct"])
                            for row in sampled_rows
                            if row[label_key] == label
                        ]
                    )
                )
                for label in strata
            ]
            estimates[sample_index] = float(np.mean(sampled_recalls))
        return score, _percentile_interval(estimates)

    if any(len(values) < 2 for values in strata.values()):
        rounded = round(score, 6)
        return score, [rounded, rounded]

    estimates = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        sampled_recalls = []
        for values in strata.values():
            values = np.asarray(values, dtype=float)
            indices = rng.integers(0, len(values), size=len(values))
            sampled_recalls.append(float(values[indices].mean()))
        estimates[sample_index] = float(np.mean(sampled_recalls))
    return score, _percentile_interval(estimates)


def _semantic_self_prediction(row):
    choice = row.get("choice")
    options = row.get("answer_options", [])
    if not isinstance(choice, int) or not 1 <= choice <= len(options):
        return None
    return str(options[choice - 1]).strip().lower().startswith("yes")


def summarize_result_group(rows, bootstrap_samples=5000, seed=0):
    if not rows:
        raise ValueError("At least one result is required.")
    episode_ids = [row["episode_id"] for row in rows]
    duplicates = [
        episode_id
        for episode_id, count in Counter(episode_ids).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate episode results: {duplicates[:5]}")

    rng = np.random.default_rng(int(seed))
    correct = [float(bool(row.get("correct"))) for row in rows]
    valid = [float(bool(row.get("valid"))) for row in rows]
    pair_ids = [row.get("nuisance_pair_id") for row in rows]
    pair_counts = Counter(pair_ids)
    use_pair_bootstrap = bool(pair_ids) and all(pair_ids) and set(
        pair_counts.values()
    ) == {2}
    if use_pair_bootstrap:
        accuracy_bootstrap_values = [
            float(
                np.mean(
                    [
                        bool(row.get("correct"))
                        for row in rows
                        if row["nuisance_pair_id"] == pair_id
                    ]
                )
            )
            for pair_id in pair_counts
        ]
        bootstrap_unit = "nuisance_pair"
        independent_units = len(pair_counts)
        cluster_key = "nuisance_pair_id"
    else:
        accuracy_bootstrap_values = correct
        bootstrap_unit = "episode"
        independent_units = len(rows)
        cluster_key = None
    summary = {
        "episodes": len(rows),
        "bootstrap_unit": bootstrap_unit,
        "independent_units": independent_units,
        "accuracy": round(float(np.mean(correct)), 6),
        "accuracy_ci95": _bootstrap_mean(
            accuracy_bootstrap_values,
            bootstrap_samples,
            rng,
        ),
        "invalid_rate": round(1.0 - float(np.mean(valid)), 6),
    }

    target_presence_values = {
        row.get("target_present") for row in rows
    }
    if target_presence_values == {True, False}:
        by_label = {
            label: [row for row in rows if row.get("target_present") is label]
            for label in (True, False)
        }
        self_recall = float(
            np.mean([bool(row["correct"]) for row in by_label[True]])
        )
        nonself_recall = float(
            np.mean([bool(row["correct"]) for row in by_label[False]])
        )
        balanced, balanced_ci = _stratified_macro_recall(
            rows,
            "target_present",
            bootstrap_samples,
            rng,
            cluster_key=cluster_key,
        )
        semantic_predictions = [
            prediction
            for prediction in (_semantic_self_prediction(row) for row in rows)
            if prediction is not None
        ]
        summary.update(
            {
                "metric_family": "binary_self_other",
                "self_recall": round(self_recall, 6),
                "nonself_recall": round(nonself_recall, 6),
                "balanced_accuracy": round(balanced, 6),
                "balanced_accuracy_ci95": balanced_ci,
                "self_attribution_rate": round(
                    float(np.mean(semantic_predictions)),
                    6,
                )
                if semantic_predictions
                else None,
            }
        )
        return summary

    target_positions = sorted(
        {row.get("target_index") for row in rows if row.get("target_index") is not None}
    )
    target_recall = {}
    for position in target_positions:
        position_rows = [row for row in rows if row.get("target_index") == position]
        target_recall[str(position)] = round(
            float(np.mean([bool(row["correct"]) for row in position_rows])),
            6,
        )
    predicted = Counter(
        row.get("choice") for row in rows if bool(row.get("valid"))
    )
    valid_count = sum(predicted.values())
    predicted_distribution = {
        str(position): round(predicted[position] / valid_count, 6)
        if valid_count
        else 0.0
        for position in target_positions
    }
    macro_recall, macro_recall_ci = _stratified_macro_recall(
        rows,
        "target_index",
        bootstrap_samples,
        rng,
        cluster_key=cluster_key,
    )
    summary.update(
        {
            "metric_family": "multi_arm_identification",
            "target_position_recall": target_recall,
            "predicted_position_distribution": predicted_distribution,
            "macro_position_recall": round(macro_recall, 6),
            "macro_position_recall_ci95": macro_recall_ci,
        }
    )
    return summary
