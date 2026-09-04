from collections import Counter


def _rate(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def summarize_logic_audit(base_rows, audit_rows):
    base_ids = [row["episode_id"] for row in base_rows]
    duplicate_base = [
        episode_id
        for episode_id, count in Counter(base_ids).items()
        if count > 1
    ]
    if duplicate_base:
        raise ValueError(f"Duplicate ordinary results: {duplicate_base[:5]}")

    audit_ids = [row["episode_id"] for row in audit_rows]
    duplicate_audits = [
        episode_id
        for episode_id, count in Counter(audit_ids).items()
        if count > 1
    ]
    if duplicate_audits:
        raise ValueError(f"Duplicate logic audits: {duplicate_audits[:5]}")

    initially_correct_ids = {
        row["episode_id"] for row in base_rows if bool(row.get("correct"))
    }
    unexpected = sorted(set(audit_ids) - initially_correct_ids)
    if unexpected:
        raise ValueError(
            "Logic audits may only exist for initially correct results: "
            f"{unexpected[:5]}"
        )

    audited_ids = set(audit_ids)
    passed = sum(bool(row.get("logic_audit_pass")) for row in audit_rows)
    initially_correct = len(initially_correct_ids)
    completed = len(audit_rows)
    pending = len(initially_correct_ids - audited_ids)
    complete = pending == 0
    total = len(base_rows)

    return {
        "ordinary_results": total,
        "initially_correct": initially_correct,
        "initial_accuracy": _rate(initially_correct, total),
        "logic_audits_completed": completed,
        "logic_audits_pending": pending,
        "logic_audit_complete": complete,
        "logic_audit_pass": passed,
        "logic_audit_fail": completed - passed,
        "logic_audit_pass_rate_among_completed": _rate(passed, completed),
        "logic_audit_pass_rate_among_initially_correct": (
            _rate(passed, initially_correct) if complete else None
        ),
        "logic_adjusted_correct": passed,
        "logic_adjusted_accuracy": _rate(passed, total) if complete else None,
        "logic_adjusted_accuracy_lower_bound": _rate(passed, total),
    }
