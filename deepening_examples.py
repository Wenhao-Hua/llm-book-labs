"""Recompute the book's constructed analytical examples using only the standard library."""
import argparse
import itertools
import json
import math
from pathlib import Path


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def examples():
    forward = 0.0
    reverse = 0.0
    for label in [0, 1]:
        forward -= 0.2 * (sigmoid(forward) - label)
    for label in [1, 0]:
        reverse -= 0.2 * (sigmoid(reverse) - label)
    paths = {"AE": 0.6 * 0.5, "AX": 0.6 * 0.5,
             "BE": 0.4 * 0.9, "BX": 0.4 * 0.1}
    choices = []
    for a, b, c in itertools.product([0, 1], repeat=3):
        length = 650 * a + 250 * b + 500 * c
        if length <= 1000 and a <= b:
            choices.append({"A": a, "B": b, "C": c, "length": length,
                            "score": 8 * a + 2 * b + 7 * c})
    best = max(choices, key=lambda row: row["score"])
    advantage = math.log(1.5)
    preference_loss = -0.6 * math.log(sigmoid(advantage)) - 0.4 * math.log(sigmoid(-advantage))
    embeddings = {"a": (1, 0), "b": (0, 1), "c": (1, 1)}
    def mean(tokens):
        return [sum(embeddings[token][axis] for token in tokens) / len(tokens) for axis in range(2)]
    needed = {(q, part) for q in range(100) for part in range(2)}
    concentrated = needed - {(q, part) for q in range(10) for part in range(2)}
    distributed = needed - {(q, 0) for q in range(20)}
    def completeness(found):
        return sum(all((q, part) in found for part in range(2)) for q in range(100)) / 100
    target = ((1, 0), (0, 1))
    candidate = ((1, 0), (0, 0))
    def squared_error(update):
        return sum((target[i][j] - update[i][j]) ** 2 for i in range(2) for j in range(2))
    report = {
        "scope": "Constructed examples, not pretrained-model measurements.",
        "chapter02": {"mean_abc": mean("abc"), "mean_cba": mean("cba")},
        "chapter03": {"full_batch_bias": 0, "labels_0_1": forward, "labels_1_0": reverse},
        "chapter04": {"joint_probabilities": paths, "best": max(paths, key=paths.get)},
        "chapter05": {"balanced": {"A": (0.9 + 0.6) / 2, "B": (0.95 + 0.4) / 2},
                      "target": {"A": 0.9 * 0.9 + 0.1 * 0.6, "B": 0.9 * 0.95 + 0.1 * 0.4}},
        "chapter06": {"feasible": choices, "best": best},
        "chapter07": {"item_recall": len(concentrated) / len(needed), "concentrated_complete": completeness(concentrated),
                      "distributed_complete": completeness(distributed)},
        "chapter09": {"direct": 0.08 * 100,
                      "ask": 1 + 0.8 * 0.01 * 100 + 0.2 * (2 + 0.01 * 100)},
        "chapter10": {"zero_squared_error": squared_error(((0, 0), (0, 0))), "rank_one_candidate_squared_error": squared_error(candidate)},
        "chapter11": {"optimal_advantage": advantage, "preferred_fraction": sigmoid(advantage),
                      "loss": preference_loss},
        "chapter12": {"original_seconds": 10, "new_seconds": 6 + 4 / 4,
                      "speedup": 10 / (6 + 4 / 4), "limit": 1 / 0.6},
        "chapter15": {"old_model_old_index": 0.8, "old_model_new_index": 0.88,
                      "new_model_old_index": 0.76, "new_model_new_index": 0.84}
    }
    assert math.isclose(sum(paths.values()), 1) and paths["BE"] > paths["AE"]
    assert math.isclose(forward, -reverse) and forward > 0
    assert (best["A"], best["B"], best["C"]) == (1, 1, 0)
    assert math.isclose(report["chapter09"]["ask"], 2.4)
    assert math.isclose(sigmoid(advantage), 0.6)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("deepening-results.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(examples(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
