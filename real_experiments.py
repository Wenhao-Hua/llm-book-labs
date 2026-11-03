"""Small, real pretrained-model experiments. Offline by default; no external writes.

Run from this directory: python real_experiments.py --experiment all
The manually authored data are demonstrations, not an independent benchmark.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parent
MODELS = {
    "retrieval": ("sentence-transformers/all-MiniLM-L6-v2", "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"),
    "finetune": ("prajjwal1/bert-tiny", "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"),
    "cache": ("gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_path(kind, cache_root):
    repo, revision = MODELS[kind]
    path = cache_root / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(f"Missing pinned model {repo}@{revision}. See real-experiments/README.txt; no automatic download.")
    files = [p for p in path.rglob("*") if p.is_file()]
    return path, {"repo": repo, "revision": revision,
                  "files": {str(p.relative_to(path)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(files)}}


def retrieval(data, path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).eval()

    def encode(texts):
        batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            return F.normalize(pooled, p=2, dim=1)

    docs = data["documents"]
    vectors = encode([d["text"] for d in docs])
    queries = encode([q["text"] for q in data["queries"]])
    scores = (queries @ vectors.T).tolist()
    records = []
    for query, values in zip(data["queries"], scores):
        order = sorted(range(len(docs)), key=lambda i: (-values[i], docs[i]["id"]))
        available = [i for i in order if docs[i]["current"] and query["role"] in docs[i]["roles"]]
        relevant = set(query["relevant"])
        def rank(indices):
            return [{"id": docs[i]["id"], "cosine": values[i]} for i in indices]
        def recall(indices, k):
            return len(relevant & {docs[i]["id"] for i in indices[:k]}) / len(relevant)
        records.append({**query, "raw_ranking": rank(order), "filtered_ranking": rank(available),
                        "raw_recall_at_1": recall(order, 1), "filtered_recall_at_1": recall(available, 1),
                        "filtered_recall_at_3": recall(available, 3)})
    pairs = ["Employees may borrow cameras.", "Employees may not borrow cameras."]
    negation = encode(pairs)
    # Padding invariance checks whether pooling accidentally counts padded positions.
    separate = encode([pairs[0]])
    together = encode([pairs[0], "This longer sentence introduces additional padding for the first item in the same batch."])
    padding_error = (separate[0] - together[0]).abs().max().item()
    assert padding_error < 1e-5
    assert all(all(d["current"] and q["role"] in d["roles"] for r in q["filtered_ranking"] for d in docs if d["id"] == r["id"]) for q in records)
    return {"scope": "Real pretrained sentence embeddings on authored English teaching data; not Chinese retrieval or factual entailment evaluation.",
            "embedding_shape": list(vectors.shape), "max_length": 256, "pooling": "attention-mask mean then L2 normalization",
            "records": records, "negation_texts": pairs, "negation_cosine": float(negation[0] @ negation[1]),
            "padding_max_abs_difference": padding_error,
            "mean_filtered_recall_at_1": sum(r["filtered_recall_at_1"] for r in records) / len(records),
            "mean_filtered_recall_at_3": sum(r["filtered_recall_at_3"] for r in records) / len(records)}


def finetune(data, path, out):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    dataset = data["classification"]
    train, holdout = dataset["train"], dataset["holdout"]
    assert not ({r["text"] for r in train} & {r["text"] for r in holdout})
    def batch(rows):
        x = tokenizer([r["text"] for r in rows], padding=True, truncation=True, max_length=64, return_tensors="pt")
        return x, torch.tensor([r["label"] for r in rows], dtype=torch.long)
    train_x, train_y = batch(train)
    eval_x, eval_y = batch(holdout)
    runs = []
    for seed in (7, 17, 29):
        random.seed(seed)
        torch.manual_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2, local_files_only=True)
        original = model.bert.encoder.layer[0].attention.self.query.weight.detach().clone()
        def evaluate(x, y):
            model.eval()
            with torch.inference_mode():
                output = model(**x, labels=y)
                preds = output.logits.argmax(-1)
            return {"loss": output.loss.item(), "correct": int((preds == y).sum()), "count": len(y),
                    "predictions": preds.tolist(), "logits": output.logits.tolist()}
        initial = evaluate(eval_x, eval_y)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        trajectory = []
        start = time.perf_counter()
        # Fixed 20 epochs, 3 batches each. No early stopping or choice based on holdout scores.
        for epoch in range(20):
            model.train()
            order = torch.randperm(len(train))
            total = 0.0
            for ids in order.split(8):
                optimizer.zero_grad(set_to_none=True)
                output = model(**{k: v[ids] for k, v in train_x.items()}, labels=train_y[ids])
                assert torch.isfinite(output.loss)
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += output.loss.item() * len(ids)
            trajectory.append({"epoch": epoch + 1, "training_batch_loss": total / len(train)})
        elapsed = time.perf_counter() - start
        final_train, final = evaluate(train_x, train_y), evaluate(eval_x, eval_y)
        changed = (model.bert.encoder.layer[0].attention.self.query.weight - original).abs().max().item()
        assert changed > 0
        checkpoint = out / f"finetune-seed-{seed}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        reloaded = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).eval()
        with torch.inference_mode():
            reload_logits = reloaded(**eval_x).logits
        delta = (reload_logits - torch.tensor(final["logits"])).abs().max().item()
        assert delta < 1e-6
        runs.append({"seed": seed, "initial_holdout": initial, "final_train": final_train, "final_holdout": final,
                     "training_seconds": elapsed, "encoder_weight_max_change": changed, "reload_max_abs_difference": delta,
                     "trajectory": trajectory, "checkpoint_sha256": sha(checkpoint / "model.safetensors")})
        print("finetune", seed, "holdout", final["correct"], "/", final["count"], flush=True)
        del model, optimizer, reloaded
        gc.collect()
    return {"scope": "Actual full fine-tuning of a small pretrained BERT on authored English intent data; no language-generation fine-tuning, no LoRA, no independent deployment benchmark.",
            "labels": dataset["labels"], "train_count": len(train), "holdout_count": len(holdout),
            "holdout_role": "author-designed held-out exercise; no claim of independent sampling or absence of pretraining contamination",
            "majority_baseline_correct": max(sum(r["label"] == i for r in holdout) for i in (0, 1)),
            "config": {"epochs": 20, "batch_size": 8, "steps": 60, "lr": 2e-4, "weight_decay": 0.01, "max_length": 64, "gradient_clip": 1.0},
            "holdout_rows": holdout, "runs": runs}


def cache_experiment(data, path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, attn_implementation="eager").eval()
    prompt = data["cache_prompt"]
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        prefix = model(ids, use_cache=True)
        new_id = prefix.logits[:, -1].argmax(-1, keepdim=True)
        full = model(torch.cat([ids, new_id], dim=1), use_cache=False)
        incremental = model(new_id, past_key_values=prefix.past_key_values, use_cache=True)
        max_error = (full.logits[:, -1] - incremental.logits[:, -1]).abs().max().item()
        assert torch.allclose(full.logits[:, -1], incremental.logits[:, -1], atol=3e-4, rtol=1e-5)
        past = incremental.past_key_values
        pairs = past.to_legacy_cache() if hasattr(past, "to_legacy_cache") else tuple(past)
        actual_bytes = sum(t.numel() * t.element_size() for pair in pairs for t in pair)
        c = model.config
        expected_bytes = 2 * c.n_layer * (ids.shape[1] + 1) * c.n_embd * 4
        assert actual_bytes == expected_bytes

    def generate(use_cache):
        running, past = ids.clone(), None
        elapsed = []
        with torch.inference_mode():
            for _ in range(16):
                started = time.perf_counter()
                inp = running[:, -1:] if use_cache and past is not None else running
                output = model(inp, past_key_values=past, use_cache=use_cache)
                next_id = output.logits[:, -1].argmax(-1, keepdim=True)
                running = torch.cat([running, next_id], dim=1)
                past = output.past_key_values if use_cache else None
                elapsed.append(time.perf_counter() - started)
        return running[0, ids.shape[1]:].tolist(), elapsed
    generate(True)  # warm-up, omitted from measurements
    generate(False)
    trials = []
    for repeat in range(3):
        trial = {}
        # Alternate measurement order to avoid always giving one mode the first slot.
        for enabled in ((True, False) if repeat % 2 == 0 else (False, True)):
            tokens, seconds = generate(enabled)
            trial["cached" if enabled else "full_prefix"] = {"token_ids": tokens, "step_seconds": seconds,
                "total_seconds": sum(seconds), "first_step_seconds": seconds[0], "mean_later_step_seconds": sum(seconds[1:]) / 15}
        assert trial["cached"]["token_ids"] == trial["full_prefix"]["token_ids"]
        trials.append(trial)
    return {"scope": "Real GPT-2 CPU fp32 eager-attention computation; local forward-step timings exclude tokenization, loading, network, queueing and text rendering; not a serving benchmark.",
            "prompt": prompt, "prompt_token_ids": ids[0].tolist(), "max_abs_logit_difference": max_error,
            "cache_tensor_bytes": actual_bytes, "formula_tensor_bytes": expected_bytes,
            "cache_sequence_length": ids.shape[1] + 1,
            "generation": tokenizer.decode(trials[0]["cached"]["token_ids"]),
            "fixed_generated_tokens": 16, "stop_rule": "fixed length, including any EOS; this is a computation comparison",
            "trials": trials}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["all", *MODELS], default="all")
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--output", type=Path, default=BASE / "real-experiments/results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dataset_path = BASE / "real-experiments/data.json"
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    environment = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
                   "threads": torch.get_num_threads(), "device": "cpu", "dtype": "float32",
                   "packages": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "huggingface-hub", "numpy", "tokenizers", "safetensors"]},
                   "script_sha256": sha(__file__), "data_sha256": sha(dataset_path)}
    write(args.output / "environment.json", environment)
    for kind in MODELS if args.experiment == "all" else [args.experiment]:
        path, provenance = load_path(kind, args.cache_root)
        print("running", kind, provenance["repo"], flush=True)
        started = time.perf_counter()
        if kind == "retrieval":
            report = retrieval(data, path)
        elif kind == "finetune":
            report = finetune(data, path, args.output)
        else:
            report = cache_experiment(data, path)
        write(args.output / (kind + ".json"), {"model": provenance, "environment": environment,
              "wall_seconds_including_load": time.perf_counter() - started, "result": report})
        print("completed", kind, flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
