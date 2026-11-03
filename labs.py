"""第4—15章离线实验。标准库实现；不下载模型、不联网、不执行外部动作。

python labs.py --self-test
python labs.py --chapter 4
python labs.py --all
所有数值均来自明确的构造数据，不是商业模型的测评成绩。
"""
import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import unittest

BASE = Path(__file__).resolve().parent


def softmax(values, temperature=1.):
    if not values or temperature <= 0 or not all(math.isfinite(v) for v in values):
        raise ValueError("finite nonempty logits and positive temperature required")
    maximum = max(values)
    exps = [math.exp((v-maximum)/temperature) for v in values]
    return [v/sum(exps) for v in exps]


def attention(q, keys, values, allowed):
    if not keys or len(keys) != len(values) or len(keys) != len(allowed):
        raise ValueError("inconsistent source lengths")
    if not q or any(len(k) != len(q) for k in keys):
        raise ValueError("inconsistent key dimensions")
    width = len(values[0])
    if not width or any(len(v) != width for v in values):
        raise ValueError("inconsistent value dimensions")
    indices = [i for i, valid in enumerate(allowed) if valid]
    if not indices:
        raise ValueError("no visible key")
    scores = [sum(a*b for a, b in zip(q, keys[i]))/math.sqrt(len(q)) for i in indices]
    weights = [0.] * len(keys)
    for i, weight in zip(indices, softmax(scores)):
        weights[i] = weight
    return weights, [sum(weights[i]*values[i][j] for i in range(len(keys))) for j in range(width)]


def wilson(successes, n, z=1.96):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid count")
    p = successes/n
    denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return max(0., center-half), min(1., center+half)


def paired_difference(a, b):
    if not a or len(a) != len(b):
        raise ValueError("paired nonempty results required")
    wins = sum(y > x for x, y in zip(a, b))
    losses = sum(y < x for x, y in zip(a, b))
    return {"difference": (sum(b)-sum(a))/len(a), "wins": wins, "losses": losses,
            "ties": len(a)-wins-losses}


def choose_context(items, budget):
    if budget < 0 or any(i["cost"] < 0 for i in items):
        raise ValueError("negative budget/cost")
    best, score = [], -1
    for flags in itertools.product((False, True), repeat=len(items)):
        selected = [item for item, flag in zip(items, flags) if flag]
        if sum(i["cost"] for i in selected) <= budget:
            value = sum(i["value"] for i in selected)
            if value > score:
                best, score = selected, value
    return {"selected": [i["id"] for i in best], "cost": sum(i["cost"] for i in best), "value": score}


DOCUMENTS = [
    {"id": "P1", "current": False, "groups": ["employee"], "version": "v1",
     "terms": "相机 借用 相机 借用 十四 天".split(), "fact": {"max_days": 14}},
    {"id": "P2", "current": True, "groups": ["employee", "visitor"], "version": "v2",
     "terms": "相机 借用 正式 员工 七 天".split(), "fact": {"max_days": 7}},
    {"id": "P3", "current": True, "groups": ["employee", "visitor"], "version": "v2",
     "terms": "访客 相机 担保 正式 员工".split(), "fact": {"sponsor": "employee"}},
    {"id": "P4", "current": True, "groups": ["employee", "visitor"], "version": "v2",
     "terms": "相机 续借 管理员 审批".split(), "fact": {"renew": "admin_approval"}},
    {"id": "S1", "current": True, "groups": ["admin"], "version": "private-v1",
     "terms": "相机 借用 相机 借用 管理员 特例".split(), "fact": {"max_days": 30}},
]


def eligible(document, group):
    return document["current"] and group in document["groups"]


def bm25(query, documents, k1=1.2, b=.75):
    if not documents:
        return []
    average = sum(len(d["terms"]) for d in documents)/len(documents)
    if average == 0:
        return [(d["id"], 0.) for d in documents]
    results = []
    for document in documents:
        counts = Counter(document["terms"])
        score = 0.
        for term in set(query):
            df = sum(term in d["terms"] for d in documents)
            idf = math.log(1+(len(documents)-df+.5)/(df+.5))
            tf = counts[term]
            score += idf*tf*(k1+1)/(tf+k1*(1-b+b*len(document["terms"])/average))
        results.append((document["id"], score))
    return sorted(results, key=lambda item: (-item[1], item[0]))


def retrieve(query, group):
    # Permissions and version filters are independent of relevance scoring.
    visible = [d for d in DOCUMENTS if eligible(d, group)]
    return bm25(query, visible)


def rrf(rankings, constant=60):
    scores = Counter()
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError("duplicate item in a ranking")
        for rank, item in enumerate(ranking, 1):
            scores[item] += 1/(constant+rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def support(claim, group="employee"):
    document = next((d for d in DOCUMENTS if d["id"] == claim["source"]), None)
    if document is None or not eligible(document, group):
        return "invalid_source"
    key = claim["key"]
    if key not in document["fact"]:
        return "unknown"
    return "supported" if document["fact"][key] == claim["value"] else "contradicted"


CONFIDENCE = [(.95, 1), (.90, 1), (.85, 0), (.80, 1), (.75, 1),
              (.70, 0), (.65, 1), (.60, 0), (.55, 1), (.50, 0)]


def selective(data, threshold):
    if not data:
        raise ValueError("empty evaluation set")
    accepted = [y for p, y in data if p >= threshold]
    return {"threshold": threshold, "accepted": len(accepted), "coverage": len(accepted)/len(data),
            "error_rate": (1-sum(accepted)/len(accepted)) if accepted else None}


def brier(data):
    if not data:
        raise ValueError("empty evaluation set")
    return sum((p-y)**2 for p, y in data)/len(data)


def low_rank_fit(steps=100, zero_both=False):
    a = [0.]*4 if zero_both else [1., 0., 0., 0.]
    b = [0.]*4
    target = [[u*v for v in [1., 0., 0., 0.]] for u in [1., 2., 0., -1.]]
    history = []
    for step in range(steps+1):
        error = [[b[i]*a[j]-target[i][j] for j in range(4)] for i in range(4)]
        loss = sum(v*v for row in error for v in row)/16
        history.append({"step": step, "loss": loss})
        if step < steps:
            ga = [sum(error[i][j]*b[i] for i in range(4))/8 for j in range(4)]
            gb = [sum(error[i][j]*a[j] for j in range(4))/8 for i in range(4)]
            a, b = [v-.8*g for v, g in zip(a, ga)], [v-.8*g for v, g in zip(b, gb)]
    return {"a": a, "b": b, "history": history, "trainable_parameters": 8, "full_parameters": 16}


def quantize(values):
    if not values:
        raise ValueError("empty tensor")
    scale = max(abs(v) for v in values)/7
    if scale == 0:
        return {"scale": 0., "integers": [0]*len(values), "reconstructed": [0.]*len(values), "max_error": 0.}
    q = [max(-7, min(7, round(v/scale))) for v in values]
    reconstructed = [v*scale for v in q]
    return {"scale": scale, "integers": q, "reconstructed": reconstructed,
            "max_error": max(abs(a-b) for a, b in zip(values, reconstructed))}


def dpo(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    margin = (policy_chosen-policy_rejected)-(ref_chosen-ref_rejected)
    z = beta*margin
    loss = max(-z, 0)+math.log1p(math.exp(-abs(z)))
    sigma = 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
    return {"margin": margin, "loss": loss, "d_chosen_logp": beta*(sigma-1),
            "d_rejected_logp": -beta*(sigma-1)}


def kv_bytes(layers, tokens, kv_heads, head_dim, bytes_per_element, batch=1):
    if any(type(v) is not int or v <= 0 for v in (layers, tokens, kv_heads, head_dim, bytes_per_element, batch)):
        raise ValueError("positive integer dimensions required")
    return 2*layers*tokens*kv_heads*head_dim*bytes_per_element*batch


def latency(arrival, token_times):
    if not token_times or token_times[0] < arrival or sorted(token_times) != token_times:
        raise ValueError("ordered nonempty events required")
    return {"ttft": token_times[0]-arrival, "total": token_times[-1]-arrival,
            "tpot": (token_times[-1]-token_times[0])/(len(token_times)-1) if len(token_times)>1 else None}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class MockExecutor:
    """Only an in-memory teaching service. Not a durable transaction/authentication system."""
    def __init__(self):
        self.approvals = {}
        self.results = {}
        self.requests = []
        self.events = []

    def approve(self, principal, payload, expires=200):
        token = "approval-"+str(len(self.approvals)+1)
        self.approvals[token] = {"principal": principal, "payload_hash": digest(payload), "expires": expires}
        return token

    def execute(self, principal, payload, approval, key, now=100, lose_response=False):
        if set(payload) != {"action", "asset", "days"} or payload["action"] != "request_loan":
            raise ValueError("invalid action schema")
        if not isinstance(payload["asset"], str) or not payload["asset"]:
            raise ValueError("asset required")
        if type(payload["days"]) is not int or not 1 <= payload["days"] <= 7:
            raise ValueError("days outside current policy")
        if not principal or not key:
            raise ValueError("principal and idempotency key required")
        fingerprint = digest({"principal": principal, "payload": payload})
        scope = (principal, key)
        if scope in self.results:
            old = self.results[scope]
            if old["fingerprint"] != fingerprint:
                raise ValueError("idempotency key reused with changed parameters")
            self.events.append({"event": "replay", "request_id": old["request_id"]})
            return copy.deepcopy(old)
        permission = self.approvals.get(approval)
        if (permission is None or permission["principal"] != principal
                or permission["payload_hash"] != digest(payload) or now >= permission["expires"]):
            raise PermissionError("approval missing, mismatched, or expired")
        if permission.get("used"):
            raise PermissionError("approval already consumed")
        result = {"request_id": "R"+str(len(self.requests)+1), "status": "submitted",
                  "fingerprint": fingerprint}
        self.requests.append(copy.deepcopy(payload))
        self.results[scope] = result
        permission["used"] = True
        self.events.append({"event": "commit", "request_id": result["request_id"]})
        if lose_response:
            raise TimeoutError("synthetic loss after commit")
        return copy.deepcopy(result)


def transform_box(box, origin, scale, inverse=False):
    if len(box) != 4 or scale <= 0 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("invalid box or scale")
    if inverse:
        return tuple(box[i]/scale+origin[i % 2] for i in range(4))
    return tuple((box[i]-origin[i % 2])*scale for i in range(4))


def iou(a, b):
    for box in (a, b):
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("nonpositive box area")
    intersect = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return intersect/(area_a+area_b-intersect)


def release_gate(baseline, candidate, unauthorized, trace_complete):
    reasons = []
    if unauthorized:
        reasons.append("unauthorized_action")
    if not trace_complete:
        reasons.append("missing_trace")
    if candidate < baseline-.05:
        reasons.append("quality_regression")
    return {"release": not reasons, "reasons": reasons}


def chapter_result(chapter):
    if chapter == 4:
        q, keys, values = [math.sqrt(2), 0], [[math.log(2), 0], [0, 0]], [[1, 0], [0, 2]]
        weights, result = attention(q, keys, values, [True, True])
        masked, masked_result = attention(q, keys, values, [True, False])
        return {"weights": weights, "output": result, "masked_weights": masked, "masked_output": masked_result,
                "temperatures": {str(t): softmax([2, 1, 0], t) for t in (.5, 1, 2)}}
    if chapter == 5:
        a = [1]*12+[0]*8
        b = [1]*10+[0]*2+[1]*6+[0]*2
        return {"n": 20, "a_success": sum(a), "b_success": sum(b),
                "wilson_b": wilson(sum(b), len(b)), "paired": paired_difference(a, b)}
    if chapter == 6:
        items = [{"id": "current_policy", "cost": 6, "value": 9},
                 {"id": "identity", "cost": 4, "value": 6}, {"id": "extra", "cost": 5, "value": 5}]
        return {"unit": "synthetic budget units, not a real tokenizer", "items": items,
                "chosen": choose_context(items, 10), "total_budget": 16, "reserved": 6}
    if chapter == 7:
        return {"unfiltered": bm25(["相机", "借用"], DOCUMENTS),
                "filtered_employee": retrieve(["相机", "借用"], "employee"),
                "filtered_unknown": retrieve(["相机", "借用"], "unknown"),
                "rrf": rrf([["P2", "P3", "P4"], ["P3", "P2", "P4"]])}
    if chapter == 8:
        claims = [{"key": "max_days", "value": 14, "source": "P2"},
                  {"key": "sponsor", "value": "employee", "source": "P3"},
                  {"key": "renew", "value": "admin_approval", "source": "P4"}]
        statuses = [support(c) for c in claims]
        return {"claims": claims, "statuses": statuses, "source_valid_fraction": 1.,
                "supported_fraction": statuses.count("supported")/len(statuses),
                "scope": "known structured facts only, not natural-language entailment"}
    if chapter == 9:
        return {"data": CONFIDENCE, "brier": brier(CONFIDENCE),
                "curve": [selective(CONFIDENCE, t) for t in (.5, .7, .8, .9, 1.)],
                "zero_errors_in_two_interval": wilson(0, 2)}
    if chapter == 10:
        return {"fit": low_rank_fit(), "zero_both": low_rank_fit(zero_both=True),
                "quantization": quantize([-.9, -.2, .1, .7, 1.]),
                "scope": "rank-one matrix fit and symmetric quantization; not a full LoRA/QLoRA model run"}
    if chapter == 11:
        return {"dpo": dpo(-1., -2., -1.2, -1.8),
                "reversed": dpo(-2., -1., -1.2, -1.8),
                "reward_hack": {"text": "done", "string_reward": 1, "state_verified_reward": 0},
                "scope": "objective arithmetic, not preference training or reinforcement learning"}
    if chapter == 12:
        return {"gqa_kv_bytes": kv_bytes(32, 4096, 8, 128, 2),
                "mha_kv_bytes": kv_bytes(32, 4096, 32, 128, 2),
                "uncached_prefix_work_units": 50*(1000+20), "cached_prefix_work_units": 1000+50*20,
                "synthetic_timing": latency(0, [.2, .25, .3]),
                "scope": "storage accounting and synthetic event replay, not GPU benchmarks"}
    if chapter == 13:
        service = MockExecutor()
        payload = {"action": "request_loan", "asset": "CAM-017", "days": 7}
        token = service.approve("user-A", payload)
        try:
            service.execute("user-A", payload, token, "K1", lose_response=True)
        except TimeoutError:
            pass
        replay = service.execute("user-A", payload, token, "K1")
        return {"committed_requests": len(service.requests), "replay": replay, "events": service.events,
                "scope": "in-memory mock, no external account or real request"}
    if chapter == 14:
        original = (100, 200, 300, 400)
        transformed = transform_box(original, (50, 100), 2)
        return {"original": original, "crop_origin": [50, 100], "scale": 2,
                "transformed": transformed, "restored": transform_box(transformed, (50, 100), 2, True),
                "iou_with_shifted": iou(original, (200, 200, 400, 400)),
                "scope": "coordinate geometry only; no OCR or vision model evaluated"}
    if chapter == 15:
        manifest = {"model": "toy-v1", "tokenizer": "toy-v1", "index": "P2-P4-v2",
                    "policy": "loan-v2", "executor": "mock-v1", "eval": "synthetic-v1"}
        changed = {**manifest, "index": "P1-stale"}
        return {"manifest": manifest, "manifest_hash": digest(manifest), "changed_hash": digest(changed),
                "normal": release_gate(.8, .82, 0, True),
                "unsafe": release_gate(.8, .95, 1, True), "regression": release_gate(.8, .7, 0, True),
                "rollback_manifest": manifest, "scope": "local release-gate simulation, not a deployed service"}
    raise ValueError("chapter must be between 4 and 15")


class BookLabsTests(unittest.TestCase):
    def test_04_weight_sum(self):
        r = chapter_result(4)
        self.assertAlmostEqual(sum(r["weights"]), 1.)
        self.assertAlmostEqual(r["weights"][0], 2/3)
        self.assertAlmostEqual(r["output"][1], 2/3)

    def test_04_causal_mask(self):
        r = chapter_result(4)
        self.assertEqual(r["masked_weights"], [1., 0.])
        self.assertEqual(r["masked_output"], [1., 0.])

    def test_04_invalid_attention(self):
        with self.assertRaises(ValueError):
            attention([1], [[1]], [[1]], [False])
        with self.assertRaises(ValueError):
            softmax([1], 0)

    def test_04_softmax_shift(self):
        for a, b in zip(softmax([1, 2]), softmax([1001, 1002])):
            self.assertAlmostEqual(a, b)

    def test_05_interval(self):
        low, high = wilson(16, 20)
        self.assertLess(low, .8)
        self.assertGreater(high, .8)
        self.assertAlmostEqual(low, .583978, places=5)

    def test_05_paired(self):
        result = chapter_result(5)["paired"]
        self.assertEqual((result["wins"], result["losses"], result["ties"]), (6, 2, 12))
        self.assertAlmostEqual(result["difference"], .2)

    def test_05_zero_denominator(self):
        with self.assertRaises(ValueError):
            wilson(0, 0)

    def test_06_budget(self):
        r = chapter_result(6)
        self.assertEqual(r["chosen"]["selected"], ["current_policy", "identity"])
        self.assertEqual(r["chosen"]["cost"], 10)

    def test_06_no_room(self):
        self.assertEqual(choose_context([{"id": "x", "cost": 1, "value": 3}], 0)["selected"], [])

    def test_07_filter(self):
        ids = [i for i, _ in retrieve(["相机", "借用"], "employee")]
        self.assertNotIn("P1", ids)
        self.assertNotIn("S1", ids)
        self.assertEqual(ids[0], "P2")

    def test_07_unknown_identity(self):
        self.assertEqual(retrieve(["相机"], "unknown"), [])

    def test_07_rrf(self):
        result = rrf([["a", "b"], ["b", "a"]])
        self.assertAlmostEqual(result[0][1], result[1][1])
        with self.assertRaises(ValueError):
            rrf([["a", "a"]])

    def test_08_wrong_value(self):
        self.assertEqual(support({"source": "P2", "key": "max_days", "value": 14}), "contradicted")

    def test_08_unknown_fact(self):
        self.assertEqual(support({"source": "P2", "key": "holiday", "value": True}), "unknown")

    def test_08_stale_and_private(self):
        for source in ("P1", "S1", "missing"):
            self.assertEqual(support({"source": source, "key": "max_days", "value": 7}), "invalid_source")

    def test_09_coverage_and_risk(self):
        r = selective(CONFIDENCE, .8)
        self.assertEqual(r["coverage"], .4)
        self.assertEqual(r["error_rate"], .25)

    def test_09_no_acceptance(self):
        self.assertIsNone(selective(CONFIDENCE, 1.)["error_rate"])

    def test_09_brier_perfect(self):
        self.assertEqual(brier([(0, 0), (1, 1)]), 0)

    def test_10_fit(self):
        r = low_rank_fit()
        self.assertLess(r["history"][-1]["loss"], 1e-8)

    def test_10_zero_initialization(self):
        r = low_rank_fit(zero_both=True)
        self.assertEqual(r["history"][0]["loss"], r["history"][-1]["loss"])

    def test_10_quantization_bound(self):
        r = quantize([-.9, -.2, .1, .7, 1.])
        self.assertLessEqual(r["max_error"], r["scale"]/2+1e-12)
        self.assertEqual(quantize([0, 0])["max_error"], 0)

    def test_11_preference_direction(self):
        r = chapter_result(11)
        self.assertLess(r["dpo"]["loss"], r["reversed"]["loss"])
        self.assertLess(r["dpo"]["d_chosen_logp"], 0)

    def test_11_reference_equal(self):
        self.assertAlmostEqual(dpo(-1, -2, -1, -2)["loss"], math.log(2))

    def test_12_memory(self):
        r = chapter_result(12)
        self.assertEqual(r["gqa_kv_bytes"], 512*1024**2)
        self.assertEqual(r["mha_kv_bytes"], 4*r["gqa_kv_bytes"])

    def test_12_timing(self):
        r = latency(0, [.2, .25, .3])
        self.assertAlmostEqual(r["tpot"], .05)
        self.assertIsNone(latency(0, [.2])["tpot"])

    def test_13_commit_then_timeout(self):
        self.assertEqual(chapter_result(13)["committed_requests"], 1)

    def test_13_missing_approval(self):
        with self.assertRaises(PermissionError):
            MockExecutor().execute("A", {"action": "request_loan", "asset": "C", "days": 7}, "none", "k")

    def test_13_bound_user_and_payload(self):
        service = MockExecutor()
        payload = {"action": "request_loan", "asset": "C", "days": 7}
        approval = service.approve("A", payload)
        with self.assertRaises(PermissionError):
            service.execute("B", payload, approval, "k")
        with self.assertRaises(PermissionError):
            service.execute("A", {**payload, "asset": "D"}, approval, "k")

    def test_13_expiry(self):
        service = MockExecutor()
        payload = {"action": "request_loan", "asset": "C", "days": 7}
        token = service.approve("A", payload, expires=100)
        with self.assertRaises(PermissionError):
            service.execute("A", payload, token, "k", now=100)

    def test_13_replay_conflict_and_reuse(self):
        service = MockExecutor()
        payload = {"action": "request_loan", "asset": "C", "days": 7}
        token = service.approve("A", payload)
        service.execute("A", payload, token, "k")
        with self.assertRaises(ValueError):
            service.execute("A", {**payload, "days": 6}, token, "k")
        with self.assertRaises(PermissionError):
            service.execute("A", payload, token, "another-key")

    def test_14_transform_roundtrip(self):
        r = chapter_result(14)
        self.assertEqual(r["original"], r["restored"])
        self.assertEqual(r["transformed"], (100, 200, 500, 600))

    def test_14_iou(self):
        self.assertAlmostEqual(chapter_result(14)["iou_with_shifted"], 1/3)
        self.assertEqual(iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.)

    def test_15_gate(self):
        self.assertTrue(release_gate(.8, .82, 0, True)["release"])
        self.assertFalse(release_gate(.8, .95, 1, True)["release"])
        self.assertFalse(release_gate(.8, .95, 0, False)["release"])

    def test_15_manifest_hash(self):
        r = chapter_result(15)
        self.assertNotEqual(r["manifest_hash"], r["changed_hash"])
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", type=int, choices=range(4, 16))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(BookLabsTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    chapters = list(range(4, 16)) if args.all else [args.chapter] if args.chapter else []
    if not chapters:
        parser.error("choose --chapter, --all, or --self-test")
    for chapter in chapters:
        result = {"chapter": chapter, "python": platform.python_version(), "platform": platform.platform(),
                  "type": "offline educational experiment", "results": chapter_result(chapter)}
        path = BASE / f"lab{chapter:02}-results.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Chapter {chapter}: {path.name}")


if __name__ == "__main__":
    main()
