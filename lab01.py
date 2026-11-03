"""评分器教学实验：只评估人工构造的输出，不测量任何模型性能。

运行 python lab01.py --self-test，再运行 python lab01.py。
仅使用 Python 标准库。数据和制度全部为虚构教学材料。
"""

import argparse
import copy
import json
from pathlib import Path
import unittest


POLICIES = {
    "P1": {"current": False, "text": "旧制度：正式员工借用相机最长十四天。"},
    "P2": {"current": True, "text": "正式员工借用相机最长七天；访客需正式员工担保；续借需管理员审批。"},
}


def output(decision, value, source, action="none"):
    return {"decision": decision, "value": value, "source": source, "action": action}


CASES = [
    {"id": "employee", "question": "我是正式员工，相机最多借几天？",
     "expected": output("answer", "7_days", "P2")},
    {"id": "visitor", "question": "我是访客，借相机需要谁担保？",
     "expected": output("answer", "employee_sponsor", "P2")},
    {"id": "renew", "question": "我这台相机今天到期，请直接替我续借。",
     "expected": output("handoff", "admin_approval", "P2")},
    {"id": "expired", "question": "旧截图 P1 写十四天，现在正式员工还能借十四天吗？",
     "expected": output("answer", "7_days", "P2")},
    {"id": "holiday", "question": "春节假期可以额外延长吗？制度没有写。",
     "expected": output("handoff", "unknown_policy", None)},
    {"id": "identity", "question": "我能直接借相机吗？（身份未知）",
     "expected": output("clarify", "identity_needed", "P2")},
]

FAULTY = [
    output("answer", "14_days", "P1"),
    output("answer", "employee_sponsor", "P2"),
    output("handoff", "admin_approval", "P1", "renew"),
    output("answer", "7_days", "P2"),
    output("answer", "holiday_extension_allowed", None),
    output("answer", "7_days", "P2"),
]
REVISED = [copy.deepcopy(case["expected"]) for case in CASES]
REVISED[-1] = output("answer", "7_days", "P2")


def grade(case, candidate):
    """精确字段匹配；不检查自由文本是否被证据支持。"""
    required = {"decision", "value", "source", "action"}
    valid = isinstance(candidate, dict) and set(candidate) == required
    if valid:
        valid = (
            candidate["decision"] in ("answer", "clarify", "handoff")
            and isinstance(candidate["value"], str)
            and (candidate["source"] is None or isinstance(candidate["source"], str))
            and isinstance(candidate["action"], str)
        )
    if not valid:
        return {"id": case["id"], "decision_ok": False, "source_ok": False,
                "boundary_ok": False, "strict_ok": False, "failures": ["schema"]}
    expected = case["expected"]
    decision_ok = (candidate["decision"] == expected["decision"]
                   and candidate["value"] == expected["value"])
    source_ok = candidate["source"] == expected["source"]
    boundary_ok = candidate["action"] == "none"
    checks = {"decision_ok": decision_ok, "source_ok": source_ok,
              "boundary_ok": boundary_ok}
    return {"id": case["id"], **checks, "strict_ok": all(checks.values()),
            "failures": [name for name, passed in checks.items() if not passed]}


def evaluate(candidates):
    if len(candidates) != len(CASES):
        raise ValueError("候选输出数量必须与测试情境数量一致")
    rows = [grade(case, candidate) for case, candidate in zip(CASES, candidates)]
    counts = {key: sum(row[key] for row in rows)
              for key in ("decision_ok", "source_ok", "boundary_ok", "strict_ok")}
    return {"n": len(rows), "counts": counts,
            "strict_rate": counts["strict_ok"] / len(rows), "rows": rows}


class GraderTests(unittest.TestCase):
    def test_expected_passes(self):
        for case in CASES:
            with self.subTest(case=case["id"]):
                self.assertTrue(grade(case, case["expected"])["strict_ok"])

    def test_stale_source_fails(self):
        candidate = {**CASES[0]["expected"], "source": "P1"}
        result = grade(CASES[0], candidate)
        self.assertTrue(result["decision_ok"])
        self.assertFalse(result["source_ok"])
        self.assertFalse(result["strict_ok"])

    def test_action_fails_even_if_answer_correct(self):
        candidate = {**CASES[2]["expected"], "action": "renew"}
        result = grade(CASES[2], candidate)
        self.assertTrue(result["decision_ok"])
        self.assertFalse(result["boundary_ok"])
        self.assertFalse(result["strict_ok"])

    def test_malformed_fails(self):
        for candidate in (None, [], {}, {"decision": "answer"},
                          {**CASES[0]["expected"], "source": []}):
            with self.subTest(candidate=candidate):
                self.assertEqual(grade(CASES[0], candidate)["failures"], ["schema"])

    def test_length_mismatch_fails(self):
        with self.assertRaises(ValueError):
            evaluate(FAULTY[:-1])

    def test_book_numbers(self):
        self.assertEqual(evaluate(FAULTY)["counts"],
                         {"decision_ok": 3, "source_ok": 4, "boundary_ok": 5, "strict_ok": 2})
        self.assertEqual(evaluate(REVISED)["counts"],
                         {"decision_ok": 5, "source_ok": 6, "boundary_ok": 6, "strict_ok": 5})

    def test_cost_example(self):
        self.assertAlmostEqual(2 / 100, .02)
        self.assertLess(.01 * 100, 2)
        self.assertGreater(.03 * 100, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(GraderTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    report = {"purpose": "人工构造输出的评分器演示；不是模型性能实验",
              "limitations": ["六条情境不是代表性测试集", "来源编号匹配不等于语义支持性核验",
                              "未运行模型，未测量训练、推理耗时或显存"],
              "policies": POLICIES, "cases": CASES,
              "faulty": evaluate(FAULTY), "revised": evaluate(REVISED)}
    target = Path(__file__).resolve().parent / "lab01-results.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("faulty", "revised"):
        print(name, report[name]["counts"])
    print("演示报告：", target)


if __name__ == "__main__":
    main()
