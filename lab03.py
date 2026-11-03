"""第三章：用两个参数观察训练。只使用标准库；数据是人工构造的数值点。

运行 --self-test 检验公式；默认执行100次全批量更新并保存完整轨迹。
"""
import argparse
import json
import math
import platform
from pathlib import Path
import unittest

TRAIN = [(-2., 0), (-1., 0), (1., 1), (2., 1)]
DEV = [(-1.5, 0), (-.5, 0), (.5, 1), (1.5, 1)]


def validate(data):
    if not data or any(not math.isfinite(x) or y not in (0, 1) for x, y in data):
        raise ValueError("nonempty finite inputs and binary labels required")


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def example_loss(z, y):
    # Numerically stable binary cross-entropy from a logit; no log(0).
    return max(z, 0) - y*z + math.log1p(math.exp(-abs(z)))


def measure(data, w, b):
    validate(data)
    probabilities = [sigmoid(w*x+b) for x, _ in data]
    loss = sum(example_loss(w*x+b, y) for x, y in data) / len(data)
    accuracy = sum(int(p >= .5) == y for p, (_, y) in zip(probabilities, data)) / len(data)
    return {"loss": loss, "accuracy": accuracy, "probabilities": probabilities}


def gradients(data, w, b):
    validate(data)
    residuals = [(sigmoid(w*x+b)-y, x) for x, y in data]
    return (sum(r*x for r, x in residuals) / len(data),
            sum(r for r, _ in residuals) / len(data))


def finite_difference(data, w, b, eps=1e-6):
    if eps <= 0:
        raise ValueError("eps must be positive")
    gw = (measure(data, w+eps, b)["loss"]-measure(data, w-eps, b)["loss"])/(2*eps)
    gb = (measure(data, w, b+eps)["loss"]-measure(data, w, b-eps)["loss"])/(2*eps)
    return gw, gb


def train(steps=100, lr=.4):
    if type(steps) is not int or steps < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("nonnegative integer steps and positive finite learning rate required")
    w, b = 0., 0.
    history = []
    for step in range(steps+1):
        history.append({"step": step, "w": w, "b": b,
                        "train": measure(TRAIN, w, b), "dev": measure(DEV, w, b)})
        if step < steps:
            gw, gb = gradients(TRAIN, w, b)
            w, b = w-lr*gw, b-lr*gb
    return history


def quadratic_demo(lr=2.2, steps=3):
    w = 1.
    result = []
    for step in range(steps+1):
        result.append({"step": step, "w": w, "loss": .5*w*w})
        w -= lr*w
    return result


def causal_pairs(tokens):
    if len(tokens) < 2:
        raise ValueError("need at least two tokens")
    return tokens[:-1], tokens[1:]


def check_groups(train_ids, eval_ids):
    overlap = set(train_ids) & set(eval_ids)
    if overlap:
        raise ValueError("shared source groups: " + ", ".join(sorted(overlap)))
    return True


class LearningTests(unittest.TestCase):
    def test_initial_prediction_and_loss(self):
        initial = measure(TRAIN, 0., 0.)
        self.assertEqual(initial["probabilities"], [.5]*4)
        self.assertAlmostEqual(initial["loss"], math.log(2))
        self.assertEqual(initial["accuracy"], .5)

    def test_initial_gradient(self):
        self.assertEqual(gradients(TRAIN, 0., 0.), (-.75, 0.))

    def test_first_update(self):
        record = train(1)[1]
        self.assertAlmostEqual(record["w"], .3)
        self.assertAlmostEqual(record["b"], 0.)
        self.assertLess(record["train"]["loss"], math.log(2))

    def test_gradient_against_finite_difference(self):
        for analytic, numerical in zip(gradients(TRAIN, .2, -.1), finite_difference(TRAIN, .2, -.1)):
            self.assertAlmostEqual(analytic, numerical, places=8)

    def test_unequal_batch_weighting(self):
        whole = gradients(TRAIN, .2, -.1)
        first = gradients(TRAIN[:1], .2, -.1)
        rest = gradients(TRAIN[1:], .2, -.1)
        for j in (0, 1):
            self.assertAlmostEqual(whole[j], (first[j]+3*rest[j])/4)

    def test_loss_decreases_for_fixed_demo(self):
        history = train()
        self.assertTrue(all(a["train"]["loss"] > b["train"]["loss"]
                            for a, b in zip(history, history[1:])))
        self.assertEqual(history[-1]["train"]["accuracy"], 1.)
        self.assertLess(history[-1]["train"]["loss"], .04)

    def test_deterministic_without_randomness(self):
        self.assertEqual(train(5), train(5))

    def test_extreme_logits_are_finite(self):
        for z in (-1000., 1000.):
            for y in (0, 1):
                self.assertTrue(math.isfinite(example_loss(z, y)))
                self.assertGreaterEqual(example_loss(z, y), 0)

    def test_quadratic_overshoot(self):
        history = quadratic_demo()
        self.assertAlmostEqual(history[1]["w"], -1.2)
        self.assertTrue(all(a["loss"] < b["loss"] for a, b in zip(history, history[1:])))

    def test_causal_shift(self):
        inputs, labels = causal_pairs([2, 3, 4, 5, 6])
        self.assertEqual(inputs, [2, 3, 4, 5])
        self.assertEqual(labels, [3, 4, 5, 6])

    def test_group_overlap(self):
        self.assertTrue(check_groups(["doc_a"], ["doc_b"]))
        with self.assertRaises(ValueError):
            check_groups(["doc_a", "doc_b"], ["doc_a"])

    def test_invalid_inputs(self):
        operations = [lambda: gradients([], 0, 0), lambda: measure([(1, 2)], 0, 0),
                      lambda: measure([(float("nan"), 0)], 0, 0), lambda: train(-1),
                      lambda: train(lr=0), lambda: causal_pairs([1])]
        for operation in operations:
            with self.assertRaises(ValueError):
                operation()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(LearningTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    history = train()
    analytic = gradients(TRAIN, .2, -.1)
    numerical = finite_difference(TRAIN, .2, -.1)
    report = {"purpose": "合成数值点上的两参数逻辑回归；不是大模型训练或应用性能评估",
              "python": platform.python_version(), "platform": platform.platform(),
              "settings": {"steps": 100, "lr": .4, "batch_size": 4, "initial_w": 0, "initial_b": 0},
              "train_data": TRAIN, "dev_data": DEV, "history": history,
              "gradient_check": {"analytic": analytic, "numerical": numerical,
                                 "max_abs_error": max(abs(a-b) for a, b in zip(analytic, numerical))},
              "quadratic_overshoot": quadratic_demo(),
              "causal_pairs": causal_pairs([2, 3, 4, 5, 6]),
              "limitations": ["开发点也是人工构造；没有独立真实测试集",
                              "未学习文本特征或嵌入；未训练语言模型",
                              "该数据无噪声且线性可分；不代表实际优化难度"]}
    target = Path(__file__).resolve().parent / "lab03-results.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for step in (0, 1, 10, 100):
        row = history[step]
        print(step, "w=", row["w"], "train=", row["train"]["loss"],
              "dev=", row["dev"]["loss"], "accuracy=", row["train"]["accuracy"])
    print("gradient check:", report["gradient_check"])
    print("实验报告：", target)


if __name__ == "__main__":
    main()
