"""第二章：分词、查表与表示的信息损失。Python 标准库，无模型下载。

词表和向量均为教学构造。分词是最长匹配，不是 BPE/WordPiece 的实现。
"""
import argparse
import json
import math
import platform
from pathlib import Path
import unittest

TOKENS = ["[PAD]", "[UNK]", "我", "借", "相机", "七", "天", "不", "能", "员工", "。", "摄像机"]
VOCAB = {token: i for i, token in enumerate(TOKENS)}
MATCHABLE = sorted(TOKENS[2:], key=lambda token: (-len(token), token))
EMBEDDINGS = [
    [0., 0., 0.], [0., 0., 0.], [0., 0., 1.], [0., 1., 0.],
    [1., 0., 0.], [0., 0., .7], [0., 0., .5], [-1., 0., 0.],
    [1., 0., 0.], [.1, .1, 1.], [0., 0., 0.], [.8, .6, 0.],
]


def encode(text):
    """保持空格和标点的输入位置；未知码点逐个映射为 UNK，报告偏移。"""
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    records = []
    cursor = 0
    while cursor < len(text):
        token = next((t for t in MATCHABLE if text.startswith(t, cursor)), None)
        stop = cursor + (len(token) if token is not None else 1)
        records.append({"token": token if token is not None else "[UNK]",
                        "id": VOCAB[token] if token is not None else VOCAB["[UNK]"],
                        "start": cursor, "end": stop, "surface": text[cursor:stop]})
        cursor = stop
    return records


def lookup(ids):
    if any(type(i) is not int or i < 0 or i >= len(EMBEDDINGS) for i in ids):
        raise ValueError("token id outside vocabulary")
    return [EMBEDDINGS[i][:] for i in ids]


def batch(texts):
    if not texts:
        raise ValueError("empty batch")
    encoded = [encode(text) for text in texts]
    width = max(len(row) for row in encoded)
    if not width:
        raise ValueError("all sequences are empty")
    ids = [[record["id"] for record in row] + [0] * (width - len(row)) for row in encoded]
    mask = [[1] * len(row) + [0] * (width - len(row)) for row in encoded]
    return {"ids": ids, "mask": mask, "vectors": [lookup(row) for row in ids],
            "id_shape": [len(texts), width], "embedding_shape": [len(texts), width, 3]}


def masked_mean(vectors, mask):
    if not vectors or len(vectors) != len(mask):
        raise ValueError("vectors/mask length mismatch or empty input")
    if any(m not in (0, 1) for m in mask) or not sum(mask):
        raise ValueError("mask must contain binary values and at least one valid item")
    width = len(vectors[0])
    if not width or any(len(row) != width for row in vectors):
        raise ValueError("inconsistent vector dimensions")
    return [sum(row[j] * m for row, m in zip(vectors, mask)) / sum(mask)
            for j in range(width)]


def cosine(a, b):
    if not a or len(a) != len(b):
        raise ValueError("cosine requires equal nonempty dimensions")
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if not norm_a or not norm_b:
        raise ValueError("cosine is undefined for zero vectors")
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def sentence_mean(text):
    ids = [record["id"] for record in encode(text)]
    return masked_mean(lookup(ids), [1] * len(ids))


class RepresentationTests(unittest.TestCase):
    def test_unicode_lengths(self):
        self.assertEqual(len("相机"), 2)
        self.assertEqual(len("相机".encode("utf-8")), 6)
        self.assertEqual(len(encode("相机")), 1)

    def test_longest_match(self):
        self.assertEqual([r["id"] for r in encode("我借相机")], [2, 3, 4])
        self.assertEqual([r["token"] for r in encode("摄像机")], ["摄像机"])

    def test_offsets(self):
        text = "我借相机七天。"
        records = encode(text)
        self.assertEqual("".join(text[r["start"]:r["end"]] for r in records), text)
        self.assertEqual(records[2]["start"], 2)
        self.assertEqual(records[2]["end"], 4)

    def test_unknown_and_whitespace(self):
        self.assertEqual([r["id"] for r in encode("仪器")], [1, 1])
        self.assertEqual([r["id"] for r in encode(" ")], [1])
        self.assertEqual(encode(""), [])

    def test_batch_and_padding(self):
        result = batch(["我借相机", "员工借相机七天"])
        self.assertEqual(result["ids"][0], [2, 3, 4, 0, 0])
        self.assertEqual(result["mask"][0], [1, 1, 1, 0, 0])
        self.assertEqual(result["embedding_shape"], [2, 5, 3])
        self.assertEqual(masked_mean(result["vectors"][0], result["mask"][0]), [1/3]*3)

    def test_cosine(self):
        self.assertAlmostEqual(cosine(EMBEDDINGS[4], EMBEDDINGS[11]), .8)
        self.assertAlmostEqual(cosine([1, 0], [2, 0]), 1.)
        self.assertAlmostEqual(cosine([1, 0], [-1, 0]), -1.)

    def test_order_loss(self):
        self.assertEqual(sentence_mean("我借相机"), sentence_mean("相机借我"))
        self.assertNotEqual([r["id"] for r in encode("我借相机")],
                            [r["id"] for r in encode("相机借我")])

    def test_negation_counterexample(self):
        similarity = cosine(sentence_mean("能借相机"), sentence_mean("不能借相机"))
        self.assertAlmostEqual(similarity, 3/math.sqrt(10))
        self.assertGreater(similarity, .94)

    def test_renumbering_invariance(self):
        ids = [2, 3, 4]
        permutation = list(reversed(range(len(EMBEDDINGS))))
        moved_table = [None] * len(EMBEDDINGS)
        for old_id, new_id in enumerate(permutation):
            moved_table[new_id] = EMBEDDINGS[old_id][:]
        self.assertEqual(lookup(ids), [moved_table[permutation[i]] for i in ids])
        self.assertNotEqual(lookup(ids), [EMBEDDINGS[permutation[i]] for i in ids])

    def test_invalid_inputs(self):
        operations = [lambda: lookup([-1]), lambda: lookup([12]),
                      lambda: cosine([0, 0], [1, 0]), lambda: cosine([1], [1, 2]),
                      lambda: batch([]), lambda: batch([""]),
                      lambda: masked_mean([[1, 2]], [0]),
                      lambda: masked_mean([[1, 2]], [1, 1])]
        for operation in operations:
            with self.assertRaises(ValueError):
                operation()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(RepresentationTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    padded = batch(["我借相机", "员工借相机七天"])
    report = {
        "purpose": "构造词表与向量的表示机制实验；不是预训练模型性能测量",
        "python": platform.python_version(), "platform": platform.platform(),
        "unicode": {"text": "相机", "code_points": len("相机"),
                    "utf8_bytes": len("相机".encode("utf-8")), "toy_tokens": len(encode("相机"))},
        "tokens": encode("我借相机"), "vocab": VOCAB, "embeddings": EMBEDDINGS,
        "padded_batch": padded,
        "masked_mean": masked_mean(padded["vectors"][0], padded["mask"][0]),
        "unmasked_mean": masked_mean(padded["vectors"][0], [1]*5),
        "camera_cosine": cosine(EMBEDDINGS[4], EMBEDDINGS[11]),
        "order_counterexample": {"a": sentence_mean("我借相机"), "b": sentence_mean("相机借我")},
        "negation_cosine": cosine(sentence_mean("能借相机"), sentence_mean("不能借相机")),
        "limits": ["没有训练模型", "没有真实语义质量测量", "未知码点不可从UNK编号恢复",
                   "本脚本不实现BPE、注意力、位置编码或真正的句向量模型"],
    }
    target = Path(__file__).resolve().parent / "lab02-results.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("unicode", "masked_mean", "unmasked_mean",
                                                  "camera_cosine", "negation_cosine")},
                     ensure_ascii=False, indent=2))
    print("实验报告：", target)


if __name__ == "__main__":
    main()
