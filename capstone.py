"""第15章综合演练：仅使用虚构资料与内存服务，不调用模型或外部系统。"""
import argparse
import copy
import json
from pathlib import Path
import unittest

from labs import DOCUMENTS, MockExecutor, bm25, eligible, digest


class LoanWorkflow:
    """Identity/group here are trusted test inputs, not claims extracted from user text."""
    def __init__(self, documents=None):
        self.documents = copy.deepcopy(DOCUMENTS if documents is None else documents)
        self.service = MockExecutor()
        self.trace = []

    def propose(self, principal, group, asset, days):
        if not principal or group not in {"employee", "visitor"}:
            return {"status": "clarify", "question": "请通过可信身份通道确认身份。"}
        if group == "visitor":
            return {"status": "handoff", "reason": "访客担保与适用借期需要单独核验；本演练只自动处理员工。"}
        if asset != "CAM-017" or type(days) is not int or days < 1:
            return {"status": "invalid_input", "reason": "仅支持 CAM-017 与正整数借期。"}
        visible = [d for d in self.documents if eligible(d, group)]
        ranking = bm25(["相机", "借用"], visible)
        self.trace.append({"stage": "retrieve", "ids": [d[0] for d in ranking]})
        policies = [d for d in visible if "max_days" in d["fact"]]
        if len(policies) != 1:
            return {"status": "handoff", "reason": "缺少唯一的现行可用借期依据。"}
        policy = policies[0]
        maximum = policy["fact"]["max_days"]
        # Match the executor's declared teaching policy; never silently reinterpret a new version.
        if policy["id"] != "P2" or policy["version"] != "v2" or maximum != 7:
            return {"status": "handoff", "reason": "资料与执行器策略版本不兼容。"}
        if days > maximum:
            return {"status": "policy_rejected", "source": policy["id"], "maximum": maximum}
        claim = {"key": "max_days", "value": maximum, "source": policy["id"]}
        supported = policy["fact"].get(claim["key"]) == claim["value"]
        if not supported:
            raise AssertionError("structured evidence mismatch")
        payload = {"action": "request_loan", "asset": asset, "days": days}
        self.trace.append({"stage": "verify", "claim": claim, "status": "supported"})
        return {"status": "awaiting_approval", "principal": principal, "payload": payload,
                "evidence": claim, "policy_hash": digest(policy),
                "message": "现行员工借期不超过七天；尚未提交，等待对本次申请参数的提交授权。"}

    def submit(self, proposal, approval, key, lose_response=False):
        if proposal.get("status") != "awaiting_approval":
            raise ValueError("only a verified proposal may be submitted")
        # Recheck live teaching documents to detect changes between preparation and execution.
        payload = proposal["payload"]
        fresh = self.propose(proposal["principal"], "employee", payload["asset"], payload["days"])
        if fresh.get("status") != "awaiting_approval" or fresh["policy_hash"] != proposal["policy_hash"]:
            raise PermissionError("policy changed after proposal")
        try:
            result = self.service.execute(proposal["principal"], payload, approval, key,
                                          lose_response=lose_response)
        except TimeoutError:
            self.trace.append({"stage": "timeout", "action": "replay_same_key"})
            result = self.service.execute(proposal["principal"], payload, approval, key)
        request_index = int(result["request_id"][1:])-1
        if self.service.requests[request_index] != payload:
            raise AssertionError("stored state differs from approved request")
        self.trace.append({"stage": "confirm", "result": copy.deepcopy(result)})
        return {"status": "submitted", "request_id": result["request_id"],
                "message": "申请已提交；这不代表管理员已经批准或设备已经交付。"}


def run_demo():
    workflow = LoanWorkflow()
    proposal = workflow.propose("user-A", "employee", "CAM-017", 7)
    before = len(workflow.service.requests)
    # Test fixture acts as an external approver. A model must not issue its own approvals.
    approval = workflow.service.approve("user-A", proposal["payload"])
    result = workflow.submit(proposal, approval, "business-request-001", lose_response=True)
    return {"scope": "deterministic offline mock, no model or external submission",
            "before_approval_requests": before, "proposal": proposal, "result": result,
            "stored_requests": workflow.service.requests, "trace": workflow.trace,
            "executor_events": workflow.service.events,
            "unknown_identity": LoanWorkflow().propose("", "unknown", "CAM-017", 7),
            "over_limit": LoanWorkflow().propose("user-A", "employee", "CAM-017", 14),
            "missing_evidence": LoanWorkflow([]).propose("user-A", "employee", "CAM-017", 7)}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.flow = LoanWorkflow()
        self.proposal = self.flow.propose("user-A", "employee", "CAM-017", 7)

    def test_proposal_has_no_side_effect(self):
        self.assertEqual(self.proposal["status"], "awaiting_approval")
        self.assertEqual(self.flow.service.requests, [])
        self.assertNotIn("P1", self.flow.trace[0]["ids"])
        self.assertNotIn("S1", self.flow.trace[0]["ids"])

    def test_missing_approval(self):
        with self.assertRaises(PermissionError):
            self.flow.submit(self.proposal, None, "key")
        self.assertEqual(self.flow.service.requests, [])

    def test_timeout_then_replay(self):
        approval = self.flow.service.approve("user-A", self.proposal["payload"])
        result = self.flow.submit(self.proposal, approval, "key", lose_response=True)
        again = self.flow.submit(self.proposal, approval, "key")
        self.assertEqual(result, again)
        self.assertEqual(len(self.flow.service.requests), 1)

    def test_parameter_changed_after_approval(self):
        approval = self.flow.service.approve("user-A", self.proposal["payload"])
        changed = copy.deepcopy(self.proposal)
        changed["payload"]["days"] = 6
        with self.assertRaises(PermissionError):
            self.flow.submit(changed, approval, "key")
        self.assertEqual(self.flow.service.requests, [])

    def test_other_principal(self):
        approval = self.flow.service.approve("user-B", self.proposal["payload"])
        with self.assertRaises(PermissionError):
            self.flow.submit(self.proposal, approval, "key")

    def test_missing_and_conflicting_evidence(self):
        self.assertEqual(LoanWorkflow([]).propose("A", "employee", "CAM-017", 7)["status"], "handoff")
        docs = copy.deepcopy(DOCUMENTS)
        docs[0]["current"] = True
        self.assertEqual(LoanWorkflow(docs).propose("A", "employee", "CAM-017", 7)["status"], "handoff")

    def test_unknown_and_visitor(self):
        self.assertEqual(self.flow.propose("", "unknown", "CAM-017", 7)["status"], "clarify")
        self.assertEqual(self.flow.propose("B", "visitor", "CAM-017", 7)["status"], "handoff")

    def test_over_limit_and_invalid(self):
        self.assertEqual(self.flow.propose("A", "employee", "CAM-017", 14)["status"], "policy_rejected")
        self.assertEqual(self.flow.propose("A", "employee", "CAM-017", True)["status"], "invalid_input")

    def test_policy_changes_before_submit(self):
        approval = self.flow.service.approve("user-A", self.proposal["payload"])
        self.flow.documents[1]["current"] = False
        with self.assertRaises(PermissionError):
            self.flow.submit(self.proposal, approval, "key")
        self.assertEqual(self.flow.service.requests, [])

    def test_new_policy_requires_compatible_executor(self):
        docs = copy.deepcopy(DOCUMENTS)
        docs[1]["fact"]["max_days"] = 10
        self.assertEqual(LoanWorkflow(docs).propose("A", "employee", "CAM-017", 8)["status"], "handoff")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__], verbosity=2)
    else:
        output = Path(__file__).with_name("capstone-results.json")
        output.write_text(json.dumps(run_demo(), ensure_ascii=False, indent=2), encoding="utf-8")
        print("综合演练完成："+str(output))
