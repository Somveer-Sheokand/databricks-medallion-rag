"""Tiny retrieval eval: one distinctive question per ingested document,
checked against the live vector index via the same code path the MCP server
uses. Not part of the `pytest` suite (which promises "no Databricks needed")
-- this hits the real Databricks workspace, so it's a separate opt-in check.

Catches regressions from changes to chunking, cleaning, or the embedding
model that silently hurt retrieval quality, by asserting each question's
expected document shows up somewhere in the top-k results.

Update EVAL_CASES after re-running scripts/fetch_arxiv_sample.py -- doc_ids
are content-hashed paths, so a different document set needs different
expected doc_ids (see databricks/README.md's "Getting documents in" for how
to look these up: query bronze_raw_docs for doc_id + a text preview).

Usage:
    python scripts/eval_retrieval.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv


@dataclass
class EvalCase:
    question: str
    expected_doc_id: str
    label: str  # short human-readable title, for readable failure output


# One case per document in the default arxiv_sample dataset (17 docs as of
# the last full rebuild). doc_id -> title, from bronze_raw_docs.
EVAL_CASES: List[EvalCase] = [
    EvalCase("How is RAG adapted for regulatory compliance question answering in financial services?",
             "bd660fac2bae2160", "Automated Regulatory Compliance QA in Financial Services"),
    EvalCase("What are low-cost assays for measuring language model behavior across vendors and releases?",
             "31cbb60fb8b72485", "Low-Cost Assays for Measuring Model Behavior"),
    EvalCase("How do surface style cues explain zero-shot code attribution by language models?",
             "b242fab63c27d724", "Style, Not Self: Zero-Shot Code Attribution"),
    EvalCase("What is self-play pretraining with zero data?",
             "f08c2365fc029b76", "Self-Play Pretraining with Zero Data"),
    EvalCase("How reproducible are LLM evaluation conclusions based on inferred prompt structure?",
             "ec9a583965dbfbaa", "How Reproducible Are Evaluation Conclusions?"),
    EvalCase("How does PrivDrift audit user-secret leakage during topic drift in conversations?",
             "6047bc9dbf8f1bfc", "PrivDrift: Auditing User-Secret Leakage"),
    EvalCase("What is the R-DEIM Net model for paraphrase detection?",
             "8452686189b5018d", "R-DEIM Net for Paraphrase Detection"),
    EvalCase("How is audio description generation framed as a constrained global optimization problem?",
             "3b7d5db3ad5d2ec1", "Audio Description as Constrained Global Optimization"),
    EvalCase("How is simulation used to screen customer experience AI agents before production deployment?",
             "5f78e69240509a6a", "Screen Before You Serve: Simulation for CX AI Agents"),
    EvalCase("What does the GRASP framework do for strategic planning with agentic AI?",
             "b868de474af4b5f1", "GRASP: Strategic Planning with Agentic AI"),
    EvalCase("Does a language model's stated reason for rejecting a candidate actually do any work?",
             "18ac60a3012b41ac", "Does a Model's Stated Rejection Reason Do Any Work?"),
    EvalCase("What does ExplorationBench measure about AI systems' exploration ability?",
             "2fe2a987a78cf9b6", "ExplorationBench"),
    EvalCase("How does PoEM predict RL outcomes from existing policies without retraining?",
             "ff59b203ec623f64", "PoEM: Predicting RL Outcomes from Existing Policies"),
    EvalCase("How does retrieval-augmented fact checking work for spoken claims?",
             "11d29cd242c4e75d", "To Trust or Not to Trust: RAG Fact Checking in Speech"),
    EvalCase("How does SemMSA handle multimodal sentiment analysis with incomplete data?",
             "8eb7558ea29cb45c", "SemMSA: Robust Multimodal Sentiment Analysis"),
    EvalCase("How can natural context flip the output of a decision model according to JevOut?",
             "71f18db0c5d74d02", "JevOut: Natural Context Can Flip Decision Models"),
    EvalCase("How do agents detect online conspiratorial discourse?",
             "9771115aa1ed666b", "Agentic Detection of Online Conspiracies"),
]

TOP_K = 5


def main() -> int:
    load_dotenv(Path.cwd() / ".env")  # DATABRICKS_HOST / DATABRICKS_TOKEN, for local runs

    from rag_common.config import load_config
    from mcp_server.tools.retrieve import search_documents

    load_config()  # fails fast with a clear error if config/config.yaml is missing

    passed = 0
    for case in EVAL_CASES:
        hits = search_documents(case.question, top_k=TOP_K)
        doc_ids = [h["doc_id"] for h in hits]
        rank = doc_ids.index(case.expected_doc_id) + 1 if case.expected_doc_id in doc_ids else None
        if rank:
            passed += 1
            print(f"PASS  rank {rank}  {case.label}")
        else:
            print(f"FAIL  not in top {TOP_K}  {case.label}")
            print(f"      question: {case.question}")
            print(f"      got doc_ids: {doc_ids}")

    total = len(EVAL_CASES)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
