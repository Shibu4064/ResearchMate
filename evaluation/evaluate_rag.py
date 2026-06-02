from __future__ import annotations

import json
from pathlib import Path

from src.researchmate.rag_pipeline import RAGPipeline


def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lower = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in lower)
    return hits / len(keywords)


def main() -> None:
    test_file = Path(__file__).parent / "test_questions.json"
    tests = json.loads(test_file.read_text())
    pipeline = RAGPipeline()

    results = []
    for item in tests:
        response = pipeline.ask(item["question"])
        score = keyword_hit_rate(response["answer"], item.get("expected_keywords", []))
        results.append(
            {
                "question": item["question"],
                "keyword_hit_rate": score,
                "answer_preview": response["answer"][:300],
                "n_sources": len(response["sources"]),
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
