from typing import List, Dict, Any, Optional

class RRFFormatter:
    """
    Reciprocal Rank Fusion (RRF) and Hybrid Fact Reranker.
    Combines dense vector retrieval scores with Knowledge Graph spreading activation weights.
    """
    def __init__(self, k_constant: int = 60):
        self.k_constant = k_constant

    def rerank_facts(
        self,
        facts: List[Dict[str, Any]],
        vector_candidates: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        # If vector candidates provided, merge using RRF
        merged_scores: Dict[str, float] = {}
        fact_map: Dict[str, Dict[str, Any]] = {}

        # 1. Score graph activation facts
        for rank, fact in enumerate(facts):
            fact_key = f"{fact['source']}_{fact['relation']}_{fact['target']}"
            fact_map[fact_key] = fact
            rrf_score = 1.0 / (self.k_constant + rank + 1)
            merged_scores[fact_key] = merged_scores.get(fact_key, 0.0) + rrf_score

        # 2. Score vector candidates if present
        if vector_candidates:
            for rank, item in enumerate(vector_candidates):
                fact_key = item.get("key", str(item))
                if fact_key in fact_map:
                    rrf_score = 1.0 / (self.k_constant + rank + 1)
                    merged_scores[fact_key] += rrf_score

        sorted_keys = sorted(merged_scores.keys(), key=lambda k: merged_scores[k], reverse=True)
        res = []
        for k in sorted_keys:
            f = fact_map[k]
            f["rrf_score"] = round(merged_scores[k], 4)
            res.append(f)
        return res if res else sorted(facts, key=lambda x: x.get('activation_score', 0.0), reverse=True)

    def format_prompt(self, facts: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
        sorted_facts = self.rerank_facts(facts)
        lines = ["[MEMORY CONTEXT - MEDHAS RECALLED FACTS]"]
        char_count = len(lines[0])
        max_chars = max_tokens * 4  # Approximation: 1 token ~ 4 chars

        seen_reasons = set()
        for fact in sorted_facts:
            reason = fact.get('reason', '')
            if reason and reason in seen_reasons:
                continue
            if reason:
                seen_reasons.add(reason)

            score = fact.get('rrf_score', fact.get('activation_score', 1.0))
            line = f"• ({fact['source']}) --[{fact['relation']}]--> ({fact['target']}) | Reason: {reason} [Score: {score}]"
            if char_count + len(line) + 1 > max_chars:
                break
            lines.append(line)
            char_count += len(line) + 1

        if len(lines) == 1:
            lines.append("• No relevant facts retrieved.")

        return "\n".join(lines)

