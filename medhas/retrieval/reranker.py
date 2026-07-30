from typing import List, Dict, Any

class RRFFormatter:
    def __init__(self, k_constant: int = 60):
        self.k_constant = k_constant

    def rerank_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(facts, key=lambda x: x.get('activation_score', 0.0), reverse=True)

    def format_prompt(self, facts: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
        sorted_facts = self.rerank_facts(facts)
        lines = ["[MEMORY CONTEXT - MEDHAS RECALLED FACTS]"]
        char_count = len(lines[0])
        max_chars = max_tokens * 4  # Approximation: 1 token ~ 4 chars

        seen_reasons = set()
        for fact in sorted_facts:
            reason = fact['reason']
            if reason in seen_reasons:
                continue
            seen_reasons.add(reason)

            line = f"• ({fact['source']}) --[{fact['relation']}]--> ({fact['target']}) | Reason: {reason} [Score: {fact.get('activation_score', 1.0)}]"
            if char_count + len(line) + 1 > max_chars:
                break
            lines.append(line)
            char_count += len(line) + 1

        return "\n".join(lines)
