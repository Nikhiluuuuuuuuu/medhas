import re
import json
from typing import List, Dict, Any, Optional, Callable

EXTRACTION_PROMPT = """Extract all entities and canonical relationship triples (Subject, Predicate, Object) from the text.
Rules:
1. Resolve all pronouns (she, he, they, his, her, su, él, ella) to full named entities.
2. Output MUST be a JSON array of objects with keys: "source", "relation", "target", "category_src", "category_tgt".
3. Keep predicates standardized, uppercase, and concise (e.g. LIVES_IN, OWNS, INHERITED_FROM, HAS_ALLERGY, MARRIED_TO, TRADED_WITH, PROFESSION_IS, LOCATION_IS).
4. Works in any language—preserve proper names while standardizing relationships.

Text:
"{text}"
"""

class UniversalDynamicExtractor:
    """
    Universal Dynamic OpenIE Engine.
    Dynamically extracts arbitrary Subject-Predicate-Object triplets and open categories
    from ANY raw text across any language using LLM structured extraction or robust fallback parsing.
    """
    def __init__(self, default_llm_fn: Optional[Callable[[str], str]] = None):
        self.stop_words = {"the", "a", "an", "this", "that", "these", "those", "el", "la", "los", "las", "un", "una"}
        self.default_llm_fn = default_llm_fn

    def _clean_phrase(self, text: str) -> str:
        text = text.strip()
        # Clean relative pronouns prefix
        text = re.sub(r'^(who|which|that|where|que|quien|donde)\s+', '', text, flags=re.IGNORECASE).strip()
        words = text.split()
        if words and words[0].lower() in self.stop_words:
            words = words[1:]
        return " ".join(words).strip()

    def _dynamic_category(self, phrase: str) -> str:
        cleaned = self._clean_phrase(phrase)
        if not cleaned:
            return "Concept"
        if any(c.isupper() for c in cleaned):
            return "NamedEntity"
        if any(c.isdigit() for c in cleaned):
            return "Metric"
        return "Concept"

    def extract_triplets(self, raw_text: str, custom_llm_fn: Optional[Callable[[str], str]] = None) -> List[Dict[str, Any]]:
        llm = custom_llm_fn or self.default_llm_fn
        if llm is not None:
            try:
                prompt = EXTRACTION_PROMPT.format(text=raw_text)
                resp = llm(prompt)
                match = re.search(r'\[.*\]', resp, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list):
                        triplets = []
                        for item in parsed:
                            if isinstance(item, dict) and "source" in item and "relation" in item and "target" in item:
                                triplets.append({
                                    "source": self._clean_phrase(str(item["source"])),
                                    "relation": str(item["relation"]).strip().upper().replace(" ", "_"),
                                    "target": self._clean_phrase(str(item["target"])),
                                    "reason": raw_text,
                                    "category_src": item.get("category_src", self._dynamic_category(str(item["source"]))),
                                    "category_tgt": item.get("category_tgt", self._dynamic_category(str(item["target"])))
                                })
                        if triplets:
                            return triplets
            except Exception:
                pass

        # Fallback dynamic Open-IE parsing
        return self._fallback_extract(raw_text)

    def _fallback_extract(self, raw_text: str) -> List[Dict[str, Any]]:
        triplets = []
        sentences = [s.strip() for s in re.split(r'[.!?;\n]+', raw_text) if s.strip()]

        verb_split_pattern = re.compile(
            r'\s+((?:is|are|was|were|has|have|had|been|became|treats|leads|blocked|requires|causes|enables|supports|uses|contains|produces|affects|results|belongs|creates|modifies|interacts|relates|connects|serves|runs|implements|triggers|drives|transfers|monitors|prevents|inhibits|activates|binds|encodes|executes|hosts|deploys|manages|controls|works|operates|built|designed|written|located|found|stored|retains|prunes|retrieves|extracts|processes|serializes|fetches|married|traded|owns|bought|lives|es|esta|sont|est)\s*(?:by|to|in|of|from|with|for|on|at|about|over|under|into|against|through|via|as|than|con|en|de)?)\s+',
            re.IGNORECASE
        )

        for sentence in sentences:
            parts = verb_split_pattern.split(sentence, maxsplit=1)
            if len(parts) == 3:
                subj = self._clean_phrase(parts[0])
                pred = parts[1].strip().upper().replace(' ', '_')
                obj = self._clean_phrase(parts[2])
                if subj and obj and len(subj) < 60 and len(obj) < 120 and not subj.lower().startswith("who married"):
                    triplets.append({
                        "source": subj,
                        "relation": pred,
                        "target": obj,
                        "reason": sentence,
                        "category_src": self._dynamic_category(subj),
                        "category_tgt": self._dynamic_category(obj)
                    })

        return triplets

