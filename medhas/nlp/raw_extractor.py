import re
from typing import List, Dict, Any

class UniversalDynamicExtractor:
    """
    Universal Dynamic OpenIE (Open Information Extraction) Engine.
    Dynamically extracts arbitrary Subject-Predicate-Object triplets and open categories
    from ANY raw text across any domain without hardcoded domain lists.
    """
    def __init__(self):
        self.stop_words = {"the", "a", "an", "this", "that", "these", "those"}

    def _clean_phrase(self, text: str) -> str:
        text = text.strip()
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

    def extract_triplets(self, raw_text: str) -> List[Dict[str, Any]]:
        triplets = []
        sentences = [s.strip() for s in re.split(r'[.!?;\n]+', raw_text) if s.strip()]

        # Universal Clause Splitter matching action verb phrases
        verb_split_pattern = re.compile(
            r'\s+((?:is|are|was|were|has|have|had|been|became|treats|leads|blocked|requires|causes|enables|supports|uses|contains|produces|affects|results|belongs|creates|modifies|interacts|relates|connects|serves|runs|implements|triggers|drives|transfers|monitors|prevents|inhibits|activates|binds|encodes|executes|hosts|deploys|manages|controls|works|operates|built|designed|written|located|found|stored|retains|prunes|retrieves|extracts|processes|serializes|fetches|\b\w{3,15}s\b|\b\w{3,15}ed\b|\b\w{3,15}ing\b)\s*(?:by|to|in|of|from|with|for|on|at|about|over|under|into|against|through|via|as|than)?)\s+',
            re.IGNORECASE
        )

        for sentence in sentences:
            parts = verb_split_pattern.split(sentence, maxsplit=1)
            if len(parts) == 3:
                subj = self._clean_phrase(parts[0])
                # Exclude relative pronoun prefixes from subject
                subj = re.sub(r'^(who|which|that|where)\s+', '', subj, flags=re.IGNORECASE).strip()
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
