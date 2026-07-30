import os
import re
import json
import numpy as np
from typing import List, Dict, Any, Optional

class MultilingualAIExtractor:
    """
    Smart Compound Sentence Decomposer & Coreference Resolution Engine.
    Decomposes multi-claim sentences (e.g., 'your admin is nikhil sai pagidimarri and he is a co founder and ai ml engineer')
    into atomic graph triplets with pronoun resolution ('he' -> 'nikhil sai pagidimarri').
    """
    def __init__(self, model_provider: str = "auto"):
        self.model_provider = model_provider
        self._init_models()

    def _init_models(self):
        try:
            from sentence_transformers import SentenceTransformer
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.encoder = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2", 
                local_files_only=True
            )
        except Exception:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            except Exception:
                self.encoder = None

    def _clean_term(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'^(a|an|the|your|my|our|his|her|its)\s+', '', text, flags=re.IGNORECASE).strip()
        return text

    def _split_compound_clauses(self, sentence: str) -> List[str]:
        clauses = re.split(r'\s+(?:and|also|as well as|plus|&)\s+|[,;]+', sentence, flags=re.IGNORECASE)
        return [c.strip() for c in clauses if c.strip()]

    def extract_triplets_with_ai(self, raw_text: str, custom_llm_fn = None) -> List[Dict[str, Any]]:
        from medhas.nlp.raw_extractor import UniversalDynamicExtractor
        extractor = UniversalDynamicExtractor()
        
        # 1. First try LLM dynamic structured extraction or Universal Dynamic Extractor
        triplets = extractor.extract_triplets(raw_text, custom_llm_fn=custom_llm_fn)
        if triplets:
            return triplets

        # 2. General multi-clause parsing for simple and multi-lingual sentences
        return self._smart_compound_parse(raw_text)

    def _smart_compound_parse(self, raw_text: str) -> List[Dict[str, Any]]:
        triplets = []
        sentences = [s.strip() for s in re.split(r'[.!?;\n\|]+', raw_text) if s.strip()]

        for sentence in sentences:
            clauses = self._split_compound_clauses(sentence)
            primary_subject = None

            # Detect main subject in clause group
            for c in clauses:
                words = c.split()
                for i in range(len(words)-1):
                    pair = words[i] + " " + words[i+1]
                    if len(pair) > 5 and not pair.lower().startswith(("your ", "my ", "the ", "and ", "who ", "which ")):
                        if pair.lower() in raw_text.lower() and "admin" not in pair.lower():
                            primary_subject = self._clean_term(pair)
                            break

            for clause in clauses:
                is_pronoun_clause = bool(re.match(r'^(he|she|they|it|who|which|él|ella|su)\b', clause, re.IGNORECASE))
                match_is = re.search(
                    r'^(.*?)\s+(is|are|was|were|works as|serves as|leads|blocked by|requires|causes|enables|supports|uses|contains|produces|affects|results|belongs|creates|modifies|interacts|relates|connects|serves|runs|implements|triggers|drives|transfers|monitors|prevents|inhibits|activates|binds|encodes|executes|hosts|deploys|manages|controls|works|operates|built|designed|written|located|found|stored|retains|prunes|retrieves|extracts|processes|serializes|fetches|married|owns|traded|bought|lives in|está bloqueado por|nécessite|नेतृत्व करती हैं)\s+(.*)$',
                    clause,
                    re.IGNORECASE
                )

                if match_is:
                    raw_subj = match_is.group(1).strip()
                    verb_phrase = match_is.group(2).strip().upper().replace(" ", "_")
                    raw_obj = match_is.group(3).strip()

                    subj = self._clean_term(raw_subj) if raw_subj else (primary_subject or "Entity")
                    obj = self._clean_term(raw_obj)

                    if is_pronoun_clause and primary_subject:
                        subj = primary_subject

                    if subj and obj and not subj.lower().startswith("who married"):
                        triplets.append({
                            "source": subj if subj else "Entity",
                            "relation": verb_phrase,
                            "target": obj if obj else "Value",
                            "reason": sentence,
                            "category_src": "Entity",
                            "category_tgt": "Entity"
                        })
                elif primary_subject:
                    cleaned_obj = self._clean_term(clause)
                    if cleaned_obj and cleaned_obj != primary_subject and len(cleaned_obj) < 80:
                        triplets.append({
                            "source": primary_subject,
                            "relation": "ROLE",
                            "target": cleaned_obj,
                            "reason": sentence,
                            "category_src": "Person",
                            "category_tgt": "Role"
                        })

        return triplets

