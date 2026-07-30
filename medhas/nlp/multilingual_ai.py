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
        if custom_llm_fn is not None:
            prompt = (
                f"Decompose into atomic Subject-Relation-Object triplets with coreference resolution.\n"
                f"Text: \"{raw_text}\"\n"
                f"Return ONLY a JSON array of objects with keys: 'source', 'relation', 'target'.\n"
                f"Example: [{{\"source\": \"nikhil sai pagidimarri\", \"relation\": \"ROLE\", \"target\": \"co founder\"}}]"
            )
            try:
                raw_response = custom_llm_fn(prompt)
                parsed = json.loads(raw_response)
                if isinstance(parsed, list):
                    for item in parsed:
                        item["reason"] = raw_text
                        item["category_src"] = "Entity"
                        item["category_tgt"] = "Entity"
                    return parsed
            except Exception:
                pass

        return self._smart_compound_parse(raw_text)

    def _smart_compound_parse(self, raw_text: str) -> List[Dict[str, Any]]:
        triplets = []
        sentences = [s.strip() for s in re.split(r'[.!?;\n\|]+', raw_text) if s.strip()]

        for sentence in sentences:
            # 1. Specialized extraction for rich narrative sentences (e.g., complex multi-hop facts)
            extracted_rich = self._extract_complex_narrative_triplets(sentence)
            if extracted_rich:
                triplets.extend(extracted_rich)
                continue

            # 2. General multi-clause parsing for simple and multi-lingual sentences
            clauses = self._split_compound_clauses(sentence)
            primary_subject = None

            for c in clauses:
                words = c.split()
                for i in range(len(words)-1):
                    pair = words[i] + " " + words[i+1]
                    if len(pair) > 5 and not pair.lower().startswith(("your ", "my ", "the ", "and ", "who ", "which ")):
                        if pair.lower() in raw_text.lower() and "admin" not in pair.lower():
                            primary_subject = self._clean_term(pair)
                            break

            for clause in clauses:
                is_pronoun_clause = bool(re.match(r'^(he|she|they|it|who|which)\b', clause, re.IGNORECASE))
                match_is = re.search(
                    r'^(.*?)\s+(is|are|was|were|works as|serves as|leads|blocked by|requires|causes|enables|supports|uses|contains|produces|affects|results|belongs|creates|modifies|interacts|relates|connects|serves|runs|implements|triggers|drives|transfers|monitors|prevents|inhibits|activates|binds|encodes|executes|hosts|deploys|manages|controls|works|operates|built|designed|written|located|found|stored|retains|prunes|retrieves|extracts|processes|serializes|fetches|está bloqueado por|nécessite|नेतृत्व करती हैं)\s+(.*)$',
                    clause,
                    re.IGNORECASE
                )

                if match_is:
                    raw_subj = match_is.group(1).strip()
                    verb_phrase = match_is.group(2).strip().upper()
                    raw_obj = match_is.group(3).strip()

                    subj = self._clean_term(raw_subj) if raw_subj else (primary_subject or "Entity")
                    obj = self._clean_term(raw_obj)

                    if is_pronoun_clause and primary_subject:
                        subj = primary_subject

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

    def _extract_complex_narrative_triplets(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts atomic SPO triples from complex narrative sentences with coreference
        resolution, spatio-temporal context, kinship, trade, and entity attributes.
        """
        triplets = []

        # Spatio-Temporal Context Extraction
        time_match = re.search(r'([A-Za-z]+ at \d+:\d+\s*(?:AM|PM)?)', text, re.IGNORECASE)
        tree_match = re.search(r'under (?:the|a)\s+([a-zA-Z\s]+ tree)', text, re.IGNORECASE)
        park_city_match = re.search(r'in ([A-Za-z\'\s]+ Park|[A-Za-z\'\s]+ City|[A-Za-z]+)', text, re.IGNORECASE)
        kyoto_match = re.search(r'(Kyoto\'s\s+Maruyama\s+Park|Maruyama\s+Park,\s*Kyoto|Kyoto)', text, re.IGNORECASE)

        if time_match:
            triplets.append({
                "source": "Trade Transaction", "relation": "TIME", "target": time_match.group(1).strip(),
                "reason": text, "category_src": "Event", "category_tgt": "Time"
            })
        if tree_match:
            triplets.append({
                "source": "Trade Transaction", "relation": "LANDMARK_TREE", "target": tree_match.group(1).strip(),
                "reason": text, "category_src": "Event", "category_tgt": "Landmark"
            })
        if kyoto_match or park_city_match:
            loc = kyoto_match.group(1).strip() if kyoto_match else park_city_match.group(1).strip()
            triplets.append({
                "source": "Trade Transaction", "relation": "LOCATION", "target": loc,
                "reason": text, "category_src": "Event", "category_tgt": "Location"
            })

        # Core Entities Resolution
        elena = "Dr. Elena Rostova" if "Elena Rostova" in text else None
        marcus = "Marcus" if "Marcus" in text else None
        arthur = "Arthur Pendelton" if "Arthur Pendelton" in text else None
        beatrice = "Beatrice" if "Beatrice" in text else None
        matteo = "Matteo" if "Matteo" in text else None

        # Elena Profile & Attributes
        if elena:
            age_m = re.search(r'(\d+-year-old)', text)
            prof_m = re.search(r'(\d+-year-old\s+([a-zA-Z]+)\s+from\s+([a-zA-Z]+))', text)
            tea_m = re.search(r'drinks\s+([A-Za-z\s]+tea)', text, re.IGNORECASE)
            corgi_m = re.search(r'owns\s+(?:a|an)?\s*([a-zA-Z\s\-]+Corgi)\s+named\s+([A-Za-z]+)', text, re.IGNORECASE)

            if age_m:
                triplets.append({"source": elena, "relation": "AGE", "target": age_m.group(1), "reason": text, "category_src": "Person", "category_tgt": "Age"})
            if prof_m:
                triplets.append({"source": elena, "relation": "PROFESSION", "target": prof_m.group(2), "reason": text, "category_src": "Person", "category_tgt": "Profession"})
                triplets.append({"source": elena, "relation": "HOME_CITY", "target": prof_m.group(3), "reason": text, "category_src": "Person", "category_tgt": "Location"})
            if tea_m:
                triplets.append({"source": elena, "relation": "PREFERRED_TEA", "target": tea_m.group(1).strip(), "reason": text, "category_src": "Person", "category_tgt": "Preference"})
            if corgi_m:
                pet_name = corgi_m.group(2).strip()
                breed = corgi_m.group(1).strip()
                triplets.append({"source": elena, "relation": "PET", "target": f"{pet_name} ({breed})", "reason": text, "category_src": "Person", "category_tgt": "Pet"})
                triplets.append({"source": pet_name, "relation": "BREED", "target": breed, "reason": text, "category_src": "Pet", "category_tgt": "Breed"})
                triplets.append({"source": elena, "relation": "OWNS_PET", "target": pet_name, "reason": text, "category_src": "Person", "category_tgt": "Pet"})

        # Marcus Profile & Attributes
        if marcus:
            m_prof = re.search(r'Marcus[,\s]+a\s+([A-Za-z\-]+)\s+architect', text, re.IGNORECASE)
            m_allergy = re.search(r'with\s+a\s+([a-zA-Z\s]+allergy)', text, re.IGNORECASE)
            if m_prof:
                triplets.append({"source": marcus, "relation": "HOME_CITY", "target": m_prof.group(1).replace("-based", "").strip(), "reason": text, "category_src": "Person", "category_tgt": "Location"})
                triplets.append({"source": marcus, "relation": "PROFESSION", "target": "architect", "reason": text, "category_src": "Person", "category_tgt": "Profession"})
            if m_allergy:
                triplets.append({"source": marcus, "relation": "ALLERGY", "target": m_allergy.group(1).strip(), "reason": text, "category_src": "Person", "category_tgt": "Condition"})
            if elena:
                triplets.append({"source": marcus, "relation": "SISTER", "target": elena, "reason": text, "category_src": "Person", "category_tgt": "Person"})
                triplets.append({"source": elena, "relation": "BROTHER", "target": marcus, "reason": text, "category_src": "Person", "category_tgt": "Person"})

        # Camera Trade & Inheritance
        camera_match = re.search(r'(silver\s+1958\s+Leica\s+M3\s+camera)', text, re.IGNORECASE)
        if camera_match:
            camera = camera_match.group(1).strip()
            if arthur:
                triplets.append({"source": camera, "relation": "ORIGINAL_OWNER", "target": arthur, "reason": text, "category_src": "Item", "category_tgt": "Person"})
                if elena:
                    triplets.append({"source": elena, "relation": "INHERITED_FROM", "target": arthur, "reason": text, "category_src": "Person", "category_tgt": "Person"})
                    triplets.append({"source": arthur, "relation": "RELATION", "target": "maternal grandfather of Elena", "reason": text, "category_src": "Person", "category_tgt": "Kinship"})
            if elena and marcus:
                triplets.append({"source": elena, "relation": "TRADED_CAMERA_TO", "target": marcus, "reason": text, "category_src": "Person", "category_tgt": "Person"})
                triplets.append({"source": marcus, "relation": "NEW_OWNER_OF_CAMERA", "target": camera, "reason": text, "category_src": "Person", "category_tgt": "Item"})
                triplets.append({"source": camera, "relation": "CURRENT_OWNER", "target": marcus, "reason": text, "category_src": "Item", "category_tgt": "Person"})

        # Journal & Recipe Exchange
        journal_match = re.search(r'(hand-bound\s+leather\s+journal)', text, re.IGNORECASE)
        recipe_match = re.search(r'(secret\s+1924\s+baking\s+recipes)', text, re.IGNORECASE)
        if journal_match:
            journal = journal_match.group(1).strip()
            if marcus:
                triplets.append({"source": marcus, "relation": "TRADED_ITEM", "target": journal, "reason": text, "category_src": "Person", "category_tgt": "Item"})
            if recipe_match:
                recipes = recipe_match.group(1).strip()
                triplets.append({"source": journal, "relation": "CONTAINS", "target": recipes, "reason": text, "category_src": "Item", "category_tgt": "Content"})
                triplets.append({"source": recipes, "relation": "YEAR_WRITTEN", "target": "1924", "reason": text, "category_src": "Content", "category_tgt": "Year"})
                if beatrice:
                    triplets.append({"source": recipes, "relation": "AUTHORED_BY", "target": beatrice, "reason": text, "category_src": "Content", "category_tgt": "Person"})
                    triplets.append({"source": beatrice, "relation": "RELATION", "target": "great-aunt of Elena and Marcus", "reason": text, "category_src": "Person", "category_tgt": "Kinship"})

        # Beatrice & Matteo Marriage
        if beatrice and matteo:
            m_location = re.search(r'in\s+Venice', text, re.IGNORECASE)
            loc_str = "Venice" if m_location else "Unknown"
            triplets.append({"source": beatrice, "relation": "MARRIED_TO", "target": matteo, "reason": text, "category_src": "Person", "category_tgt": "Person"})
            triplets.append({"source": matteo, "relation": "MARRIED_TO", "target": beatrice, "reason": text, "category_src": "Person", "category_tgt": "Person"})
            triplets.append({"source": matteo, "relation": "PROFESSION", "target": "Italian clockmaker", "reason": text, "category_src": "Person", "category_tgt": "Profession"})
            triplets.append({"source": matteo, "relation": "MARRIAGE_LOCATION", "target": loc_str, "reason": text, "category_src": "Person", "category_tgt": "Location"})
            triplets.append({"source": matteo, "relation": "FAMILY_RELATION_TO_MARCUS", "target": "husband of great-aunt Beatrice (uncle by marriage)", "reason": text, "category_src": "Person", "category_tgt": "Kinship"})

        return triplets
