from medhas.core import MedhasMemoryCore

_memory = MedhasMemoryCore()

def nexus_remember_text(raw_text: str) -> str:
    """
    Ingests raw, unstructured natural language text directly. Automatically extracts 
    entities, predicates, and context into long-term memory.
    """
    ingested = _memory.remember_raw_text(raw_text)
    if not ingested:
        return "Processed text, but no distinct entity relationships were identified."
    
    links = [f"({src}) --[{rel}]--> ({tgt})" for src, rel, tgt in ingested]
    return f"Successfully stored {len(links)} memory relationships:\n" + "\n".join(links)

def nexus_remember(source: str, relation: str, target: str, reason: str, modality: str = "text") -> str:
    src_id, tgt_id = _memory.remember(source, relation, target, reason, modality=modality)
    return f"Successfully stored memory link: ({src_id}) --[{relation.upper()}]--> ({tgt_id})"

def nexus_recall(query: str) -> str:
    return _memory.recall(query)

if __name__ == "__main__":
    print("Medhas MCP Server initialized.")
