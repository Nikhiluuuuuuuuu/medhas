from typing import List, Dict, Any

class SpreadingActivationEngine:
    def __init__(self, decay: float = 0.75, threshold: float = 0.15):
        self.decay = decay
        self.threshold = threshold

    def query(
        self,
        query_emb: list,
        kuzu_engine,
        vector_index,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        seeds = vector_index.search(query_emb, top_k=5)
        if not seeds:
            return []

        # Adaptive thresholding: filter seeds >= 0.40, falling back to >= 0.20 or top seeds
        activations: Dict[str, float] = {node_id: sim for node_id, sim in seeds if sim >= 0.40}
        if not activations:
            activations = {node_id: sim for node_id, sim in seeds if sim >= 0.20}
        if not activations and seeds:
            activations = {seeds[0][0]: seeds[0][1]}
            if len(seeds) > 1:
                activations[seeds[1][0]] = seeds[1][1]
        if not activations:
            return []

        retrieved_facts = []
        visited_edges = set()
        curr_frontier = dict(activations)

        for hop in range(max_hops):
            next_frontier: Dict[str, float] = {}
            for node_id, energy in curr_frontier.items():
                if energy < self.threshold:
                    continue

                query_cypher = '''
                    MATCH (a:Entity {id: $nid})-[r:CONNECTS]-(b:Entity)
                    WHERE r.valid_to = 0.0
                    RETURN a.id, r.relation, b.id, r.reason, r.salience, r.weight
                '''
                res = kuzu_engine.execute(query_cypher, {"nid": node_id}).get_as_df()

                for _, row in res.iterrows():
                    # Canonical unique key for undirected fact representation
                    fact_entities = tuple(sorted([str(row['a.id']), str(row['b.id'])]))
                    canonical_key = (fact_entities, str(row['r.relation']))

                    if canonical_key not in visited_edges:
                        visited_edges.add(canonical_key)
                        retrieved_facts.append({
                            "source": row['a.id'],
                            "relation": row['r.relation'],
                            "target": row['b.id'],
                            "reason": row['r.reason'],
                            "activation_score": round(energy * row['r.salience'], 4)
                        })

                    target_node = row['b.id']
                    propagated = energy * row['r.weight'] * row['r.salience'] * self.decay
                    next_frontier[target_node] = max(next_frontier.get(target_node, 0.0), propagated)

            curr_frontier = next_frontier

        retrieved_facts.sort(key=lambda x: x['activation_score'], reverse=True)
        return retrieved_facts
