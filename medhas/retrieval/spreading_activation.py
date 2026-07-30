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
        seeds = vector_index.search(query_emb, top_k=8)
        
        # Adaptive seed selection
        activations: Dict[str, float] = {}
        if seeds:
            for node_id, sim in seeds:
                if sim >= 0.35:
                    activations[node_id] = sim
            if not activations:
                for node_id, sim in seeds:
                    if sim >= 0.15:
                        activations[node_id] = sim
            if not activations:
                # Use top seeds
                for node_id, sim in seeds[:3]:
                    activations[node_id] = max(sim, 0.5)

        # Fallback: query all active nodes if no seeds found from vector index
        if not activations:
            try:
                df_all = kuzu_engine.execute("MATCH (e:Entity) RETURN e.id LIMIT 10").get_as_df()
                if not df_all.empty:
                    for nid in df_all['e.id']:
                        activations[nid] = 0.5
            except Exception:
                pass

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
                try:
                    res = kuzu_engine.execute(query_cypher, {"nid": node_id}).get_as_df()
                except Exception:
                    continue

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
                            "activation_score": round(float(energy) * float(row['r.salience']), 4)
                        })

                    target_node = row['b.id']
                    propagated = float(energy) * float(row['r.weight']) * float(row['r.salience']) * self.decay
                    next_frontier[target_node] = max(next_frontier.get(target_node, 0.0), propagated)

            curr_frontier = next_frontier

        retrieved_facts.sort(key=lambda x: x['activation_score'], reverse=True)
        return retrieved_facts

