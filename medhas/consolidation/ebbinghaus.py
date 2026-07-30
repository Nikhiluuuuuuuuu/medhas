import time
import math

class EbbinghausScrubber:
    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold

    def run_scrubber(self, kuzu_engine):
        now = time.time()
        df = kuzu_engine.execute('''
            MATCH (a:Entity)-[r:CONNECTS]->(b:Entity)
            WHERE r.valid_to = 0.0
            RETURN a.id, r.relation, b.id, r.salience, a.last_accessed, a.access_count
        ''').get_as_df()

        for _, row in df.iterrows():
            last_accessed = row['a.last_accessed']
            access_count = row['a.access_count']
            delta_days = (now - last_accessed) / (24 * 3600)
            stability = 1.0 + math.log(1 + access_count)
            retained_salience = row['r.salience'] * math.exp(-delta_days / stability)

            if retained_salience < self.threshold and access_count < 3:
                kuzu_engine.execute('''
                    MATCH (a:Entity {id: $src})-[r:CONNECTS {relation: $rel}]->(b:Entity {id: $tgt})
                    SET r.valid_to = $now, r.salience = $sal
                ''', {"src": row['a.id'], "rel": row['r.relation'], "tgt": row['b.id'], "now": now, "sal": retained_salience})
            else:
                kuzu_engine.execute('''
                    MATCH (a:Entity {id: $src})-[r:CONNECTS {relation: $rel}]->(b:Entity {id: $tgt})
                    SET r.salience = $sal
                ''', {"src": row['a.id'], "rel": row['r.relation'], "tgt": row['b.id'], "sal": retained_salience})
