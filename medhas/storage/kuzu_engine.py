import os
import kuzu

class KuzuStorageEngine:
    def __init__(self, db_path: str = "./medhas_kuzu_db"):
        self.db_path = db_path
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._initialize_schema()

    def _initialize_schema(self):
        try:
            self.conn.execute('''
                CREATE NODE TABLE Entity(
                    id STRING,
                    category STRING,
                    embedding FLOAT[384],
                    created_at DOUBLE,
                    last_accessed DOUBLE,
                    access_count INT64,
                    PRIMARY KEY (id)
                )
            ''')
        except Exception:
            pass

        try:
            self.conn.execute('''
                CREATE REL TABLE CONNECTS(
                    FROM Entity TO Entity,
                    relation STRING,
                    reason STRING,
                    salience DOUBLE,
                    weight DOUBLE,
                    valid_from DOUBLE,
                    valid_to DOUBLE,
                    modality STRING
                )
            ''')
        except Exception:
            pass

    def execute(self, query: str, parameters: dict = None):
        if parameters:
            return self.conn.execute(query, parameters)
        return self.conn.execute(query)
