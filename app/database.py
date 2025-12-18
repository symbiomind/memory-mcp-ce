
import psycopg2
from app.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB

def get_db_connection():
    """Establishes connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB
    )
    return conn

def init_database(embedding_dim: int):
    """Initializes the database schema, creating the table if it doesn't exist."""
    table_name = f"memory_{embedding_dim}"
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id BIGSERIAL PRIMARY KEY,
            content BYTEA NOT NULL,
            embedding VECTOR({embedding_dim}) NOT NULL,
            namespace VARCHAR(100) DEFAULT 'default',
            labels JSONB DEFAULT '[]'::JSONB,
            source VARCHAR(255),
            timestamp TIMESTAMP DEFAULT NOW(),
            enc BOOLEAN DEFAULT FALSE,
            embedding_model VARCHAR(255) NOT NULL,
            state JSONB DEFAULT '{{}}'::JSONB
        );
    """)

    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_embedding_{embedding_dim} ON {table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """)

    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_labels_gin_{embedding_dim} ON {table_name} USING GIN (labels);
    """)

    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_namespace_{embedding_dim} ON {table_name} (namespace);
    """)

    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_source_{embedding_dim} ON {table_name} (source);
    """)

    conn.commit()
    cur.close()
    conn.close()
