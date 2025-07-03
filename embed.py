"""
fast_embed.py  ─  Neo4j → OpenAI → Neo4j  (single vector index)

• Pulls ONLY Dish / Ingredient / Category / Cuisine nodes that still miss `embedding`
• Generates embeddings in async BATCHES (default 64) with text-embedding-3-small
• Writes them back in one UNWIND query per batch & tags nodes :Searchable
• Ensures single vector index  menu_embed  +  full-text index  menu_search

Prereqs
-------
pip install neo4j openai aiohttp tenacity tqdm python-dotenv
.env must contain:  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY
"""

import os, asyncio, math, json, aiohttp, time
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# ───────────────────────── CONFIG ─────────────────────────
load_dotenv()
OPENAI_KEY      = os.getenv("OPENAI_API_KEY")
NEO4J_URI       = os.getenv("NEO4J_URI")
NEO4J_USER      = os.getenv("NEO4J_USER")
NEO4J_PASSWORD  = os.getenv("NEO4J_PASSWORD")

MODEL           = "text-embedding-3-small"
DIMENSIONS      = 1536
BATCH_SIZE      = 64            # keep combined tokens < 2048
HEADERS         = {"Authorization": f"Bearer {OPENAI_KEY}",
                   "Content-Type": "application/json"}

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ────────────────── ASYNC OPENAI CALL ──────────────────
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, max=8))
async def embed_batch(session, texts):
    url = "https://api.openai.com/v1/embeddings"
    async with session.post(url, json={"model": MODEL, "input": texts},
                            headers=HEADERS, timeout=60) as resp:
        if resp.status != 200:
            raise RuntimeError(await resp.text())
        data = await resp.json()
        return [d["embedding"] for d in data["data"]]

# ──────────── FETCH NODES STILL MISSING EMBEDDING ────────────
FETCH_Q = """
MATCH (n)
WHERE n.embedding IS NULL
  AND any(l IN labels(n) WHERE l IN ['Dish','Ingredient','Category','Cuisine'])
RETURN elementId(n) AS eid,
       CASE WHEN 'Dish' IN labels(n)
            THEN coalesce(n.name,'') + '. ' + coalesce(n.description,'')
            ELSE n.name END        AS text
"""

def fetch_rows():
    with driver.session() as s:
        return s.run(FETCH_Q).data()

# ─────────── BULK WRITE BACK EMBEDDINGS ───────────
WRITE_Q = """
UNWIND $rows AS row
MATCH (n) WHERE elementId(n) = row.eid
SET n.embedding = row.vec,
    n:Searchable
"""

def write_rows(rows):
    with driver.session() as s:
        s.execute_write(lambda tx: tx.run(WRITE_Q, rows=rows))

# ─────────── ENSURE INDEXES EXIST (NO DROP) ───────────
def create_indexes():
    VEC_IDX = f"""
    CREATE VECTOR INDEX menu_embed IF NOT EXISTS
    FOR (n:Searchable) ON (n.embedding)
    OPTIONS {{indexConfig:{{`vector.dimensions`:{DIMENSIONS},
                           `vector.similarity_function`:'cosine'}}}}
    """
    FTS_IDX = """
    CREATE FULLTEXT INDEX menu_search IF NOT EXISTS
    FOR (n:Searchable) ON EACH [n.name, n.description]
    """
    with driver.session() as s:
        s.execute_write(lambda tx: tx.run(VEC_IDX))
        s.execute_write(lambda tx: tx.run(FTS_IDX))

# ─────────── MAIN PIPELINE ───────────
async def main():
    rows = fetch_rows()
    if not rows:
        print("All nodes already embedded ✓")
        create_indexes()
        return

    print(f"Embedding {len(rows)} nodes …")
    async with aiohttp.ClientSession() as http:
        for i in tqdm(range(0, len(rows), BATCH_SIZE)):
            chunk = rows[i:i+BATCH_SIZE]
            texts = [r["text"] for r in chunk]
            vecs  = await embed_batch(http, texts)
            write_rows([{"eid": r["eid"], "vec": v} for r, v in zip(chunk, vecs)])

    create_indexes()
    print("Embeddings + index complete ✓")

if __name__ == "__main__":
    asyncio.run(main())