Step1: Create graph schema and constraints
Run the following cypher query in Neo4j query tab:

// ────────────────────────────────────────────────────────────────
// 1️⃣  CORE ENTITY UNIQUENESS
// ────────────────────────────────────────────────────────────────
CREATE CONSTRAINT IF NOT EXISTS
  FOR (d:Dish)
  REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT IF NOT EXISTS
  FOR (c:Category)
  REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT IF NOT EXISTS
  FOR (cui:Cuisine)
  REQUIRE cui.name IS UNIQUE;

CREATE CONSTRAINT IF NOT EXISTS
  FOR (ing:Ingredient)
  REQUIRE ing.name IS UNIQUE;

CREATE CONSTRAINT IF NOT EXISTS
  FOR (all:Allergen)
  REQUIRE all.name IS UNIQUE;

CREATE CONSTRAINT IF NOT EXISTS
  FOR (dt:DietType)
  REQUIRE dt.name IS UNIQUE;

CREATE CONSTRAINT IF NOT EXISTS
  FOR (av:AvailabilityTime)
  REQUIRE av.name IS UNIQUE;


// ────────────────────────────────────────────────────────────────
// 2️⃣  NOT-NULL / EXISTENCE GUARANTEES (one prop per rule)
// ────────────────────────────────────────────────────────────────
CREATE CONSTRAINT IF NOT EXISTS
  FOR (d:Dish)
  REQUIRE d.name IS NOT NULL;

CREATE CONSTRAINT IF NOT EXISTS
  FOR (d:Dish)
  REQUIRE d.price IS NOT NULL;

CREATE CONSTRAINT IF NOT EXISTS
  FOR (c:Category)
  REQUIRE c.name IS NOT NULL;

CREATE CONSTRAINT IF NOT EXISTS
  FOR (cui:Cuisine)
  REQUIRE cui.name IS NOT NULL;


// ────────────────────────────────────────────────────────────────
// 3️⃣  LOOK-UP & FILTER INDEXES
// ────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS
  FOR (d:Dish)
  ON (d.name);

CREATE INDEX IF NOT EXISTS
  FOR (d:Dish)
  ON (d.available);

CREATE INDEX IF NOT EXISTS
  FOR (d:Dish)
  ON (d.spiceLevel);

CREATE INDEX IF NOT EXISTS
  FOR (ing:Ingredient)
  ON (ing.name);

CREATE INDEX IF NOT EXISTS
  FOR (all:Allergen)
  ON (all.name);


// ────────────────────────────────────────────────────────────────
// 4️⃣  OPTIONAL FULL-TEXT INDEX FOR SEARCH
// ────────────────────────────────────────────────────────────────
// full-text index for fuzzy search on dish name + description
CREATE FULLTEXT INDEX menu_search IF NOT EXISTS
FOR (n:Dish|Ingredient|Category|Cuisine)
ON EACH [n.name, n.description];




Step 2: Upload menu to the graph DB:
2.1 Create a menu in the following format: https://drive.google.com/file/d/1M1PRw6jb39JPB-Z0F6ABunzhfaidaeP4/view?usp=sharing

2.2. Upload the csv and make it downloadable, make sure that the gdrive csv is shared globally.

2.3 replace the id of the sheet and run the following cypher query in Neo4j query tab

LOAD CSV WITH HEADERS FROM
  'https://drive.usercontent.google.com/download?id=1M1PRw6jb39JPB-Z0F6ABunzhfaidaeP4' AS row
/* ── Split, trim & drop empties right here ─────────────────────────── */
WITH row,
     [i  IN split(coalesce(row.ingredients,''), ',')
         WHERE trim(i) <> '' | trim(i)]                             AS ingredients,
     [c  IN split(coalesce(row.best_with_category,''), ',')
         WHERE trim(c) <> '' | trim(c)]                             AS pairCats,
     [dt IN split(coalesce(row.dietary_tags,''), ',')
         WHERE trim(dt) <> '' | trim(dt)]                           AS dietTags,
     [al IN split(coalesce(row.allergens,''), ',')
         WHERE trim(al) <> '' | trim(al)]                           AS allergens,
     [at IN split(coalesce(row.availability_during,''), ',')
         WHERE trim(at) <> '' | trim(at)]                           AS availTags
/* ── Dish core node ────────────────────────────────────────────────── */
MERGE (d:Dish {id: row.id})
  ON CREATE SET d.name         = row.dish_name,
                d.description  = row.description,
                d.price        = toFloat(row.price),
                d.currency     = 'INR',
                d.prepTimeMin  = toInteger(row.prep_time_min),
                d.spiceLevel   = toInteger(row.spice_level),
                d.vegClass     = row.dietary_tags,
                d.isSignature  = (row.is_signature = '1'),
                d.available    = true
/* ── Category / Cuisine relationships ──────────────────────────────── */
MERGE (cat:Category {name: trim(row.category)})
MERGE (d)-[:IN_CATEGORY]->(cat)
MERGE (cui:Cuisine {name: trim(row.cuisine)})
MERGE (d)-[:OF_CUISINE]->(cui)
/* ── Ingredients & allergens ──────────────────────────────────────── */
FOREACH (ing IN ingredients |
  MERGE (i:Ingredient {name: ing})
  MERGE (d)-[:CONTAINS]->(i)
)
FOREACH (al IN allergens |
  MERGE (a:Allergen {name: al})
  MERGE (d)-[:HAS_ALLERGEN]->(a)
)
/* ── Diet-type & availability-time tags  (NEW) ─────────────────────── */
FOREACH (dt IN dietTags |
  MERGE (dtNode:DietType {name: dt})
  MERGE (d)-[:HAS_DIET_TYPE]->(dtNode)
)
FOREACH (at IN availTags |
  MERGE (atNode:AvailabilityTime {name: at})
  MERGE (d)-[:AVAILABLE_DURING]->(atNode)
)
/* ── “Best with” → category pairings ──────────────────────────────── */
FOREACH (bwCat IN pairCats |
  MERGE (compCat:Category {name: bwCat})
  MERGE (d)-[:PAIR_WITH_CATEGORY {source:'csv'}]->(compCat)
)


Step3: Create embed:
Run the following python function

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




RUn the following query to check if successful:

UNWIND ['Dish','Ingredient','Category','Cuisine'] AS L
MATCH (n)
WHERE L IN labels(n)              // node carries this label
  AND n.embedding IS NOT NULL     // embedding property present
RETURN
  L            AS label,
  count(n)     AS withEmbedding
ORDER BY label;



