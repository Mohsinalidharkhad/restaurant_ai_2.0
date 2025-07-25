

// 1. Create constraint
CREATE CONSTRAINT faq_id_unique IF NOT EXISTS FOR (f:FAQ) REQUIRE f.id IS UNIQUE;

// 2. Upload CSV data (minimal structure)
LOAD CSV WITH HEADERS FROM 'https://drive.usercontent.google.com/download?id=1QfH-lqi-VHeYecHHeHnPWpSGEJb19nrt' AS row
CREATE (f:FAQ:Searchable {
    id: row.ID,
    question: trim(row.Question),
    answer: trim(row.Answer),
    embedding: null,
    created_at: datetime()
});

// 3. Create vector index
CREATE VECTOR INDEX faq_embed IF NOT EXISTS
FOR (f:FAQ) ON (f.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }
};

// 4. Verify
MATCH (f:FAQ) RETURN count(f) as total_faqs;