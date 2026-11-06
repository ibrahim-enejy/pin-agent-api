import os
import psycopg2
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

app = FastAPI(title="Pin Agent Hybrid Search API")

print("Loading local embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Uses Groq by default (100% free, no card). Override LLM_BASE_URL if using Nebius.
llm_client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.getenv("LLM_API_KEY")
)
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
DB_URL = os.getenv("DATABASE_URL")

class MessageInput(BaseModel):
    chat_id: str
    text: str

def save_to_db(chat_id: str, source_type: str, raw_content: str, summary: str):
    """Embeds the concise summary into 384 numbers and saves to Neon Postgres."""
    vector = embedder.encode(summary).tolist()
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO saved_context (chat_id, source_type, raw_content, summary, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        """,
        (chat_id, source_type, raw_content, summary, str(vector))
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"[SAVED] Type: {source_type} | Summary: {summary}")

def async_process_message(chat_id: str, text: str, is_manual: bool):
    """Background worker for manual pin summarization and autonomous artifact extraction."""
    if is_manual:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Summarize the core technical takeaway or facts of this message in 2 concise sentences for a search index."},
                {"role": "user", "content": text}
            ]
        )
        summary = resp.choices[0].message.content.strip()
        save_to_db(chat_id, "manual_pin", text, summary)
    else:
        resp = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an artifact extraction agent. If the user's text contains a reusable code block, CLI command, configuration, or architectural decision, output a 1-sentence summary, followed by '|||', followed by the extracted artifact. If the message is conversational filler or lacks a concrete artifact, reply with the exact word 'NONE'."},
                {"role": "user", "content": text}
            ]
        )
        output = resp.choices[0].message.content.strip()
        if output != "NONE" and "|||" in output:
            summary, artifact = output.split("|||", 1)
            save_to_db(chat_id, "async_artifact", artifact.strip(), summary.strip())
        else:
            print("[IGNORED] No high-value artifact detected.")

@app.post("/pin")
def manual_pin(msg: MessageInput, bg_tasks: BackgroundTasks):
    bg_tasks.add_task(async_process_message, msg.chat_id, msg.text, True)
    return {"status": "Pin queued for background indexing"}

@app.post("/observe")
def observe_stream(msg: MessageInput, bg_tasks: BackgroundTasks):
    bg_tasks.add_task(async_process_message, msg.chat_id, msg.text, False)
    return {"status": "Agent observing message in background"}

@app.get("/search")
def hybrid_search(chat_id: str, q: str):
    query_vector = embedder.encode(q).tolist()
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    sql = """
    WITH semantic_search AS (
        SELECT id, RANK() OVER (ORDER BY embedding <=> %s::vector) AS rank
        FROM saved_context WHERE chat_id = %s LIMIT 20
    ),
    keyword_search AS (
        SELECT id, RANK() OVER (ORDER BY ts_rank_cd(fts, websearch_to_tsquery('english', %s)) DESC) AS rank
        FROM saved_context WHERE chat_id = %s AND fts @@ websearch_to_tsquery('english', %s) LIMIT 20
    )
    SELECT 
        c.id, c.source_type, c.summary, c.raw_content,
        COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS rrf_score
    FROM semantic_search s
    FULL OUTER JOIN keyword_search k ON s.id = k.id
    JOIN saved_context c ON c.id = COALESCE(s.id, k.id)
    ORDER BY rrf_score DESC
    LIMIT 5;
    """
    cur.execute(sql, (str(query_vector), chat_id, q, chat_id, q))
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return [{"id": r[0], "type": r[1], "summary": r[2], "content": r[3], "score": float(r[4])} for r in results]