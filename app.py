"""
CodeLens - Feature Intelligence Explorer

FastAPI app with local embeddings + Qdrant vector search + local LLM for RAG.
Serves an interactive UI for exploring feature history from Jira, Slack, and code.

Usage:
    cd codelens
    docker compose up -d  # Start Qdrant
    uv run python pipeline/ingest.py  # Ingest data
    uv run python app.py  # Start server
"""

import os
import sys
import time
import json
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    Fusion,
    FusionQuery,
)
from pydantic import BaseModel

load_dotenv()

from shared.llm import init_llm, get_llm_response, get_model_name, is_available as llm_available
from shared.embeddings import init_embeddings, get_embedding, is_available as embed_available

# Paths
MODELS_DIR = Path(__file__).parent.parent / "models"
EMBED_MODEL_PATH = MODELS_DIR / "nomic-embed-text" / "nomic-embed-text-v1.5.f16.gguf"
LLM_MODEL_PATH = MODELS_DIR / "Qwen3-4B-Q4_K_M" / "Qwen3-4B-Q4_K_M.gguf"
LLM_FALLBACK_PATH = MODELS_DIR / "cognee-distillabs-model-gguf-quantized" / "model-quantized.gguf"

# Qdrant config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "codelens_documents"

qdrant = QdrantClient(url=QDRANT_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    print("=" * 60)
    print("CodeLens - Feature Intelligence Explorer")
    print("=" * 60)
    
    # Initialize embeddings
    print(f"\nInitializing embedding model...")
    init_embeddings(str(EMBED_MODEL_PATH))
    
    # Initialize LLM
    print(f"\nInitializing LLM...")
    init_llm([
        (str(LLM_MODEL_PATH), "Qwen3-4B"),
        (str(LLM_FALLBACK_PATH), "Distil-Labs"),
    ])
    
    # Check Qdrant connection
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        print(f"\nQdrant collection: {COLLECTION_NAME}")
        print(f"  Points: {info.points_count}")
    except Exception as e:
        print(f"\nWARNING: Could not connect to Qdrant: {e}")
        print("Run: docker compose up -d && uv run python pipeline/ingest.py")
    
    # Initialize cognee integration
    print("\nInitializing cognee...")
    await init_cognee()
    
    print("\n" + "=" * 60)
    print(f"Server ready at http://localhost:8888")
    print("=" * 60 + "\n")
    
    yield


app = FastAPI(title="CodeLens - Feature Intelligence Explorer", lifespan=lifespan)


# =============================================================================
# OpenAI-compatible endpoint (so cognee can use our local Qwen3-4B)
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-4b"
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.0
    stream: bool = False

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint using our local Qwen3-4B.
    This lets cognee use our already-loaded local LLM for cognify()."""
    system_prompt = ""
    user_prompt = ""
    
    for msg in request.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_prompt = msg.content
    
    if not user_prompt:
        # Combine all messages as user prompt if no explicit user message
        user_prompt = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
    
    try:
        # Add instruction to suppress chain-of-thought for structured output
        if system_prompt:
            system_prompt = system_prompt + "\n\nIMPORTANT: Respond directly without <think> tags. Output ONLY the requested content."
        else:
            system_prompt = "Respond directly without <think> tags. Output ONLY the requested content."
        
        answer = get_llm_response(system_prompt, user_prompt, max_tokens=request.max_tokens)
        
        # Strip Qwen3 <think>...</think> blocks from response
        import re
        answer = re.sub(r'<think>.*?</think>\s*', '', answer, flags=re.DOTALL).strip()
        # Also handle unclosed think tags
        if '<think>' in answer:
            answer = answer.split('</think>')[-1].strip() if '</think>' in answer else re.sub(r'<think>.*', '', answer, flags=re.DOTALL).strip()
    except Exception as e:
        answer = f"Error: {e}"
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": get_model_name(),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/api/search")
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    source_type: str = Query(None, description="Filter by source: jira, slack, person, feature"),
    feature_id: str = Query(None, description="Filter by feature ID"),
):
    """
    Semantic search with Qdrant Prefetch + RRF Fusion.
    Returns relevant documents from Jira tickets, Slack messages, etc.
    """
    t0 = time.time()
    
    # Embed query
    query_vector = get_embedding(q)
    embed_ms = round((time.time() - t0) * 1000, 1)
    
    # Build filter
    filter_conditions = []
    if source_type:
        filter_conditions.append(FieldCondition(key="source_type", match=MatchValue(value=source_type)))
    if feature_id:
        filter_conditions.append(FieldCondition(key="feature_id", match=MatchValue(value=feature_id)))
    
    query_filter = Filter(must=filter_conditions) if filter_conditions else None
    
    # Search with Prefetch + RRF Fusion
    t1 = time.time()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=query_vector, limit=50),
            Prefetch(query=query_vector, limit=25),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    search_ms = round((time.time() - t1) * 1000, 1)
    
    # Format results
    items = []
    for point in results.points:
        payload = point.payload or {}
        items.append({
            "id": str(point.id),
            "score": round(point.score, 4),
            "text": payload.get("text", ""),
            "source_type": payload.get("source_type", ""),
            "source_id": payload.get("source_id", ""),
            "feature_id": payload.get("feature_id", ""),
            "timestamp": payload.get("timestamp", ""),
            "title": payload.get("title", ""),
            "user": payload.get("user", ""),
            "channel": payload.get("channel", ""),
            "entity_type": payload.get("entity_type", ""),
        })
    
    return {
        "query": q,
        "results": items,
        "total": len(items),
        "time_ms": round((time.time() - t0) * 1000, 1),
        "embed_ms": embed_ms,
        "search_ms": search_ms,
    }


@app.get("/api/ask")
async def ask(
    q: str = Query(..., description="Question to answer"),
    feature_id: str = Query(None, description="Filter by feature ID"),
    limit: int = Query(8, ge=1, le=20),
):
    """
    RAG Q&A: retrieve context from Qdrant, synthesize answer via local LLM.
    Returns a narrative answer with source citations.
    """
    t0 = time.time()
    
    # Embed query
    query_vector = get_embedding(q)
    
    # Build filter
    query_filter = None
    if feature_id:
        query_filter = Filter(must=[FieldCondition(key="feature_id", match=MatchValue(value=feature_id))])
    
    # Retrieve context
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=query_vector, limit=30),
            Prefetch(query=query_vector, limit=15),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    
    # Build context for LLM
    context_docs = []
    sources = []
    for point in results.points:
        payload = point.payload or {}
        text = payload.get("text", "")[:800]
        source_type = payload.get("source_type", "unknown")
        source_id = payload.get("source_id", "")
        
        context_docs.append(f"[{source_type.upper()}: {source_id}]\n{text}")
        sources.append({
            "id": str(point.id),
            "source_type": source_type,
            "source_id": source_id,
            "title": payload.get("title", payload.get("message_text", "")[:50]),
            "timestamp": payload.get("timestamp", ""),
            "user": payload.get("user", ""),
            "score": round(point.score, 4),
        })
    
    context = "\n\n---\n\n".join(context_docs)
    retrieval_ms = round((time.time() - t0) * 1000, 1)
    
    # Generate answer with LLM
    t1 = time.time()
    system_prompt = """You are a feature intelligence analyst. Synthesize Jira tickets, Slack messages, and docs into a clear, structured answer.

STRICT FORMAT RULES — follow this exact markdown structure:

### Summary
One sentence answering the question directly.

### Key Points
- Bullet point 1 (most important fact)
- Bullet point 2
- Bullet point 3

### People Involved
- **person_name** — what they did

### Timeline
- **date** — what happened

### References
- JIRA-XXXX, relevant ticket IDs

RULES:
- Be concise. No filler. Max 150 words total.
- Use ONLY the context provided. If info is missing, say "Not enough context."
- Always use the markdown headers above. Skip a section if no relevant info."""

    user_prompt = f"""Question: {q}

Context:
{context}"""

    try:
        answer = get_llm_response(system_prompt, user_prompt, max_tokens=800)
    except Exception as e:
        answer = f"Error generating answer: {e}"
    
    llm_ms = round((time.time() - t1) * 1000, 1)
    
    return {
        "question": q,
        "answer": answer,
        "sources": sources,
        "source_count": len(sources),
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "model": get_model_name(),
    }


@app.get("/api/timeline")
async def timeline(
    feature_id: str = Query(None, description="Filter by feature ID"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Get chronologically ordered events for a feature.
    Returns timeline of Jira updates, Slack messages, etc.
    """
    t0 = time.time()
    
    # Build filter for non-entity documents
    filter_conditions = [
        FieldCondition(key="entity_type", match=MatchValue(value="ticket")),
    ]
    if feature_id:
        filter_conditions.append(FieldCondition(key="feature_id", match=MatchValue(value=feature_id)))
    
    # Get tickets
    tickets = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=filter_conditions),
        limit=limit,
        with_payload=True,
    )[0]
    
    # Get messages
    filter_conditions[0] = FieldCondition(key="entity_type", match=MatchValue(value="message"))
    messages = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=filter_conditions),
        limit=limit,
        with_payload=True,
    )[0]
    
    # Combine and format
    events = []
    
    for point in tickets:
        payload = point.payload or {}
        events.append({
            "id": str(point.id),
            "type": "jira",
            "source_id": payload.get("source_id", ""),
            "title": payload.get("title", ""),
            "timestamp": payload.get("timestamp", ""),
            "status": payload.get("status", ""),
            "text": payload.get("text", "")[:200],
        })
    
    for point in messages:
        payload = point.payload or {}
        events.append({
            "id": str(point.id),
            "type": "slack",
            "source_id": payload.get("source_id", ""),
            "user": payload.get("user", ""),
            "channel": payload.get("channel", ""),
            "timestamp": payload.get("timestamp", ""),
            "text": payload.get("message_text", "")[:200],
        })
    
    # Sort by timestamp
    events.sort(key=lambda x: x.get("timestamp", ""), reverse=False)
    
    return {
        "events": events,
        "total": len(events),
        "time_ms": round((time.time() - t0) * 1000, 1),
    }


@app.get("/api/graph")
async def graph(
    feature_id: str = Query(None, description="Filter by feature ID"),
    q: str = Query(None, description="Optional search query to highlight relevant nodes"),
):
    """
    Get entity-relationship graph data for D3 visualization.
    Returns nodes (people, tickets, features) and edges (relationships).
    """
    t0 = time.time()
    
    # Build filter
    filter_conditions = []
    if feature_id:
        filter_conditions.append(FieldCondition(key="feature_id", match=MatchValue(value=feature_id)))
    
    query_filter = Filter(must=filter_conditions) if filter_conditions else None
    
    # Get all documents
    all_docs = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=query_filter,
        limit=200,
        with_payload=True,
    )[0]
    
    # If search query, get relevant doc IDs
    relevant_ids = set()
    if q:
        query_vector = get_embedding(q)
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=10,
            with_payload=True,
        )
        relevant_ids = {str(p.id) for p in results.points}
    
    # Build graph
    nodes = []
    edges = []
    seen_nodes = {}
    
    # Process documents into nodes
    for point in all_docs:
        payload = point.payload or {}
        entity_type = payload.get("entity_type", "")
        source_type = payload.get("source_type", "")
        source_id = payload.get("source_id", "")
        
        # Determine node type
        if entity_type == "person":
            node_type = "person"
            label = payload.get("name", source_id)
        elif entity_type == "feature":
            node_type = "feature"
            label = payload.get("name", source_id)
        elif source_type == "jira":
            node_type = "ticket"
            label = source_id
        elif source_type == "slack":
            node_type = "message"
            label = f"{payload.get('user', '')} in {payload.get('channel', '')}"
        else:
            continue
        
        # Create node if not seen
        if source_id not in seen_nodes:
            node = {
                "id": source_id,
                "label": label,
                "type": node_type,
                "highlighted": str(point.id) in relevant_ids,
                "payload": {
                    "timestamp": payload.get("timestamp", ""),
                    "text": payload.get("text", "")[:100],
                },
            }
            nodes.append(node)
            seen_nodes[source_id] = node
    
    # Build edges based on relationships
    feature_node = None
    people = []
    tickets = []
    messages = []
    
    for node in nodes:
        if node["type"] == "feature":
            feature_node = node
        elif node["type"] == "person":
            people.append(node)
        elif node["type"] == "ticket":
            tickets.append(node)
        elif node["type"] == "message":
            messages.append(node)
    
    # Connect people to feature
    if feature_node:
        for person in people:
            edges.append({
                "source": person["id"],
                "target": feature_node["id"],
                "label": "contributed to",
            })
        for ticket in tickets:
            edges.append({
                "source": ticket["id"],
                "target": feature_node["id"],
                "label": "implements",
            })
    
    # Connect messages to people
    for point in all_docs:
        payload = point.payload or {}
        if payload.get("source_type") == "slack":
            user = payload.get("user", "")
            source_id = payload.get("source_id", "")
            if user in seen_nodes and source_id in seen_nodes:
                edges.append({
                    "source": user,
                    "target": source_id,
                    "label": "posted",
                })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "time_ms": round((time.time() - t0) * 1000, 1),
    }


@app.get("/api/features")
async def list_features():
    """List all ingested features."""
    t0 = time.time()
    
    # Get all feature entities
    results = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[
            FieldCondition(key="entity_type", match=MatchValue(value="feature"))
        ]),
        limit=100,
        with_payload=True,
    )[0]
    
    features = []
    for point in results:
        payload = point.payload or {}
        features.append({
            "id": payload.get("feature_id", ""),
            "name": payload.get("name", ""),
            "description": payload.get("text", "")[:200],
            "timestamp": payload.get("timestamp", ""),
        })
    
    return {
        "features": features,
        "total": len(features),
        "time_ms": round((time.time() - t0) * 1000, 1),
    }


@app.get("/api/evidence/{doc_id}")
async def get_evidence(doc_id: str):
    """Get full details for a specific document/evidence."""
    t0 = time.time()
    
    try:
        results = qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[doc_id],
            with_payload=True,
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="Document not found")
        
        point = results[0]
        payload = point.payload or {}
        
        return {
            "id": str(point.id),
            "source_type": payload.get("source_type", ""),
            "source_id": payload.get("source_id", ""),
            "feature_id": payload.get("feature_id", ""),
            "text": payload.get("text", ""),
            "timestamp": payload.get("timestamp", ""),
            "title": payload.get("title", ""),
            "user": payload.get("user", ""),
            "channel": payload.get("channel", ""),
            "status": payload.get("status", ""),
            "references": payload.get("references", []),
            "time_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Cognee Integration Endpoints
# =============================================================================

COGNEE_ENABLED = False

async def init_cognee():
    """Initialize cognee with LanceDB (vectors) + OpenAI (LLM for cognify).
    
    Why LanceDB not Qdrant? Cognee v0.5.x only supports LanceDB/PGVector/ChromaDB
    as built-in vector providers. Qdrant is NOT in cognee's supported list.
    So cognee stores its own KG vectors in LanceDB (local, zero-config),
    while our main app continues to use Qdrant for search/RAG directly.
    """
    global COGNEE_ENABLED
    try:
        import cognee
        
        # --- Vector DB: LanceDB (cognee's default, works out of the box) ---
        # DO NOT set qdrant - cognee doesn't support it as a vector provider
        cognee.config.set_vector_db_provider("lancedb")
        
        # --- LLM: OpenAI API for fast cognify() ---
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            cognee.config.set_llm_provider("openai")
            cognee.config.set_llm_api_key(api_key)
            cognee.config.set_llm_model("openai/gpt-4o-mini")
            print("  Cognee LLM: OpenAI gpt-4o-mini")
        else:
            # Fallback: use our local endpoint
            cognee.config.set_llm_provider("openai")
            cognee.config.set_llm_endpoint("http://localhost:8888/v1")
            cognee.config.set_llm_api_key("local-no-key-needed")
            cognee.config.set_llm_model("openai/qwen3-4b")
            print("  Cognee LLM: local Qwen3-4B (slow)")
        
        print("  Cognee vector DB: LanceDB (local)")
        COGNEE_ENABLED = True
        print("  Cognee integration: ENABLED")
    except ImportError:
        print("  Cognee integration: DISABLED (cognee not installed)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Cognee integration: DISABLED ({e})")


@app.get("/api/cognee/status")
async def cognee_status():
    """Check if cognee is enabled and get collection info."""
    if not COGNEE_ENABLED:
        return {"enabled": False, "message": "Cognee not configured. Install with: uv add cognee"}
    
    try:
        # List cognee collections in Qdrant
        collections = qdrant.get_collections().collections
        cognee_collections = [c.name for c in collections if c.name.startswith(("DocumentChunk", "Entity", "EdgeType", "TextDocument", "TextSummary"))]
        
        return {
            "enabled": True,
            "collections": cognee_collections,
            "total_collections": len(cognee_collections),
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@app.post("/api/cognee/ingest")
async def cognee_ingest():
    """
    Run cognee.add() + cognee.cognify() on our feature data.
    This uses our local Qwen3-4B (via /v1/chat/completions) to extract
    entities, relationships, and summaries into a knowledge graph.
    Stores vectors in Qdrant and graph in cognee's graph DB (NetworkX/kuzu).
    """
    if not COGNEE_ENABLED:
        raise HTTPException(status_code=503, detail="Cognee not configured")
    
    t0 = time.time()
    data_dir = Path(__file__).parent / "data" / "features"
    
    try:
        import cognee
        
        # Clear old cognee data
        print("[cognee-ingest] Clearing previous cognee data...")
        try:
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)
        except Exception as e:
            print(f"  prune note: {e}")
        
        # Load feature JSON
        json_files = list(data_dir.glob("*.json"))
        if not json_files:
            raise HTTPException(status_code=404, detail=f"No feature JSON files in {data_dir}")
        
        all_docs = []
        for json_path in json_files:
            with open(json_path) as f:
                data = json.load(f)
            
            feature_name = data["metadata"]["dataset_name"]
            
            # Feature overview
            all_docs.append(f"Feature: {feature_name}\nDescription: {data['metadata']['description']}")
            
            # Jira tickets
            for ticket in data.get("jira_tickets", []):
                all_docs.append(
                    f"JIRA Ticket {ticket['id']}: {ticket['title']}\n"
                    f"Feature: {feature_name}\n"
                    f"Type: {ticket['type']}, Status: {ticket['status']}\n"
                    f"Description: {ticket['description']}\n"
                    f"Created: {ticket['created_at']}"
                )
            
            # Slack messages
            for convo in data.get("slack_conversations", []):
                channel = convo["channel"]
                for msg in convo.get("messages", []):
                    all_docs.append(
                        f"Slack message by {msg['user']} in {channel}\n"
                        f"Feature: {feature_name}\n"
                        f"Message: {msg['message']}\n"
                        f"Timestamp: {msg['timestamp']}"
                    )
        
        # Add to cognee as a single text block (cognee handles chunking)
        print(f"[cognee-ingest] Adding {len(all_docs)} documents to cognee...")
        combined = "\n\n---\n\n".join(all_docs)
        await cognee.add(combined, dataset_name="codelens_features")
        
        # Generate knowledge graph
        print("[cognee-ingest] Running cognee.cognify() with local Qwen3-4B...")
        print("[cognee-ingest] (This extracts entities, relationships, summaries via LLM)")
        await cognee.cognify()
        print("[cognee-ingest] cognify() complete!")
        
        elapsed = round((time.time() - t0) * 1000, 1)
        
        return {
            "status": "success",
            "documents_added": len(all_docs),
            "message": "Knowledge graph generated via cognee.cognify() using local Qwen3-4B",
            "time_ms": elapsed,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Cognee ingest error: {e}")


@app.get("/api/cognee/graph")
async def cognee_graph():
    """
    Get knowledge graph from cognee's kuzu graph DB (populated by cognify()).
    Uses cognee's internal graph engine within the running process.
    """
    if not COGNEE_ENABLED:
        raise HTTPException(status_code=503, detail="Cognee not configured")
    
    t0 = time.time()
    
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine
        
        graph_engine = await get_graph_engine()
        
        nodes = {}
        edges = []
        
        # --- Approach 1: Use cognee's vector search to find relevant IDs, then get the graph ---
        try:
            import cognee
            from cognee.api.v1.search import SearchType
            
            # Use GRAPH_COMPLETION which internally builds the CogneeGraph with 18+ nodes
            # and returns a text answer. We need the graph itself, not the answer.
            # So we do a single search and then read the graph engine directly
            results = await cognee.search(
                query_text="all entities features people tickets",
                query_type=SearchType.GRAPH_COMPLETION,
            )
            
            # Now the graph engine has been warmed up with the right user context
            # Access graph_engine again which should now have data
            graph_data = await graph_engine.get_graph_data()
            
            if isinstance(graph_data, tuple) and len(graph_data) == 2:
                nodes_data, edges_data = graph_data
                for item in (nodes_data or []):
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        nid, props = str(item[0]), item[1] if len(item) > 1 else {}
                        props = dict(props) if props else {}
                        label = props.get("name", props.get("text", nid))
                        nodes[nid] = {
                            "id": nid, "label": str(label)[:40],
                            "type": _classify_cognee_node(props),
                            "payload": {"text": str(props.get("text", props.get("description", "")))[:200]},
                        }
                for item in (edges_data or []):
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        src, tgt = str(item[0]), str(item[1])
                        rel_type = str(item[2]) if len(item) > 2 else "related_to"
                        props = dict(item[3]) if len(item) > 3 and item[3] else {}
                        label = props.get("relationship_name", rel_type)
                        edges.append({"source": src, "target": tgt, "label": str(label)})
            elif hasattr(graph_data, 'nodes') and callable(graph_data.nodes):
                for node_id, node_data in graph_data.nodes(data=True):
                    ndata = dict(node_data) if node_data else {}
                    label = ndata.get("name", ndata.get("text", str(node_id)[:30]))
                    nodes[str(node_id)] = {
                        "id": str(node_id), "label": str(label)[:40],
                        "type": _classify_cognee_node(ndata),
                        "payload": {"text": str(ndata.get("text", ""))[:200]},
                    }
                for source, target, edge_data in graph_data.edges(data=True):
                    edata = dict(edge_data) if edge_data else {}
                    label = edata.get("relationship_name", edata.get("type", "related_to"))
                    edges.append({"source": str(source), "target": str(target), "label": str(label)})
            
            print(f"  [cognee-graph] graph_data approach: {len(nodes)} nodes, {len(edges)} edges")
        except Exception as e1:
            print(f"  [cognee-graph] get_graph_data: {e1}")
            import traceback
            traceback.print_exc()
        
        # --- Approach 3: use cognee.search GRAPH_COMPLETION ---
        # GRAPH_COMPLETION returns a NetworkX subgraph from cognee's KG
        if not nodes:
            try:
                import cognee
                from cognee.api.v1.search import SearchType
                import networkx as nx
                
                queries = ["score boosting", "people", "jira tickets", "features"]
                
                def _extract_graph(obj):
                    """Recursively extract NetworkX graph from cognee SearchResult wrappers."""
                    # Unwrap SearchResult -> .search_result
                    if hasattr(obj, 'search_result'):
                        return _extract_graph(obj.search_result)
                    if isinstance(obj, (nx.Graph, nx.DiGraph)):
                        return obj
                    if isinstance(obj, (list, tuple)):
                        for item in obj:
                            g = _extract_graph(item)
                            if g is not None:
                                return g
                    return None
                
                for query in queries:
                    try:
                        results = await cognee.search(
                            query_text=query,
                            query_type=SearchType.GRAPH_COMPLETION,
                        )
                        for r in (results or []):
                            graph = _extract_graph(r)
                            if graph is not None:
                                for nid, ndata in graph.nodes(data=True):
                                    sid = str(nid)
                                    if sid not in nodes:
                                        nd = dict(ndata) if ndata else {}
                                        label = nd.get("name", nd.get("text", sid))
                                        nodes[sid] = {
                                            "id": sid,
                                            "label": str(label)[:40],
                                            "type": _classify_cognee_node(nd),
                                            "payload": {"text": str(nd.get("text", nd.get("description", "")))[:200]},
                                        }
                                for src, tgt, edata in graph.edges(data=True):
                                    ed = dict(edata) if edata else {}
                                    label = ed.get("relationship_name", ed.get("relationship_type", ed.get("type", "related_to")))
                                    edges.append({"source": str(src), "target": str(tgt), "label": str(label)})
                            elif isinstance(r, str):
                                nid = f"r-{len(nodes)}"
                                nodes[nid] = {"id": nid, "label": r[:40], "type": "entity", "payload": {"text": r[:200]}}
                            else:
                                # Unwrap SearchResult for non-graph results
                                actual = getattr(r, 'search_result', r)
                                if hasattr(actual, '__dict__'):
                                    rd = actual.__dict__
                                    nid = str(rd.get("id", f"r-{len(nodes)}"))
                                    label = rd.get("name", rd.get("text", str(actual)))
                                    if nid not in nodes:
                                        nodes[nid] = {"id": nid, "label": str(label)[:40], "type": _classify_cognee_node(rd), "payload": {"text": str(rd.get("text", ""))[:200]}}
                    except Exception as eq:
                        print(f"  [cognee-graph] query '{query}': {eq}")
                    
            except Exception as e3:
                print(f"  [cognee-graph] search fallback: {e3}")
        
        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "source": "cognee",
            "time_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting cognee graph: {e}")


def _classify_cognee_node(ndata: dict) -> str:
    """Classify a cognee graph node into a display type for D3."""
    text = str(ndata.get("name", ndata.get("text", ""))).lower()
    ntype = str(ndata.get("type", ndata.get("entity_type", ndata.get("node_type", "")))).lower()
    
    if any(kw in ntype for kw in ("person", "user", "author", "developer")):
        return "person"
    if any(kw in ntype for kw in ("ticket", "issue", "task", "bug", "story")):
        return "ticket"
    if any(kw in ntype for kw in ("feature", "product", "project", "system")):
        return "feature"
    if any(kw in ntype for kw in ("message", "conversation", "channel", "slack")):
        return "message"
    if any(kw in ntype for kw in ("concept", "technology", "tool")):
        return "concept"
    
    # Heuristic from name/text
    if any(name in text for name in ("dev_alex", "pm_sam", "qa_lee", "sarah", "alex", "sam", "lee")):
        return "person"
    if "jira" in text or text.startswith("jira-"):
        return "ticket"
    
    return "entity"


@app.get("/api/cognee/search")
async def cognee_search(
    q: str = Query(..., description="Search query"),
    search_type: str = Query("chunks", description="Search type: chunks, insights, summaries"),
):
    """
    Search using cognee's retrieval system.
    Uses cognee's vector search + knowledge graph for enhanced retrieval.
    """
    if not COGNEE_ENABLED:
        raise HTTPException(status_code=503, detail="Cognee not configured")
    
    t0 = time.time()
    
    try:
        import cognee
        from cognee.api.v1.search import SearchType
        
        # Map search type
        type_map = {
            "chunks": SearchType.CHUNKS,
            "insights": SearchType.INSIGHTS,
            "summaries": SearchType.SUMMARIES,
        }
        search_type_enum = type_map.get(search_type, SearchType.CHUNKS)
        
        # Search using cognee
        results = await cognee.search(
            query_text=q,
            query_type=search_type_enum,
        )
        
        # Format results
        formatted = []
        for i, result in enumerate(results[:20]):
            if isinstance(result, dict):
                formatted.append({
                    "rank": i + 1,
                    "text": result.get("text", result.get("content", str(result)[:500])),
                    "score": result.get("score", 0),
                    "metadata": {k: v for k, v in result.items() if k not in ("text", "content", "score", "vector")},
                })
            else:
                formatted.append({
                    "rank": i + 1,
                    "text": str(result)[:500],
                    "score": 0,
                })
        
        return {
            "query": q,
            "search_type": search_type,
            "results": formatted,
            "total": len(formatted),
            "source": "cognee",
            "time_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cognee search error: {e}")


@app.get("/api/debug/qdrant")
async def debug_qdrant():
    """
    Debug endpoint to show what's actually stored in Qdrant.
    Shows collections, sample points, and verifies embeddings are being used.
    """
    t0 = time.time()
    
    try:
        # Get all collections
        collections = qdrant.get_collections().collections
        collection_info = []
        
        for coll in collections:
            info = qdrant.get_collection(coll.name)
            sample_points = []
            
            # Get sample points from each collection
            try:
                points = qdrant.scroll(
                    collection_name=coll.name,
                    limit=3,
                    with_payload=True,
                    with_vectors=True,
                )[0]
                
                for p in points:
                    sample_points.append({
                        "id": str(p.id),
                        "payload_keys": list((p.payload or {}).keys()),
                        "payload_sample": {k: str(v)[:100] for k, v in list((p.payload or {}).items())[:5]},
                        "vector_dim": len(p.vector) if p.vector else 0,
                        "vector_sample": list(p.vector[:5]) if p.vector else [],
                    })
            except Exception as e:
                sample_points = [{"error": str(e)}]
            
            collection_info.append({
                "name": coll.name,
                "points_count": info.points_count,
                "vector_config": str(info.config.params.vectors) if info.config and info.config.params else "unknown",
                "sample_points": sample_points,
            })
        
        return {
            "total_collections": len(collections),
            "collections": collection_info,
            "time_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Frontend (served inline)
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the CodeLens interactive UI."""
    return get_frontend_html()


def get_frontend_html() -> str:
    """Return the complete frontend HTML/CSS/JS."""
    return r'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeLens</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #111113;
            --bg-secondary: #18181b;
            --bg-tertiary: #202024;
            --bg-hover: #27272a;
            --text-primary: #e4e4e7;
            --text-secondary: #a1a1aa;
            --text-muted: #71717a;
            --accent: #6366f1;
            --accent-hover: #818cf8;
            --accent-subtle: rgba(99, 102, 241, 0.12);
            --green: #34d399;
            --green-subtle: rgba(52, 211, 153, 0.12);
            --amber: #fbbf24;
            --amber-subtle: rgba(251, 191, 36, 0.12);
            --rose: #fb7185;
            --border: #27272a;
            --border-subtle: #1e1e21;
            --radius: 8px;
            --radius-lg: 12px;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            font-size: 14px;
            -webkit-font-smoothing: antialiased;
        }

        /* Header */
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 0.75rem 1.5rem;
        }
        .header-content {
            max-width: 1800px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-shrink: 0;
        }
        .logo h1 {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }
        .logo .dot {
            width: 6px; height: 6px;
            background: var(--accent);
            border-radius: 50%;
        }
        .search-container {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            flex: 1;
            max-width: 640px;
        }
        .search-input {
            flex: 1;
            padding: 0.5rem 0.75rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            font-size: 0.8125rem;
            transition: border-color 0.15s;
        }
        .search-input:focus {
            outline: none;
            border-color: var(--accent);
        }
        .search-input::placeholder { color: var(--text-muted); }

        .btn {
            padding: 0.5rem 0.875rem;
            border-radius: var(--radius);
            border: 1px solid transparent;
            font-size: 0.8125rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
            white-space: nowrap;
        }
        .btn-primary {
            background: var(--accent);
            color: white;
        }
        .btn-primary:hover { background: var(--accent-hover); }
        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            border-color: var(--border);
        }
        .btn-secondary:hover { background: var(--bg-hover); color: var(--text-primary); }
        
        /* Layout */
        .main-container {
            display: grid;
            grid-template-columns: 240px 1fr 300px;
            height: calc(100vh - 49px);
            max-width: 1800px;
            margin: 0 auto;
        }
        .panel {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            overflow-y: auto;
            overflow-x: hidden;
        }
        .panel:last-child { border-right: none; }
        .panel-header {
            padding: 0.625rem 0.875rem;
            border-bottom: 1px solid var(--border);
            background: var(--bg-secondary);
            position: sticky; top: 0; z-index: 10;
            display: flex; align-items: center; justify-content: space-between;
        }
        .panel-title {
            font-size: 0.6875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
        }

        /* Timeline */
        .timeline { padding: 0.5rem 0.75rem; position: relative; }
        .tl-item {
            display: flex; gap: 0.625rem;
            padding: 0.5rem 0.375rem;
            cursor: pointer;
            border-radius: var(--radius);
            transition: background 0.1s;
            position: relative;
        }
        .tl-item:hover { background: var(--bg-tertiary); }
        .tl-item.active { background: var(--accent-subtle); }
        .tl-pip {
            width: 8px; height: 8px;
            border-radius: 50%;
            margin-top: 4px;
            flex-shrink: 0;
        }
        .tl-pip.jira { background: var(--accent); }
        .tl-pip.slack { background: var(--green); }
        .tl-body { min-width: 0; }
        .tl-date { font-size: 0.6875rem; color: var(--text-muted); }
        .tl-label {
            display: flex; align-items: center; gap: 0.375rem;
            font-size: 0.75rem; font-weight: 500; color: var(--text-primary);
            margin-top: 0.125rem;
        }
        .tl-tag {
            font-size: 0.5625rem; padding: 1px 5px;
            border-radius: 3px; font-weight: 600; text-transform: uppercase;
        }
        .tl-tag.jira { background: var(--accent-subtle); color: var(--accent); }
        .tl-tag.slack { background: var(--green-subtle); color: var(--green); }
        .tl-text {
            font-size: 0.6875rem; color: var(--text-secondary);
            margin-top: 0.125rem; line-height: 1.4;
            overflow: hidden; text-overflow: ellipsis;
            display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
        }

        /* Timeline hover preview */
        .tl-preview {
            position: fixed;
            left: 248px;
            width: 300px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 0.875rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.03);
            z-index: 200;
            pointer-events: none;
            opacity: 0;
            transform: translateX(-4px);
            transition: opacity 0.15s ease, transform 0.15s ease;
        }
        .tl-preview.visible {
            opacity: 1;
            transform: translateX(0);
        }
        .tl-preview-tag {
            font-size: 0.5625rem; font-weight: 600; text-transform: uppercase;
            padding: 2px 6px; border-radius: 3px; display: inline-block; margin-bottom: 0.375rem;
        }
        .tl-preview-tag.jira { background: var(--accent-subtle); color: var(--accent); }
        .tl-preview-tag.slack { background: var(--green-subtle); color: var(--green); }
        .tl-preview-title {
            font-size: 0.8125rem; font-weight: 600; color: var(--text-primary);
            margin-bottom: 0.25rem; line-height: 1.3;
        }
        .tl-preview-meta {
            font-size: 0.6875rem; color: var(--text-muted); margin-bottom: 0.5rem;
        }
        .tl-preview-body {
            font-size: 0.75rem; color: var(--text-secondary); line-height: 1.5;
            max-height: 120px; overflow: hidden;
        }

        /* Center: Chat */
        .chat-panel {
            display: flex; flex-direction: column;
            height: 100%;
        }
        .chat-scroll {
            flex: 1; overflow-y: auto;
            padding: 1.25rem 1.5rem;
            display: flex; flex-direction: column;
        }
        .chat-scroll .welcome { flex: 1; }
        #chatMessages { flex: 1; }
        .chat-input-area {
            padding: 0.625rem 1.5rem;
            border-top: 1px solid var(--border);
            background: var(--bg-secondary);
            flex-shrink: 0;
        }
        .chat-input-row {
            display: flex; gap: 0.5rem;
        }
        .chat-input {
            flex: 1;
            padding: 0.5rem 0.75rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            font-size: 0.8125rem;
        }
        .chat-input:focus { outline: none; border-color: var(--accent); }
        .chat-suggestions {
            display: flex; flex-wrap: wrap; gap: 0.375rem;
            margin-bottom: 0.5rem;
        }
        .chip {
            padding: 0.25rem 0.625rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 100px;
            font-size: 0.6875rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.1s;
        }
        .chip:hover { border-color: var(--accent); color: var(--text-primary); background: var(--accent-subtle); }

        /* Welcome */
        .welcome {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            height: 100%; text-align: center; color: var(--text-muted);
            padding: 2rem;
        }
        .welcome h2 { color: var(--text-primary); font-size: 1.125rem; font-weight: 600; margin-bottom: 0.375rem; }
        .welcome p { font-size: 0.8125rem; max-width: 360px; line-height: 1.5; margin-bottom: 1.25rem; }
        .example-queries { display: flex; flex-wrap: wrap; gap: 0.375rem; justify-content: center; }

        /* Chat Messages */
        .msg { margin-bottom: 1rem; }
        .msg-q {
            background: var(--accent-subtle);
            border-left: 2px solid var(--accent);
            border-radius: 0 var(--radius) var(--radius) 0;
            padding: 0.5rem 0.75rem;
            font-size: 0.8125rem;
            color: var(--text-primary);
        }
        .msg-a {
            padding: 0.75rem 0;
        }
        /* Markdown rendered inside answers */
        .msg-a h3 {
            font-size: 0.8125rem; font-weight: 600; color: var(--accent);
            margin: 0.875rem 0 0.375rem 0; padding-bottom: 0.25rem;
            border-bottom: 1px solid var(--border);
        }
        .msg-a h3:first-child { margin-top: 0; }
        .msg-a h4 {
            font-size: 0.75rem; font-weight: 600; color: var(--text-primary);
            margin: 0.625rem 0 0.25rem 0;
        }
        .msg-a p {
            font-size: 0.8125rem; line-height: 1.55; color: var(--text-secondary);
            margin-bottom: 0.375rem;
        }
        .msg-a ul, .msg-a ol {
            margin: 0.25rem 0 0.5rem 1.125rem;
            font-size: 0.8125rem; color: var(--text-secondary);
        }
        .msg-a li { line-height: 1.55; margin-bottom: 0.2rem; }
        .msg-a li::marker { color: var(--accent); }
        .msg-a strong { color: var(--text-primary); }
        .msg-a code {
            background: var(--bg-tertiary); padding: 1px 4px; border-radius: 3px;
            font-size: 0.75rem;
        }
        .msg-meta {
            font-size: 0.6875rem; color: var(--text-muted);
            margin-top: 0.375rem;
            display: flex; gap: 0.75rem;
        }

        /* Loading */
        .loading {
            display: flex; align-items: center; justify-content: center;
            padding: 1.5rem; color: var(--text-muted); font-size: 0.8125rem;
        }
        .spin {
            width: 16px; height: 16px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
            margin-right: 0.5rem;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Graph Panel */
        .graph-wrap { position: relative; height: calc(100% - 37px); overflow: hidden; }
        .graph-bar {
            display: flex; gap: 0.25rem; padding: 0.375rem 0.5rem;
            position: absolute; top: 0; left: 0; z-index: 20;
        }
        .graph-bar button, .graph-ctrl button {
            padding: 0.2rem 0.5rem; border-radius: 4px;
            border: 1px solid var(--border); background: var(--bg-tertiary);
            color: var(--text-muted); cursor: pointer; font-size: 0.625rem; font-weight: 500;
        }
        .graph-bar button.active { background: var(--accent); color: white; border-color: var(--accent); }
        .graph-bar button:hover:not(.active) { background: var(--bg-hover); color: var(--text-primary); }
        .graph-bar .build-btn { background: var(--green); color: #111; border-color: var(--green); font-weight: 600; }
        .graph-ctrl {
            display: flex; gap: 0.2rem; position: absolute; top: 0.375rem; right: 0.5rem; z-index: 20;
        }
        .graph-ctrl button { width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; }
        .graph-ctrl button:hover { background: var(--bg-hover); color: var(--text-primary); }
        #graph-svg { width: 100%; height: 100%; cursor: grab; }
        #graph-svg:active { cursor: grabbing; }

        .node { cursor: pointer; }
        .node:hover { filter: brightness(1.2); }
        .node circle { fill: var(--amber); stroke: var(--bg-primary); stroke-width: 2px; }
        .node rect { fill: var(--accent); stroke: var(--bg-primary); stroke-width: 2px; rx: 3; }
        .node polygon { fill: #60a5fa; stroke: var(--bg-primary); stroke-width: 2px; }
        .node.type-concept circle { fill: var(--green); }
        .node.type-entity circle { fill: #64748b; }
        .node.highlighted circle, .node.highlighted rect, .node.highlighted polygon {
            stroke: var(--green); stroke-width: 3px;
        }
        .node-label {
            font-size: 9px; fill: var(--text-secondary); pointer-events: none; font-weight: 500;
            text-shadow: 0 0 3px var(--bg-primary), 0 0 3px var(--bg-primary);
        }
        .link { stroke: var(--border); stroke-opacity: 0.4; stroke-width: 1px; }
        .link.highlighted { stroke: var(--green); stroke-opacity: 1; stroke-width: 2px; }
        .link-label { font-size: 7px; fill: var(--text-muted); fill-opacity: 0.5; }

        .graph-legend {
            position: absolute; bottom: 0.5rem; left: 0.5rem;
            background: var(--bg-secondary); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 0.5rem; font-size: 0.625rem;
        }
        .lg-item { display: flex; align-items: center; gap: 0.375rem; margin-bottom: 0.125rem; color: var(--text-muted); }
        .lg-dot { width: 7px; height: 7px; border-radius: 50%; }

        .node-tooltip {
            position: absolute; background: var(--bg-secondary);
            border: 1px solid var(--accent); border-radius: var(--radius);
            padding: 0.5rem; font-size: 0.75rem; max-width: 240px;
            pointer-events: none; z-index: 100;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4); display: none;
        }
        .node-tooltip.visible { display: block; }
        .node-tooltip h4 { margin-bottom: 0.125rem; color: var(--accent); font-size: 0.75rem; }
        .node-tooltip p { color: var(--text-secondary); margin: 0; line-height: 1.3; }
        .tooltip-type { font-size: 0.5625rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); margin-bottom: 0.125rem; }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div class="logo"><div class="dot"></div><h1>CodeLens</h1></div>
            <div class="search-container">
                <input type="text" class="search-input" id="searchInput"
                    placeholder="Ask about any feature..." />
                <button class="btn btn-primary" onclick="askQuestion()">Ask</button>
                <button class="btn btn-secondary" onclick="searchDocs()">Search</button>
            </div>
        </div>
    </header>

    <main class="main-container">
        <!-- Timeline -->
        <div class="panel" id="timelinePanel">
            <div class="panel-header"><span class="panel-title">Timeline</span></div>
            <div class="timeline" id="timeline">
                <div class="loading" id="timelineLoading" style="display:none;"><div class="spin"></div>Loading...</div>
            </div>
        </div>

        <!-- Chat -->
        <div class="panel" style="border-right:1px solid var(--border);">
            <div class="chat-panel">
                <div class="chat-scroll" id="chatScroll">
                    <div class="welcome" id="welcomeState">
                        <h2>Explore Feature History</h2>
                        <p>Ask questions about your features to understand the story behind the code.</p>
                        <div class="example-queries">
                            <span class="chip" onclick="setQuery('Why was score boosting added?')">Why was score boosting added?</span>
                            <span class="chip" onclick="setQuery('Who worked on hybrid queries?')">Who worked on hybrid queries?</span>
                            <span class="chip" onclick="setQuery('What user feedback led to this?')">What user feedback led to this?</span>
                        </div>
                    </div>
                    <div id="chatMessages"></div>
                    <div class="loading" id="answerLoading" style="display:none;"><div class="spin"></div>Analyzing...</div>
                </div>
                <div class="chat-input-area" id="chatInputArea" style="display:none;">
                    <div class="chat-suggestions" id="followupSuggestions"></div>
                    <div class="chat-input-row">
                        <input type="text" class="chat-input" id="followupInput" placeholder="Ask a follow-up..." />
                        <button class="btn btn-primary" onclick="askFollowup()">Send</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Graph -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">Knowledge Graph</span>
                <span style="font-size:0.5625rem;color:var(--text-muted);">cognee</span>
            </div>
            <div class="graph-wrap">
                <div class="graph-bar">
                    <button id="btnCogneeIngest" class="build-btn" onclick="runCogneeIngest()">Build KG</button>
                </div>
                <div class="graph-ctrl">
                    <button onclick="zoomIn()">+</button>
                    <button onclick="zoomOut()">&minus;</button>
                    <button onclick="resetZoom()">&#8634;</button>
                </div>
                <svg id="graph-svg"></svg>
                <div class="node-tooltip" id="nodeTooltip">
                    <div class="tooltip-type" id="tooltipType"></div>
                    <h4 id="tooltipTitle"></h4>
                    <p id="tooltipContent"></p>
                </div>
                <div class="graph-legend">
                    <div class="lg-item"><div class="lg-dot" style="background:var(--amber)"></div>People</div>
                    <div class="lg-item"><div class="lg-dot" style="background:var(--accent)"></div>Tickets</div>
                    <div class="lg-item"><div class="lg-dot" style="background:#60a5fa"></div>Features</div>
                    <div class="lg-item"><div class="lg-dot" style="background:var(--green)"></div>Concepts</div>
                </div>
            </div>
        </div>
    </main>
    <div class="tl-preview" id="tlPreview">
        <div class="tl-preview-tag" id="tlPreviewTag"></div>
        <div class="tl-preview-title" id="tlPreviewTitle"></div>
        <div class="tl-preview-meta" id="tlPreviewMeta"></div>
        <div class="tl-preview-body" id="tlPreviewBody"></div>
    </div>
    <script>
        let currentFeatureId = null;
        let graphSimulation = null;
        let graphZoom = null;
        let graphSvgGroup = null;
        let lastQuestion = '';
        let lastAnswer = '';
        let graphSource = 'cognee';
        let turnCount = 0;
        let timelineData = [];

        marked.setOptions({ breaks: true, gfm: true });

        document.addEventListener('DOMContentLoaded', () => {
            loadTimeline();
            loadGraph();
            document.getElementById('searchInput').addEventListener('keypress', e => { if (e.key === 'Enter') askQuestion(); });
            document.getElementById('followupInput').addEventListener('keypress', e => { if (e.key === 'Enter') askFollowup(); });
        });

        function fmt(iso) {
            if (!iso) return '';
            try { return new Date(iso).toLocaleDateString('en-US', {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}); }
            catch { return iso; }
        }

        function md(text) { return marked.parse(text || ''); }

        function scrollChat() {
            const el = document.getElementById('chatScroll');
            el.scrollTop = el.scrollHeight;
        }

        function appendMsg(role, html, meta) {
            const c = document.getElementById('chatMessages');
            const d = document.createElement('div');
            d.className = 'msg';
            if (role === 'user') {
                d.innerHTML = '<div class="msg-q">' + html + '</div>';
            } else {
                d.innerHTML = '<div class="msg-a">' + html + '</div>' + (meta ? '<div class="msg-meta">' + meta + '</div>' : '');
            }
            c.appendChild(d);
            scrollChat();
        }

        function setQuery(q) { document.getElementById('searchInput').value = q; askQuestion(); }

        async function runCogneeIngest() {
            const btn = document.getElementById('btnCogneeIngest');
            const orig = btn.textContent;
            btn.textContent = 'Building...'; btn.disabled = true; btn.style.opacity = '0.5';
            try {
                const r = await fetch('/api/cognee/ingest', {method:'POST'});
                if (r.ok) { btn.textContent = 'Done'; loadGraph(lastQuestion || null); }
                else { btn.textContent = 'Error'; }
            } catch { btn.textContent = 'Error'; }
            finally { setTimeout(() => { btn.textContent = orig; btn.disabled = false; btn.style.opacity = '1'; }, 3000); }
        }

        async function askQuestion() {
            const q = document.getElementById('searchInput').value.trim();
            if (!q) return;
            lastQuestion = q;
            turnCount = 0;

            // Reset chat
            document.getElementById('welcomeState').style.display = 'none';
            document.getElementById('chatMessages').innerHTML = '';
            document.getElementById('chatInputArea').style.display = 'block';
            document.getElementById('answerLoading').style.display = 'flex';

            appendMsg('user', q);

            try {
                const p = new URLSearchParams({q});
                if (currentFeatureId) p.append('feature_id', currentFeatureId);
                const res = await fetch('/api/ask?' + p);
                const data = await res.json();
                lastAnswer = data.answer;
                turnCount++;

                document.getElementById('answerLoading').style.display = 'none';
                const meta = `<span>${data.model}</span><span>${data.source_count} sources</span><span>${Math.round(data.retrieval_ms + data.llm_ms)}ms</span>`;
                appendMsg('assistant', md(data.answer), meta);
                makeSuggestions(q, data.sources);
                loadGraph(q);
            } catch (err) {
                document.getElementById('answerLoading').style.display = 'none';
                appendMsg('assistant', '<p style="color:var(--rose)">Error: ' + err.message + '</p>');
            }
        }

        function makeSuggestions(q, sources) {
            const s = [];
            const people = [...new Set((sources||[]).filter(x=>x.user).map(x=>x.user))];
            const tickets = [...new Set((sources||[]).filter(x=>x.source_type==='jira').map(x=>x.source_id))];
            if (people.length) s.push('What else did ' + people[0] + ' contribute?');
            if (tickets.length) s.push('What are the acceptance criteria for ' + tickets[0] + '?');
            s.push('What were the technical challenges?');
            s.push('What are the next steps?');
            document.getElementById('followupSuggestions').innerHTML = s.slice(0,4).map(x =>
                '<span class="chip" onclick="askFollowupSuggestion(\'' + x.replace(/'/g,"\\'") + '\')">' + x + '</span>'
            ).join('');
        }

        function askFollowupSuggestion(q) { document.getElementById('followupInput').value = q; askFollowup(); }

        async function askFollowup() {
            const q = document.getElementById('followupInput').value.trim();
            if (!q) return;
            document.getElementById('followupInput').value = '';

            const ctx = 'Previous question: "' + lastQuestion + '"\nPrevious answer summary: ' + (lastAnswer||'').substring(0,300) + '...\n\nFollow-up question: ' + q;

            appendMsg('user', q);
            document.getElementById('answerLoading').style.display = 'flex';

            try {
                const p = new URLSearchParams({q: ctx});
                if (currentFeatureId) p.append('feature_id', currentFeatureId);
                const res = await fetch('/api/ask?' + p);
                const data = await res.json();
                lastQuestion = q;
                lastAnswer = data.answer;
                turnCount++;

                document.getElementById('answerLoading').style.display = 'none';
                const meta = '<span>' + data.model + '</span><span>' + data.source_count + ' sources</span><span>Turn ' + turnCount + '</span>';
                appendMsg('assistant', md(data.answer), meta);
                makeSuggestions(q, data.sources);
            } catch (err) {
                document.getElementById('answerLoading').style.display = 'none';
                appendMsg('assistant', '<p style="color:var(--rose)">Error: ' + err.message + '</p>');
            }
        }

        async function searchDocs() {
            const q = document.getElementById('searchInput').value.trim();
            if (!q) return;
            document.getElementById('welcomeState').style.display = 'none';
            document.getElementById('chatMessages').innerHTML = '';
            document.getElementById('chatInputArea').style.display = 'none';
            document.getElementById('answerLoading').style.display = 'flex';

            try {
                const res = await fetch('/api/search?' + new URLSearchParams({q, limit:10}));
                const data = await res.json();
                document.getElementById('answerLoading').style.display = 'none';

                let html = '<h3>Search Results</h3><p>Found ' + data.total + ' documents (' + data.time_ms + 'ms)</p>';
                data.results.forEach(r => {
                    const tag = r.source_type === 'jira' ? '<span class="tl-tag jira" style="display:inline-block;margin-right:6px">JIRA</span>' : '<span class="tl-tag slack" style="display:inline-block;margin-right:6px">SLACK</span>';
                    html += '<div style="padding:0.5rem 0;border-bottom:1px solid var(--border)">' + tag + '<strong>' + (r.title||r.source_id) + '</strong><br><span style="font-size:0.75rem;color:var(--text-muted)">' + (r.user||'') + ' &middot; ' + fmt(r.timestamp) + ' &middot; score: ' + (r.score||0).toFixed(3) + '</span><br><span style="font-size:0.8125rem;color:var(--text-secondary)">' + (r.text||'').substring(0,200) + '</span></div>';
                });
                appendMsg('assistant', html);
            } catch (err) {
                document.getElementById('answerLoading').style.display = 'none';
                appendMsg('assistant', '<p style="color:var(--rose)">Error: ' + err.message + '</p>');
            }
        }

        async function loadTimeline() {
            const c = document.getElementById('timeline');
            document.getElementById('timelineLoading').style.display = 'flex';
            try {
                const p = new URLSearchParams({limit:50});
                if (currentFeatureId) p.append('feature_id', currentFeatureId);
                const res = await fetch('/api/timeline?' + p);
                const data = await res.json();
                timelineData = data.events || [];
                document.getElementById('timelineLoading').style.display = 'none';
                if (!timelineData.length) { c.innerHTML = '<div class="loading">No events yet.</div>'; return; }
                c.innerHTML = timelineData.map((e, i) =>
                    '<div class="tl-item" data-id="' + e.id + '" data-idx="' + i + '">' +
                    '<div class="tl-pip ' + e.type + '"></div>' +
                    '<div class="tl-body">' +
                    '<div class="tl-date">' + fmt(e.timestamp) + '</div>' +
                    '<div class="tl-label">' + (e.user||e.source_id) + ' <span class="tl-tag ' + e.type + '">' + e.type + '</span></div>' +
                    '<div class="tl-text">' + (e.title||e.text) + '</div>' +
                    '</div></div>'
                ).join('');
                // Attach hover preview listeners
                c.querySelectorAll('.tl-item').forEach(el => {
                    el.addEventListener('mouseenter', showTlPreview);
                    el.addEventListener('mousemove', moveTlPreview);
                    el.addEventListener('mouseleave', hideTlPreview);
                });
            } catch (err) {
                document.getElementById('timelineLoading').style.display = 'none';
                c.innerHTML = '<div class="loading">Error loading timeline.</div>';
            }
        }

        function showTlPreview(ev) {
            const idx = parseInt(ev.currentTarget.dataset.idx);
            const e = timelineData[idx];
            if (!e) return;
            const preview = document.getElementById('tlPreview');
            const tag = document.getElementById('tlPreviewTag');
            const title = document.getElementById('tlPreviewTitle');
            const meta = document.getElementById('tlPreviewMeta');
            const body = document.getElementById('tlPreviewBody');

            tag.className = 'tl-preview-tag ' + e.type;
            tag.textContent = e.type === 'jira' ? (e.source_id || 'JIRA') : '#' + (e.channel || 'slack');
            title.textContent = e.title || e.text || '';
            meta.textContent = (e.user ? e.user + ' · ' : '') + fmt(e.timestamp) + (e.status ? ' · ' + e.status : '');
            body.textContent = e.text || e.title || '';

            const rect = ev.currentTarget.getBoundingClientRect();
            const y = Math.min(rect.top, window.innerHeight - 220);
            preview.style.top = Math.max(8, y) + 'px';
            preview.classList.add('visible');
        }

        function moveTlPreview(ev) {
            const rect = ev.currentTarget.getBoundingClientRect();
            const preview = document.getElementById('tlPreview');
            const y = Math.min(rect.top, window.innerHeight - 220);
            preview.style.top = Math.max(8, y) + 'px';
        }

        function hideTlPreview() {
            document.getElementById('tlPreview').classList.remove('visible');
        }

        // --- Graph ---
        function zoomIn() { if (graphZoom) d3.select('#graph-svg').transition().duration(200).call(graphZoom.scaleBy, 1.3); }
        function zoomOut() { if (graphZoom) d3.select('#graph-svg').transition().duration(200).call(graphZoom.scaleBy, 0.7); }
        function resetZoom() { if (graphZoom) d3.select('#graph-svg').transition().duration(200).call(graphZoom.transform, d3.zoomIdentity); }

        function showTooltip(ev, d) {
            const t = document.getElementById('nodeTooltip');
            document.getElementById('tooltipType').textContent = d.type;
            document.getElementById('tooltipTitle').textContent = d.label;
            document.getElementById('tooltipContent').textContent = d.payload?.text || '';
            t.style.left = Math.min(ev.pageX+10, innerWidth-260) + 'px';
            t.style.top = Math.max(ev.pageY-10, 10) + 'px';
            t.classList.add('visible');
        }
        function hideTooltip() { document.getElementById('nodeTooltip').classList.remove('visible'); }

        async function loadGraph(query) {
            const svg = d3.select('#graph-svg');
            const {width, height} = svg.node().getBoundingClientRect();
            svg.selectAll('*').remove();
            try {
                const data = await (await fetch('/api/cognee/graph')).json();
                if (!data.nodes?.length) {
                    svg.append('text').attr('x',width/2).attr('y',height/2).attr('text-anchor','middle').attr('fill','#52525b').attr('font-size','12px')
                       .text('Click "Build KG" to generate graph');
                    return;
                }
                const n = data.nodes.length;
                graphZoom = d3.zoom().scaleExtent([0.1,5]).on('zoom', ev => graphSvgGroup.attr('transform', ev.transform));
                svg.call(graphZoom);
                svg.append('defs').append('marker').attr('id','ah').attr('viewBox','-0 -5 10 10').attr('refX',22).attr('refY',0).attr('orient','auto').attr('markerWidth',5).attr('markerHeight',5).append('path').attr('d','M0,-5L10,0L0,5').attr('fill','#3f3f46');
                graphSvgGroup = svg.append('g');
                const sim = d3.forceSimulation(data.nodes)
                    .force('link', d3.forceLink(data.edges).id(d=>d.id).distance(n>15?100:80).strength(0.4))
                    .force('charge', d3.forceManyBody().strength(n>15?-400:-300))
                    .force('center', d3.forceCenter(width/2, height/2))
                    .force('collision', d3.forceCollide().radius(n>15?45:35).strength(0.8));
                const link = graphSvgGroup.append('g').selectAll('line').data(data.edges).enter().append('line').attr('class','link').attr('marker-end','url(#ah)');
                const lbl = graphSvgGroup.append('g').selectAll('text').data(data.edges).enter().append('text').attr('class','link-label').attr('text-anchor','middle').attr('dy',-3).text(d=>d.label);
                const node = graphSvgGroup.append('g').selectAll('.node').data(data.nodes).enter().append('g')
                    .attr('class', d => 'node type-' + d.type + (d.highlighted?' highlighted':''))
                    .on('mouseover', (ev,d) => { showTooltip(ev,d); link.classed('highlighted', l=>l.source.id===d.id||l.target.id===d.id); })
                    .on('mouseout', () => { hideTooltip(); link.classed('highlighted',false); })
                    .on('click', (ev,d) => { document.getElementById('searchInput').value = d.label; })
                    .call(d3.drag().on('start',(ev)=>{if(!ev.active)sim.alphaTarget(0.3).restart();ev.subject.fx=ev.subject.x;ev.subject.fy=ev.subject.y;}).on('drag',(ev)=>{ev.subject.fx=ev.x;ev.subject.fy=ev.y;}).on('end',(ev)=>{if(!ev.active)sim.alphaTarget(0);}));
                node.each(function(d) {
                    const el = d3.select(this);
                    if (d.type==='person') el.append('circle').attr('r',14);
                    else if (d.type==='ticket') el.append('rect').attr('width',24).attr('height',24).attr('x',-12).attr('y',-12).attr('rx',3);
                    else if (d.type==='feature') el.append('polygon').attr('points','0,-16 14,8 -14,8');
                    else el.append('circle').attr('r', d.type==='concept'?12:10);
                });
                node.append('text').attr('class','node-label').attr('dy', d=>d.type==='feature'?26:d.type==='person'||d.type==='ticket'?24:22).attr('text-anchor','middle').text(d=>d.label.length>12?d.label.substring(0,12)+'...':d.label);
                sim.on('tick', () => {
                    data.nodes.forEach(d => { d.x=Math.max(30,Math.min(width-30,d.x)); d.y=Math.max(30,Math.min(height-30,d.y)); });
                    link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
                    lbl.attr('x',d=>(d.source.x+d.target.x)/2).attr('y',d=>(d.source.y+d.target.y)/2);
                    node.attr('transform',d=>'translate('+d.x+','+d.y+')');
                });
                sim.on('end', () => {
                    let x0=Infinity,y0=Infinity,x1=-Infinity,y1=-Infinity;
                    data.nodes.forEach(d=>{if(d.x<x0)x0=d.x;if(d.y<y0)y0=d.y;if(d.x>x1)x1=d.x;if(d.y>y1)y1=d.y;});
                    const sc = Math.min(width/(x1-x0+80), height/(y1-y0+80), 1.5);
                    svg.transition().duration(400).call(graphZoom.transform, d3.zoomIdentity.translate(width/2-(x0+x1)/2*sc, height/2-(y0+y1)/2*sc).scale(sc));
                });
                graphSimulation = sim;
            } catch (err) {
                svg.append('text').attr('x',width/2).attr('y',height/2).attr('text-anchor','middle').attr('fill','#52525b').attr('font-size','11px').text('Error: '+err.message);
            }
        }
    </script>
</body>
</html>'''


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
