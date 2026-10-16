"""
agents/retrieval_agent.py -- validated retrieval agent, wrapped to the
standard {success, answer, sources, metadata} interface using onnx embeddings.
"""
import os
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from onnx_embed import get_embedding
from bedrock_client import invoke_claude

REGION = "us-east-1"

#-------------Define OpenSearch Credentials------------------
def get_opensearch_client():
    host = os.environ["OPENSEARCH_ENDPOINT"]
    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(credentials.access_key, credentials.secret_key,
                     REGION, "es", session_token=credentials.token)
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth, use_ssl=True, verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

#-----------Retrieving Top K (default 5) semantic similarity embedded chunks from OpenSearch------
def retrieve(question: str, top_k: int = 5) -> list[dict]:
    query_vector = get_embedding(question)
    client = get_opensearch_client()
    response = client.search(
        index="documents",
        body={"size": top_k, "query": {"knn": {"embedding": {"vector": query_vector, "k": top_k}}}}
    )
    return [
        {"content": hit["_source"]["content"], "source": hit["_source"]["source"], "score": hit["_score"]}
        for hit in response["hits"]["hits"]
    ]

#----------Generating NL Answer from context from OpenSearch using Claude------------
def generate_cited_answer(question: str, chunks: list[dict]) -> tuple[str, dict]:
    context = "\n\n".join([f"[Source {i+1}: {c['source']}]\n{c['content']}" for i, c in enumerate(chunks)])
    prompt = (
        "Answer the question using ONLY the context provided below. "
        "If the answer is not in the context, say 'I cannot find this information in the provided documents.' "
        "Always cite the source number(s) you used. "
        "Respond in plain text only. Do not use markdown, bullet points, headers, or bold formatting.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    result = invoke_claude(prompt, max_tokens=500)
    return result["text"], {"input": result["input_tokens"], "output": result["output_tokens"]}

#----------Double Checking Contextual Grounding of Answer using Sources---------
def check_grounding(answer: str, chunks: list[dict]) -> bool:
    import re
    cited_sources = re.findall(r"Source (\d+)", answer)
    if not cited_sources:
        return "cannot find" in answer.lower()
    return all(1 <= int(s) <= len(chunks) for s in cited_sources)

#-----------Full Compile of Retrieval Agent for Running-----------
def run_retrieval_agent(question: str) -> dict:
    chunks = retrieve(question)
    if not chunks:
        return {
            "success": False,
            "answer": "No relevant documents found.",
            "sources": [],
            "metadata": {"chunk_count": 0, "grounded": False, "token_usage": {"input": 0, "output": 0}},
        }

    answer, usage = generate_cited_answer(question, chunks)
    return {
        "success": True,
        "answer": answer,
        "sources": [c["source"] for c in chunks],
        "metadata": {
            "chunk_count": len(chunks),
            "grounded": check_grounding(answer, chunks),
            "token_usage": usage,
        },
    }