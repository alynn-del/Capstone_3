"""
lambda_handler.py -- wraps the orchestrator in the format Lambda expects.
"""
import json
from orchestrator import build_orchestrator
from tokenomics_logger import run_traced_query, calculate_full_query_cost

orchestrator = build_orchestrator()


def handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        question = body.get("question", "").strip()

        if not question:
            return {"statusCode": 400, "body": json.dumps({"error": "question is required"})}

        result = run_traced_query(orchestrator, question)

        sql_validation = None
        if result.get("sql_result"):
            sql_validation = result["sql_result"].get("metadata", {}).get("validation")

        cost_breakdown = calculate_full_query_cost(
            sql_tokens=result.get("sql_result", {}).get("metadata", {}).get("token_usage") if result.get("sql_result") else None,
            retrieval_tokens=result.get("retrieval_result", {}).get("metadata", {}).get("token_usage") if result.get("retrieval_result") else None,
            synthesis_tokens=result.get("token_usage"),
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "answer": result["final_answer"],
                "route": result["route"],
                "sql_validation": sql_validation,
                "token_usage": result.get("token_usage", {}),
                "cost": cost_breakdown,
            })
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
