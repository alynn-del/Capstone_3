"""
agents/sql_agent.py -- validated SQL Query Agent, wrapped to the standard
{success, answer, sources, metadata} interface.
"""
import os
import re
import psycopg2
from dotenv import load_dotenv
from bedrock_client import invoke_claude
load_dotenv()

SCHEMA_CONTEXT = """
Tables available:
- clients(id, name, industry, region, contract_value, created_date)
- projects(id, client_id, name, status, start_date, end_date, budget)
- consultants(id, name, grade, specialisation, project_id, billable_hours)
- invoices(id, project_id, amount, status, due_date, paid_date)

All monetary values are in USD. Dates are in ISO 8601 format.
status in projects is one of: 'active', 'completed', 'on-hold'.
"""

#---------------Helpers for Validate SQL--------------------------
BLOCKED = ["CREATE", "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
           "REINDEX", "RENAME", "MERGE", "GRANT", "REVOKE", "COMMIT", "ROLLBACK",
           "INTO", "EXEC", "EXECUTE", "CALL", "COPY"]

# REPLACE is blocked as a statement (MySQL REPLACE INTO) but allowed as the REPLACE() function which doesn't mutate db just output 
BLOCKED_PATTERN = re.compile(
    r"\b(" + "|".join(BLOCKED) + r")\b|\bREPLACE\b(?!\s*\()", re.IGNORECASE
)
# Matches string literals and comments left to right, so whichever comes first is handled correctly
LITERALS_AND_COMMENTS = re.compile(r"'(?:[^']|'')*'|--[^\n]*|/\*.*?\*/", re.DOTALL) 
FENCE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)

# replaces actual db values with "" for scanning purposes to avoid catching them as SQl commands
def _mask(match: re.Match) -> str:
    return "''" if match.group(0).startswith("'") else " "
#--------------------------------------------------------------------

#----------Validating SQL query before calling database-------------
def validate_sql(query: str) -> dict:
    sql = FENCE.sub("", (query or "").strip()).strip()
    if not sql:
        return {"valid": False, "reason": "SQL query is empty."}

    # Check a copy with strings and comments blanked out, so their contents can't trigger rules
    scan = LITERALS_AND_COMMENTS.sub(_mask, sql).strip()
    body = scan[:-1] if scan.endswith(";") else scan

    if ";" in body:
        return {"valid": False, "reason": "Multiple SQL statements are not permitted."}

    match = BLOCKED_PATTERN.search(body)
    if match:
        return {"valid": False, "reason": f"Blocked keyword: {match.group(0).upper()}"}

    if not body.upper().startswith(("SELECT", "WITH")):
        return {"valid": False, "reason": "SQL queries must start with SELECT or WITH."}

    if not re.search(r"\bFROM\b", body, flags=re.IGNORECASE):
        return {"valid": False, "reason": "Read-only queries must include a FROM clause."}

    return {"valid": True, "reason": "Read-only SQL query is valid.", "sql": sql}

#--------Generate SQL Query invoking Claude----------------
def generate_sql(question: str) -> str:
    prompt = (
        f"{SCHEMA_CONTEXT}\n\n"
        f"Generate a SQL query to answer: {question}\n\n"
        "Return ONLY the SQL query. No explanation, no markdown, no backticks."
    )
    result = invoke_claude(prompt, max_tokens=512)
    sql = result["text"].strip()
    return sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()

#-------------------Execute Query only if passed through Validation Layer------------------------
def execute_query_safely(sql: str) -> dict:
    validation = validate_sql(sql)
    if not validation["valid"]:
        return {"success": False, "error": validation["reason"], "validation": "BLOCKED", "sql": sql}

    try:
        conn = psycopg2.connect(
            host=os.getenv("RDS_HOST"), port=int(os.getenv("RDS_PORT")),
            dbname=os.getenv("RDS_DATABASE"), user=os.getenv("RDS_USER"),
            password=os.getenv("RDS_PASSWORD")
        )
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        conn.close()
        return {"success": True, "columns": cols, "rows": rows, "validation": "PASSED", "sql": sql}
    except Exception as e:
        return {"success": False, "error": str(e), "validation": "EXECUTION_ERROR", "sql": sql}

#-----------Full Compile of SQL Agent for Running--------
def run_sql_agent(question: str) -> dict:
    sql = generate_sql(question)
    result = execute_query_safely(sql)

    if not result["success"]:
        return {
            "success": False,
            "answer": f"Query blocked or failed: {result['error']}",
            "sources": [f"SQL: {sql}"],
            "metadata": {"validation": result["validation"], "token_usage": {"input": 0, "output": 0}},
        }

    prompt = (
        f"Question: {question}\nColumns: {result['columns']}\n"
        f"Data: {result['rows'][:20]}\n\n"
        "Answer concisely in plain text. Do not use markdown, bullet points, headers, or bold formatting."
    )
    interpretation = invoke_claude(prompt, max_tokens=500)

    return {
        "success": result["validation"] == "PASSED",
        "answer": interpretation["text"],
        "sources": [f"SQL: {sql}"],
        "metadata": {
            "validation": result["validation"],
            "token_usage": {"input": interpretation["input_tokens"], "output": interpretation["output_tokens"]},
        },
    }


# --------------- Validator Test Cases (STEP 5)---------------
if __name__ == "__main__":
    tests = [
        # (description, sql, expect_valid)
        (
            "Case 1 — valid SELECT",
            "SELECT name, contract_value FROM clients WHERE region = 'EMEA'",
            True,
        ),
        (
            "Case 2 — destructive DELETE blocked by keyword check",
            "DELETE FROM invoices WHERE status = 'unpaid'",
            False,
        ),
        (
            "Case 3 — column name containing blocked word allowed (whole-word match)",
            "SELECT id, deleted_at FROM clients WHERE deleted_at IS NOT NULL",
            True,
        ),
        (
            "Case 4 — stacked query with blocked keyword",
            "SELECT * FROM clients; DROP TABLE clients",
            False,
        ),
        (
            "Case 5 — stacked query with no blocked keyword (proves stacking check is independent)",
            "SELECT * FROM clients; SELECT * FROM projects",
            False,
        ),
    ]

    print("\n--- validate_sql test cases ---")
    all_passed = True
    for desc, sql, expect_valid in tests:
        result = validate_sql(sql)
        passed = result["valid"] == expect_valid
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {desc}")
        if not passed:
            print(f"       expected valid={expect_valid}, got valid={result['valid']}, reason={result.get('reason')}")
    print(f"\n{'All tests passed.' if all_passed else 'Some tests FAILED.'}")
