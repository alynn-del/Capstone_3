"""
orchestrator.py -- routes a question to the SQL agent, the retrieval agent,
or both, then synthesizes a single cited answer.
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from bedrock_client import invoke_claude
from agents.sql_agent import run_sql_agent
from agents.retrieval_agent import run_retrieval_agent
from dotenv import load_dotenv
import tokenomics_logger
load_dotenv()

#--------Defining Class for States--------------------
class OrchestratorState(TypedDict):
    question: str
    route: Optional[str]
    sql_result: Optional[dict]
    retrieval_result: Optional[dict]
    final_answer: Optional[str]
    token_usage: Optional[dict]
    routing_token_usage: Optional[dict]

#----Defining Shared Tool Registry--------
TOOL_REGISTRY = {
    "sql_agent": {
        "function": run_sql_agent,
        "schema": {
            "name": "sql_agent",
            "description": (
                "Query the relational database for structured data. "
                "Tables available: clients (client names, industries, regoins, contract values, creation dates), "
                "projects (project names, statuses, budgets, timelines), "
                "consultants (consultant profiles, assignments, specialisations, billable hourrs), "
                "invoices (billing records, amounts, payment statuses). "
                "Use for counts, aggregations, filtering by status, budget lookups, "
                "and any question requiring precise figures from structured records. "
                "Not for: policies, procedures, written guidelines, narrative content, "
                "or anything not stored as a structured table row."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"]
            }
        }
    },
    "retrieval_agent": {
        "function": run_retrieval_agent,
        "schema": {
            "name": "retrieval_agent",
            "description": (
                "Search indexed documents for policy, procedure, and case study "
                "information. Documents available: cloud-migration-playbook "
                "(steps and best practices for cloud migrations), "
                "q2-2026-client-report (quarterly client performance summary), "
                "remote-work-policy (guidelines and requirements for remote work), "
                "retail-supply-chain-case-study (case study on retail supply chain outcomes). "
                "Use for questions about processes, guidelines, narrative summaries, "
                "or any content that lives in documents rather than structured records. "
                "Not for: counts, financial figures, filtering by status, or any question "
                "that requires querying rows from a database table."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"]
            }
        }
    }
}

#-------Router Classification Node (Using Claude)---------------
def route_question(state: OrchestratorState) -> OrchestratorState:
    question = state["question"]

    sql_desc = TOOL_REGISTRY["sql_agent"]["schema"]["description"]
    retrieval_desc = TOOL_REGISTRY["retrieval_agent"]["schema"]["description"]

    prompt = (
        "You are a routing agent. Given a user question, decide which tool(s) to use.\n\n"
        f"sql_agent: {sql_desc}\n"
        f"retrieval_agent: {retrieval_desc}\n\n"
        "Rules:\n"
        "- Reply with 'sql' if only structured/database data is needed.\n"
        "- Reply with 'retrieval' if only document/policy/narrative content is needed.\n"
        "- Reply with 'both' if the question needs information from both tools.\n"
        "Reply with exactly one word: sql, retrieval, or both.\n\n"
        f"Question: {question}"
    )

    result = invoke_claude(prompt, max_tokens=10)
    raw = result["text"].strip().lower()

    if "both" in raw:
        route = "both"
    elif "sql" in raw:
        route = "sql"
    else:
        route = "retrieval"

    return {
        **state,
        "route": route,
        "routing_token_usage": {"input": result["input_tokens"], "output": result["output_tokens"]},
    }

#----------Adding Logging to Routing Section (fix for circular importing)--------------
def route_question_logged(state: OrchestratorState) -> OrchestratorState:
    result = route_question(state)
    tokenomics_logger.log_decision("routing", state["question"], {"route": result["route"]})
    return result

#-------Calling SQL Agent if Routed--------------
def call_sql_agent(state: OrchestratorState) -> OrchestratorState:
    if state["route"] not in ["sql", "both"]:
        return state
    result = run_sql_agent(state["question"])
    return {**state, "sql_result": result}

#---------Calling Retrieval Agent if Routed---------
def call_retrieval_agent(state: OrchestratorState) -> OrchestratorState:
    if state["route"] not in ["retrieval", "both"]:
        return state
    result = run_retrieval_agent(state["question"])
    return {**state, "retrieval_result": result}

#------------Synthesizing Results to be Outputted-----------------
def synthesise(state: OrchestratorState) -> OrchestratorState:
    context = ""

    if state.get("sql_result") and state["sql_result"].get("success"):
        r = state["sql_result"]
        context += f"SQL Agent Result:\n{r['answer']}\nSources: {r['sources']}\n\n"

    if state.get("retrieval_result") and state["retrieval_result"].get("success"):
        r = state["retrieval_result"]
        context += f"Retrieval Agent Result:\n{r['answer']}\nSources: {r['sources']}\n\n"

    if not context:
        return {
            **state,
            "final_answer": "I was unable to find relevant information to answer this question.",
            "token_usage": {"input": 0, "output": 0}
        }

    prompt = (
        f"Using the information below, answer the following question clearly.\n"
        f"Cite which agent(s) and source(s) the information came from.\n\n"
        f"Question: {state['question']}\n\n"
        f"Information:\n{context}"
    )
    result = invoke_claude(prompt, max_tokens=800)

    return {
        **state,
        "final_answer": result["text"],
        "token_usage": {
            "input": result["input_tokens"],
            "output": result["output_tokens"]
        }
    }

#-----------Building Orchestrator Using Langraph---------------
def build_orchestrator():
    graph = StateGraph(OrchestratorState)
    graph.add_node("router", route_question_logged)
    graph.add_node("sql_agent", call_sql_agent)
    graph.add_node("retrieval_agent", call_retrieval_agent)
    graph.add_node("synthesiser", synthesise)

    graph.set_entry_point("router")
    graph.add_edge("router", "sql_agent")
    graph.add_edge("sql_agent", "retrieval_agent")
    graph.add_edge("retrieval_agent", "synthesiser")
    graph.add_edge("synthesiser", END)

    return graph.compile()


#---------------Validation Status Helper------------------
def get_validation_statuses(result: OrchestratorState) -> dict:
    sql_status = None
    if result.get("sql_result"):
        sql_status = result["sql_result"].get("metadata", {}).get("validation", "UNKNOWN")

    retrieval_status = None
    if result.get("retrieval_result"):
        has_sources = bool(result["retrieval_result"].get("sources"))
        retrieval_status = "PASSED" if has_sources else "NO SOURCES"

    return {"sql": sql_status, "retrieval": retrieval_status}


#---------------Question Test Cases (STEP 6/7)-----------
if __name__ == "__main__":
    orchestrator = build_orchestrator()   
    test_questions = [
        # "Which consultant has the highest total billable hours?",                                                                            # RDS only (1)
        # "What is the total budget across all active projects?",                                                                              # RDS only (2)
        # "What is the company's policy on remote work?",                                                                                      # documents only (3)
        "What were key points in the Q2 Client Report for 2026?",                                                                                     # documents only (4)
        # "How many active projects do we have, and what is the project delivery methodology described in our documentation?",                 # both (5)
        # "Who are our top clients by contract value, and what does our client onboarding process say about contract setup?",                  # both (6)
        # "Delete all invoices that are overdue.",                                                                                             # unsafe SQL (7)
        # "What is the weather forecast for next week?",                                                                                       # unanswerable (8)
        # "How many active projects do we have?",                                                                                              # RDS only (9)
        # "What is our onboarding process?",                                                                                                   # documents only (10)
    ]

    cost_rows = []

    for q in test_questions:
        print(f"\n{'='*60}\nQ: {q}")
        result = tokenomics_logger.run_traced_query(orchestrator, q)

        breakdown = tokenomics_logger.calculate_full_query_cost(
            routing_tokens=result.get("routing_token_usage"),
            sql_tokens=result.get("sql_result", {}).get("metadata", {}).get("token_usage") if result.get("sql_result") else None,
            retrieval_tokens=result.get("retrieval_result", {}).get("metadata", {}).get("token_usage") if result.get("retrieval_result") else None,
            synthesis_tokens=result.get("token_usage"),
        )
        validation = get_validation_statuses(result)
        print(f"Route:              {result['route']}")
        print(f"SQL validation:     {validation['sql'] or 'N/A'}")
        print(f"Retrieval status:   {validation['retrieval'] or 'N/A'}")
        print(f"Answer: {result['final_answer']}")
        print(f"Tokens: {result['token_usage']}")
        print(f"Cost:   {breakdown}")

        cost_rows.append({
            "question": q[:55] + "..." if len(q) > 55 else q,
            "route": result["route"],
            **breakdown,
        })

    # --- 10-query cost summary table ---
    components = ["routing_cost", "sql_agent_cost", "retrieval_agent_cost", "synthesis_cost"]
    totals = {c: sum(r[c] for r in cost_rows) for c in components}
    grand_total = sum(r["total_cost"] for r in cost_rows)
    top_component = max(totals, key=totals.get)

    col_w = 57
    print(f"\n{'='*60}")
    print("10-QUERY COST SUMMARY")
    print(f"{'='*60}")
    print(f"{'Question':<{col_w}} {'Route':<10} {'Routing':>9} {'SQL':>9} {'Retrieval':>10} {'Synthesis':>10} {'Total':>9}")
    print(f"{'-'*col_w} {'-'*10} {'-'*9} {'-'*9} {'-'*10} {'-'*10} {'-'*9}")
    for r in cost_rows:
        print(
            f"{r['question']:<{col_w}} {r['route']:<10} "
            f"${r['routing_cost']:>8.6f} ${r['sql_agent_cost']:>8.6f} "
            f"${r['retrieval_agent_cost']:>9.6f} ${r['synthesis_cost']:>9.6f} "
            f"${r['total_cost']:>8.6f}"
        )
    print(f"{'-'*col_w} {'-'*10} {'-'*9} {'-'*9} {'-'*10} {'-'*10} {'-'*9}")
    print(
        f"{'TOTAL':<{col_w}} {'':<10} "
        f"${totals['routing_cost']:>8.6f} ${totals['sql_agent_cost']:>8.6f} "
        f"${totals['retrieval_agent_cost']:>9.6f} ${totals['synthesis_cost']:>9.6f} "
        f"${grand_total:>8.6f}"
    )
    print(f"\nGrand total (10 queries): ${grand_total:.6f}  |  ${grand_total * 1000:.2f} per 1,000 queries")
    print(f"Highest-cost component:   {top_component.replace('_', ' ')} (${totals[top_component]:.6f})")