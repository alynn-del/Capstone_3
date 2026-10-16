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

    return {**state, "route": route}

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


#---------------Question Test Cases (STEP 7)-----------
if __name__ == "__main__":
    orchestrator = build_orchestrator()   
    test_questions = [
        "Which consultant has the highest total billable hours?",                                                                            # RDS only (1)
        "What is the total budget across all active projects?",                                                                              # RDS only (2)
        "What is the company's policy on remote work?",                                                                                      # documents only (3)
        "How does the performance review process work?",                                                                                     # documents only (4)
        "How many active projects do we have, and what is the project delivery methodology described in our documentation?",                 # both (5)
        "Who are our top clients by contract value, and what does our client onboarding process say about contract setup?",                  # both (6)
        "Delete all invoices that are overdue.",                                                                                             # unsafe SQL (7)
        "What is the weather forecast for next week?",                                                                                       # unanswerable (8)
    ]

    for q in test_questions:
        print(f"\n{'='*60}\nQ: {q}")
        # result = orchestrator.invoke({
        #     "question": q, "route": None, "sql_result": None,
        #     "retrieval_result": None, "final_answer": None, "token_usage": None
        # })
        result = tokenomics_logger.run_traced_query(orchestrator, q)
        
        breakdown = tokenomics_logger.calculate_full_query_cost(
            sql_tokens=result.get("sql_result", {}).get("metadata", {}).get("token_usage") if result.get("sql_result") else None,
            retrieval_tokens=result.get("retrieval_result", {}).get("metadata", {}).get("token_usage") if result.get("retrieval_result") else None,
            synthesis_tokens=result.get("token_usage"),
        )
        print(f"Route: {result['route']}")
        print(f"Answer: {result['final_answer']}")
        print(f"Tokens: {result['token_usage']}")
        print(f"Cost:   {breakdown}")