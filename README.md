# Unit 3 Capstone

## Design Choices

### Choice of Router Function
Claude was selected as the routing agent rather than a manual keyword-matching approach to improve classification accuracy across a broader range of question types. While this increases token usage per query, an LLM-based router handles edge cases and ambiguous phrasing more reliably. In a real client-facing solution, the relational database and OpenSearch knowledge base would grow significantly over time — adding new clients, projects, policies, and documents — making a static keyword router increasingly fragile and difficult to maintain. A Claude-based router scales naturally with that complexity and follows production multi-agent patterns where diverse, unpredictable input cannot be anticipated in advance.

### Resolving the Circular Import Between `orchestrator.py` and `tokenomics_logger.py`
`tokenomics_logger.py` originally imported `route_question` from `orchestrator.py` to wrap it with logging, while `orchestrator.py` needed to import `tokenomics_logger` — creating a circular dependency Python cannot resolve. The fix was to move `route_question_logged` into `orchestrator.py` itself, removing the import from `tokenomics_logger` entirely so the dependency flows one way only.

## Tokenomics

### 10-Query Cost Summary

| Question | Route | Routing | SQL | Retrieval | Synthesis | Total |
|----------|-------|---------|-----|-----------|-----------|-------|
| Which consultant has the highest total billable hours? | sql | $0.001167 | $0.001716 | $0.000000 | $0.002130 | $0.005013 |
| What is the total budget across all active projects? | sql | $0.001167 | $0.001395 | $0.000000 | $0.001470 | $0.004032 |
| What is the company's policy on remote work? | retrieval | $0.001182 | $0.000000 | $0.011142 | $0.009879 | $0.022203 |
| How does the performance review process work? | retrieval | $0.001176 | $0.000000 | $0.006276 | $0.004581 | $0.012033 |
| How many active projects do we have, and what is the project delivery methodology described in our documentation? | both | $0.001197 | $0.001935 | $0.009876 | $0.006516 | $0.019524 |
| Who are our top clients by contract value, and what does our client onboarding process say about contract setup? | both | $0.001206 | $0.003918 | $0.010149 | $0.007380 | $0.022653 |
| Delete all invoices that are overdue. | sql | $0.001164 | $0.000927 | $0.000000 | $0.000000 | $0.002091 |
| What is the weather forecast for next week? | retrieval | $0.001179 | $0.000000 | $0.005883 | $0.003414 | $0.010476 |
| How many active projects do we have? | sql | $0.001161 | $0.001248 | $0.000000 | $0.001170 | $0.003579 |
| What is our onboarding process? | retrieval | $0.001176 | $0.000000 | $0.008040 | $0.003747 | $0.012963 |
| **TOTAL** | | **$0.011775** | **$0.011139** | **$0.051366** | **$0.040287** | **$0.114567** |

**Grand total (10 queries): $0.114567 — $114.57 per 1,000 queries**

**Highest-cost component: Retrieval agent ($0.051366)**

### Why Retrieval Was the Most Expensive Component

The retrieval agent cost $0.051366 across the 10 queries — 45% of the total spend — compared to $0.011139 for the SQL agent. There are two compounding reasons.

**Document context is much larger than query results.** When the retrieval agent calls Claude to generate an answer, it passes the full text of up to five retrieved document chunks as context. Those chunks can be hundreds of words each, making the input token count for a single retrieval call significantly larger than a SQL interpretation call, which only receives column names and a capped result set of up to 20 rows. To combat this issue, I changed the number of top k entries to three in order to reduce the amount of tokens feeded into the Claude interpretation call while still feeding the model an appropriate amount of context to answer the question.

**The test suite was skewed toward retrieval-heavy queries.** Of the 10 questions, 4 routed to retrieval-only, 4 routed to sql-only, and 2 routed to both — meaning retrieval ran in 6 out of 10 queries versus SQL in 6 as well, but both "both" queries also hit retrieval. More importantly, the two blocked/trivial SQL queries (the `DELETE` request and the unanswerable question) cost near zero, which pulls the SQL column total cost down. The unanswerable question about documents was still routed to retrieval, spending $0.010476 even though no useful answer existed. A possible solution to this could add a confidence threshold to skip the retrieval call when the router is uncertain, reducing wasted spend on out-of-scope queries.


## Tests

### Validate SQL Query Test Suite

The `validate_sql` function in `agents/sql_agent.py` runs before every database call, blocking destructive and malformed queries. The five cases below verify each independent protection layer. Run with `python -m agents.sql_agent` from the project root.

| # | Case | SQL | Expected | Result |
|---|------|-----|----------|--------|
| 1 | Valid SELECT | `SELECT name, contract_value FROM clients WHERE region = 'EMEA'` | ALLOWED | PASS |
| 2 | Destructive keyword (`DELETE`) | `DELETE FROM invoices WHERE status = 'unpaid'` | BLOCKED — keyword check | PASS |
| 3 | Column name containing a blocked word (`deleted_at`) | `SELECT id, deleted_at FROM clients WHERE deleted_at IS NOT NULL` | ALLOWED — whole-word match only, `\bDELETE\b` does not match `deleted_at` | PASS |
| 4 | Stacked query with a blocked keyword | `SELECT * FROM clients; DROP TABLE clients` | BLOCKED — keyword check | PASS |
| 5 | Stacked query with no blocked keyword | `SELECT * FROM clients; SELECT * FROM projects` | BLOCKED — semicolon/stacking check (independent of keyword check) | PASS |

Case 3 proves the validator matches whole words, not substrings. Case 5 proves the stacking check is a separate step that catches multi-statement injections the keyword blocklist alone would miss.

### Question Test Suite

Eight questions run end-to-end through the orchestrator to verify routing, agent execution, synthesis, and safety handling across all expected input categories. Run with `python -m orchestrator` from the project root.

| # | Category | Question | Route | Answer Summary | Validation Status |
|---|----------|----------|-------|----------------|-------------------|
| 1 | RDS only | Which consultant has the highest total billable hours? | SQL | Tom Baker - 40 hours | Passed |
| 2 | RDS only | What is the total budget across all active projects? | SQL | $1,010,000 | Passed |
| 3 | Documents only | What is the company's policy on remote work? | Retrieval | Summary of Eligibility, Availability, Equipment & Security, Home Office Stipend, Performance & Review | Passed |
| 4 | Documents only | What were key points in the Q2 Client Report for 2026? | Retrieval | Summary of Progress, Risks, Budget, Staffing, and Client Feedback| Passed |
| 5 | Both sources | How many active projects do we have, and what is the project delivery methodology described in our documentation? |Both | 4 active projects & Methodology phases listed| Passed |
| 6 | Both sources | Who are our top clients by contract value, and what does our client onboarding process say about contract setup? | Both | Table output of top clients ordered by contract value & unable to find information regarding onboarding process|Passed |
| 7 | Unsafe SQL | Delete all invoices that are overdue. | SQL | Unable to find answer in documents | Passed (Blocked by validator) |
| 8 | Unanswerable | What is the weather forecast for next week? | Retrieval | No weather forecast information in the sources| Passed (Declined Honestly) | 

**Note:** Questions 6 and 8 demonstrates how the output for the agent being unable to answer a question without information in the documents did pass validation as this is the expected behavior for the agent to not hallucinate and therefore passed the validation check that the answer was grounded in the context that there was no matching evidence.

## Reflection

### Limits of Routing and Synthesis Logic

Question 6 — "Who are our top clients by contract value, and what does our client onboarding process say about contract setup?" — most tested the limits of the orchestrator. It routed correctly to `both`, and the SQL agent returned a valid ranked list of clients. The failure was a **synthesis problem**: the retrieval agent could not find onboarding contract information in the indexed documents, so the synthesised answer combined a concrete data result with an honest "I cannot find this" from the retrieval side. The synthesis step had to compose an answer from two results that were fundamentally uneven — one factual and complete, one an acknowledged gap — which is a harder task than synthesising two complementary positive results. This points to where multi-agent systems are still fragile compared to a single well-scoped agent: a single agent either knows the answer or doesn't, and its uncertainty is contained. In a multi-agent system, partial failure in one branch still flows into synthesis, and the quality of the final answer depends on the synthesiser correctly weighting and attributing each component rather than presenting the gap as a flaw in the whole response. Question 8 (the unanswerable weather question) is a secondary example of a **routing problem** — the system had no `reject` path, so it defaulted to retrieval and spent $0.010476 retrieving nothing useful. A potential fix to this would be to add a fourth route option for out-of-scope questions to short-circuit before any agent runs.
