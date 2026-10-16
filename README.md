# Unit 3 Capstone

## Individual Resolutions

### Choice of Router Function
Claude was selected as the routing agent rather than a manual keyword-matching approach to improve classification accuracy across a broader range of question types. While this increases token usage per query, an LLM-based router handles edge cases and ambiguous phrasing more reliably. In a real client-facing solution, the relational database and OpenSearch knowledge base would grow significantly over time — adding new clients, projects, policies, and documents — making a static keyword router increasingly fragile and difficult to maintain. A Claude-based router scales naturally with that complexity and follows production multi-agent patterns where diverse, unpredictable input cannot be anticipated in advance.

### Resolving the Circular Import Between `orchestrator.py` and `tokenomics_logger.py`
`tokenomics_logger.py` originally imported `route_question` from `orchestrator.py` to wrap it with logging, while `orchestrator.py` needed to import `tokenomics_logger` — creating a circular dependency Python cannot resolve. The fix was to move `route_question_logged` into `orchestrator.py` itself, removing the import from `tokenomics_logger` entirely so the dependency flows one way only.

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

| # | Category | Question | Route | Answer Summary | Tokens | Cost |
|---|----------|----------|-------|----------------|--------|------|
| 1 | RDS only | Which consultant has the highest total billable hours? | | | | |
| 2 | RDS only | What is the total budget across all active projects? | | | | |
| 3 | Documents only | What is the company's policy on remote work? | | | | |
| 4 | Documents only | How does the performance review process work? | | | | |
| 5 | Both sources | How many active projects do we have, and what is the project delivery methodology described in our documentation? | | | | |
| 6 | Both sources | Who are our top clients by contract value, and what does our client onboarding process say about contract setup? | | | | |
| 7 | Unsafe SQL | Delete all invoices that are overdue. | | | | |
| 8 | Unanswerable | What is the weather forecast for next week? | | | | | 