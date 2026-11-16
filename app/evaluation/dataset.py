"""Hand-written evaluation set for the sample contract (Masked_apeXaudit-Proposal_v8.pdf).

Every `reference` was written by reading the contract text and checking the figures
against the pages listed in `gold_pages`. RAGAS uses `reference` as the "ground truth";
`gold_pages` is used for our own extra check: did retrieval fetch the right page at all?

Not included on purpose: questions the contract cannot answer (e.g. termination).
RAGAS metrics need a reference answer, and for those the correct behaviour ("insufficient
evidence") is tested separately by the chat API tests.
"""

EVAL_SET = [
    {
        "question": "What are the payment milestones and their percentages?",
        "reference": (
            "There are four payment milestones: Verification + Foundation 30%, "
            "Production Pipeline Live (Configured Projects) 30%, Governance + Ops Intelligence 30%, "
            "and Enhancements + Handover + Monitor 10%."
        ),
        "gold_pages": [10],
    },
    {
        "question": "What is the monthly O&M baseline fee and how many calls does it include?",
        "reference": (
            "The Monthly O&M Baseline is 30,000 INR per month (excluding taxes) "
            "and includes up to 1,000 calls per month."
        ),
        "gold_pages": [11],
    },
    {
        "question": "How much is charged per audited call?",
        "reference": "Per-call processing is charged at 5 INR per audited call (excluding taxes).",
        "gold_pages": [11],
    },
    {
        "question": "What would the monthly O&M bill be for 2,000 calls?",
        "reference": "For 2,000 calls per month the bill is 30,000 + (1,000 x 5) = 35,000 INR.",
        "gold_pages": [11],
    },
    {
        "question": "How many total man-hours are estimated for the project?",
        "reference": "The total estimated effort is approximately 1,500 man-hours.",
        "gold_pages": [9],
    },
    {
        "question": "What is the total duration of Phase-1?",
        "reference": (
            "The total duration is approximately 10-12 weeks, depending on client validation "
            "turnaround; by the end of Week 12 the system enters monitored and maintained mode."
        ),
        "gold_pages": [8],
    },
    {
        "question": "Why is Phase-1 designed as an export-driven approach?",
        "reference": (
            "Because Sell.Do does not provide developer-friendly APIs, webhooks or SDKs, "
            "so Phase-1 ingests Sell.Do exports and mCube recordings instead of integrating directly."
        ),
        "gold_pages": [3],
    },
    {
        "question": "What is the response target for a P1 (service down) support issue?",
        "reference": "For P1 (service down) issues the response target is to acknowledge within 2-4 hours.",
        "gold_pages": [13],
    },
    {
        "question": "What does the monthly O&M baseline cover?",
        "reference": (
            "It covers cloud hosting (24x7 always on), monitoring and reliability "
            "(daily health monitoring and alerts, log retention, automated backups), "
            "maintenance and minor improvements (monthly security patching, bug fixes, basic performance tuning), "
            "and a monthly O&M usage summary report."
        ),
        "gold_pages": [12],
    },
    {
        "question": "What is the fixed monthly infrastructure cost in the appendix?",
        "reference": (
            "Total fixed infrastructure OPEX (24x7 on) is 25,500 - 26,500 INR per month "
            "(about 305 - 315 USD), made up of compute, storage, networking, database and monitoring."
        ),
        "gold_pages": [14],
    },
    {
        "question": "What is explicitly out of scope for Phase-1?",
        "reference": (
            "Out of scope: direct Sell.Do API integration and webhooks, full Zoho CRM migration, "
            "deep enterprise features such as SSO/SAML and fine-grained RBAC, automated ingestion from "
            "email/WhatsApp/Slack, real-time streaming dashboards, multi-tenant productization, "
            "and complex agentic optimization workflows."
        ),
        "gold_pages": [5],
    },
    {
        "question": "What happens to the score when a fatal error is detected?",
        "reference": "Under the fatal bypass rule, any fatal error makes the score zero.",
        "gold_pages": [7, 10],
    },
]
