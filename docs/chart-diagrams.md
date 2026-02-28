# Chart Diagrams

Mermaid chart diagrams based on `plan.md` and the architecture definition.

## 1. Runtime Execution Sequence
```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as Next.js Frontend
    participant API as FastAPI Backend
    participant SUP as Supervisor Agent
    participant TA as Technical Agent
    participant FA as Fundamental Agent
    participant NA as News Agent
    participant VDB as Postgres+pgvector
    participant TV as Tavily
    participant AOAI as Azure OpenAI

    User->>FE: Submit company/query
    FE->>API: POST /analyze or /chat
    API->>SUP: Dispatch analysis request

    par Technical lane
        SUP->>TA: Run technical analysis
        TA->>AOAI: Vision interpretation
        TA-->>SUP: Technical signals
    and Fundamental lane
        SUP->>FA: Run fundamentals analysis
        FA->>VDB: Retrieve company chunks
        FA->>AOAI: Financial reasoning
        FA-->>SUP: Multi-year trends
    and News lane
        SUP->>NA: Run news analysis
        NA->>TV: Search recent events
        NA->>AOAI: Summarize + sentiment
        NA-->>SUP: News insights
    end

    SUP-->>API: Sectioned report + 6-month rating
    API-->>FE: JSON response
    FE-->>User: Render dashboard/chat answer
```

## 2. Project Milestone Timeline
```mermaid
gantt
    title Market Analyst Agent Milestones
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Completed
    Backend skeleton + notebook scaffold :done, m1, 2026-02-20, 1d
    Technical agent end-to-end         :done, m2, 2026-02-21, 1d
    News agent end-to-end              :done, m4, 2026-02-22, 1d

    section Pending
    Fundamental agent end-to-end       :active, m3, 2026-02-23, 3d
    Supervisor + API synthesis         :m5, 2026-02-26, 3d
    Frontend integration               :m6, 2026-03-02, 4d
```

## 3. Workstream Completion Snapshot
```mermaid
pie title Workstream Completion (from current plan)
    "Completed" : 3
    "In Progress" : 2
    "Pending" : 7
```
