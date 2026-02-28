# Architecture

This document separates architecture into:
- Runtime analysis flow (`/analyze`, `/chat`)
- Document ingestion flow (PDF to vector store)

## Runtime Flow
```mermaid
flowchart LR
    U[User]
    FE[Next.js Frontend]
    API[FastAPI Backend]
    SUP[Supervisor Agent]

    TA[Technical Analysis Agent]
    FA[Fundamental Analysis Agent]
    NA[News Analysis Agent]

    MKT[Market Data Provider]
    VIS[Vision Model]
    TV[Tavily API]
    AOAI[Azure OpenAI]
    VDB[(Postgres + pgvector)]

    OUT[Sectioned Report<br/>Technical + Fundamental + News<br/>6-Month Rating 1-10]

    U --> FE
    FE -->|POST /analyze, /chat| API
    API --> SUP

    SUP --> TA
    SUP --> FA
    SUP --> NA

    TA --> MKT
    TA --> VIS
    VIS --> AOAI

    FA --> VDB
    FA --> AOAI

    NA --> TV
    NA --> AOAI

    SUP --> OUT
    API --> FE
    FE --> U
```

## Document Ingestion Flow
```mermaid
flowchart LR
    PDF[Annual Report PDFs]
    DI[Azure Document Intelligence<br/>Prebuilt Model]
    MD[Markdown Output]
    CH[LangChain Markdown Header Splitter]
    META[Metadata<br/>company, doc_type, year]
    VDB[(Postgres + pgvector)]
    FA[Fundamental Analysis Agent]
    SUP[Supervisor Agent]

    PDF --> DI --> MD --> CH
    CH --> META
    CH --> VDB
    META --> VDB

    VDB --> FA
    FA --> SUP
```

## Notes
- `Technical Analysis Agent` and `News Analysis Agent` are implemented.
- `Fundamental Analysis Agent` and `Supervisor Agent` are planned/in progress.
- Notebook-based prototyping (`notebooks/agent_playground.ipynb`) validates sub-agents before full API integration.
