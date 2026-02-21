# Market Analyst Agent

## Setup

### 1) Create and activate local uv environment
```
uv venv
```

Activate:
```
.venv\Scripts\activate
```

### 2) Install dependencies
```
uv pip install -r requirements.txt
```

### 3) Configure environment variables
Create and fill `.env`:
```
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=2024-02-01
TAVILY_API_KEY=
```

### 4) Run API (optional)
```
uvicorn app.main:app --reload --app-dir backend
```

### 5) Run notebook
Open `notebooks/agent_playground.ipynb` and execute the cells.
