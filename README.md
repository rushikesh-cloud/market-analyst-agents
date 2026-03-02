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
AZURE_OPENAI_KEY=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=
AZURE_DOCUMENT_INTELLIGENCE_KEY=
PGVECTOR_CONNECTION_STRING=postgresql+psycopg://user:password@host:5432/dbname
TAVILY_API_KEY=
```

### 4) Run API (optional)
```
uvicorn app.main:app --reload --app-dir backend
```

### 4b) Run Next.js frontend
```
cd frontend
npm install
npm run dev
```
Frontend default API target is `/api`. For local split frontend/backend dev, set:
```
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

### 5) Run notebook
Open `notebooks/agent_playground.ipynb` and execute the cells.

### 6) Ingest fundamentals (PDF -> pgvector)
```
python scripts/ingest_fundamentals.py --pdf "data/raw/ACME_2023.pdf" --company "ACME" --year 2023
```
Ensure the target Postgres database has the `pgvector` extension installed:
```
CREATE EXTENSION IF NOT EXISTS vector;
```

## Docker (frontend + backend + nginx in one container)

Build image:
```bash
docker build -t market-analyst-agents:latest .
```

Run image (example):
```bash
docker run --rm -p 8080:80 --env-file .env market-analyst-agents:latest
```

- Frontend: `http://localhost:8080/`
- Backend via nginx proxy: `http://localhost:8080/api/...`
- Backend health: `http://localhost:8080/api/health`

## Deploy to Azure Container Instances (already logged in with `az`)

PowerShell example:
```powershell
$RG = "market-analyst-rg"
$LOC = "eastus"
$ACR = "marketanalystacr001"   # globally unique
$IMAGE = "market-analyst-agents:latest"
$ACI = "market-analyst-aci"
$DNS = "market-analyst-aci-001" # globally unique

az group create --name $RG --location $LOC

az acr create `
  --name $ACR `
  --resource-group $RG `
  --location $LOC `
  --sku Basic `
  --admin-enabled true

az acr build --registry $ACR --image $IMAGE .

$ACR_LOGIN_SERVER = az acr show --name $ACR --resource-group $RG --query loginServer -o tsv
$ACR_USER = az acr credential show --name $ACR --resource-group $RG --query username -o tsv
$ACR_PASS = az acr credential show --name $ACR --resource-group $RG --query "passwords[0].value" -o tsv

az container create `
  --resource-group $RG `
  --name $ACI `
  --image "$ACR_LOGIN_SERVER/$IMAGE" `
  --registry-login-server $ACR_LOGIN_SERVER `
  --registry-username $ACR_USER `
  --registry-password $ACR_PASS `
  --dns-name-label $DNS `
  --ports 80 `
  --cpu 2 `
  --memory 4 `
  --restart-policy Always `
  --secure-environment-variables `
    AZURE_OPENAI_ENDPOINT="<value>" `
    AZURE_OPENAI_KEY="<value>" `
    AZURE_OPENAI_DEPLOYMENT="<value>" `
    AZURE_OPENAI_API_VERSION="2024-02-01" `
    AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT="<value>" `
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="<value>" `
    AZURE_DOCUMENT_INTELLIGENCE_KEY="<value>" `
    PGVECTOR_CONNECTION_STRING="<value>" `
    TAVILY_API_KEY="<value>"

az container show --resource-group $RG --name $ACI --query ipAddress.fqdn -o tsv
```

After deployment:
- UI URL: `http://<fqdn>/`
- API URL: `http://<fqdn>/api/...`
