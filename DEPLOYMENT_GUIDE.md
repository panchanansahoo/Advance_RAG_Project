# Deployment & Hosting Guide: Vercel, Render & GitHub Actions

This guide walks you through hosting your **Vanilla Frontend on Vercel**, your **FastAPI Backend on Render**, and automating continuous integration and deployment with **GitHub Actions**.

---

## 1. System Architecture

```mermaid
graph LR
    Browser[User Browser]
    Vercel[Vercel: Frontend Hosting]
    Render[Render: FastAPI Backend Container]
    GitHub[GitHub Repo: Advance_RAG_Project]
    GHA[GitHub Actions Workflows]

    Browser -->|Load HTML/CSS/JS| Vercel
    Browser -->|API Requests: /api/* (Proxied)| Vercel
    Vercel -->|Proxy to https://<service>.onrender.com| Render
    GitHub -->|Push to main| GHA
    GHA -->|1. Run Tests (ci.yml)| GHA
    GHA -->|2. Trigger Deploy Hook| Render
    GHA -->|3. Deploy Build Artifacts| Vercel
```

---

## 2. Backend Deployment on Render

### Step 2.1: Create Web Service on Render
1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** and select **Blueprint** (or **Web Service**):
   - **Blueprint (Recommended)**: Connect your repository `Advance_RAG_Project`. Render will detect `render.yaml` and configure the service automatically.
   - **Manual Web Service**:
     - Name: `advanced-rag-backend` (or your preferred name)
     - Environment: `Docker`
     - Dockerfile Path: `./Dockerfile`
     - Region: Choose closest to your users (e.g., Oregon or Frankfurt)
     - Plan: `Starter` (recommended) or `Free`
     - Health Check Path: `/api/v1/health`

### Step 2.2: Add Environment Variables in Render
In your Render Service Dashboard, go to **Environment** and add:
- `APP_ENV`: `production`
- `OPENAI_API_KEY`: `your_openai_api_key`
- `GOOGLE_API_KEY`: *(optional, if using Gemini)*
- `GROQ_API_KEY`: *(optional, if using Groq)*
- `DATABASE_PROVIDER`: `sqlite`
- `VECTOR_DB_PROVIDER`: `qdrant`
- `CORS_ORIGINS`: `["*"]`
- `SUPABASE_URL`: *(e.g., `https://<project-ref>.supabase.co`)*
- `SUPABASE_ANON_KEY`: *(your Supabase anon/public key)*

> [!NOTE]
> Render provides the `PORT` variable dynamically. The `Dockerfile` is pre-configured to bind automatically to `${PORT:-8000}`.

### Step 2.3: Generate Render Deploy Hook
1. In your Render Web Service dashboard, navigate to **Settings**.
2. Scroll down to the **Deploy Hook** section.
3. Click **Add Deploy Hook** (or copy existing).
4. Copy the URL (format: `https://api.render.com/deploy/srv-xxxxxxxxxxxx?key=yyyyyyyy`).
5. Note your backend URL (e.g., `https://advanced-rag-backend.onrender.com`).

---

## 3. Frontend Deployment on Vercel

### Step 3.1: Connect Vercel Project
1. Log in to [Vercel](https://vercel.com).
2. Click **Add New...** > **Project**.
3. Import your GitHub repository `Advance_RAG_Project`.
4. In the project configuration screen:
   - **Framework Preset**: Select **Other** (do NOT choose FastAPI / Python).
   - **Root Directory**: Click **Edit** and set it to **`vanilla_frontend`** (Recommended).
   > [!TIP]
   > Setting **Root Directory** to `vanilla_frontend` ensures Vercel only deploys the static frontend files and completely ignores the Python backend. If you keep Root Directory as `./`, the included `.vercelignore` and root `vercel.json` with `"framework": null` will prevent Vercel from attempting to detect or run FastAPI.
5. Click **Deploy**.

### Step 3.2: Configure API Proxy to Render Backend
If your Render service URL differs from `https://advanced-rag-backend.onrender.com`, update the destination URL in both:
- `vercel.json` (at repo root)
- `vanilla_frontend/vercel.json`

```json
{
  "rewrites": [
    {
      "source": "/api/:match*",
      "destination": "https://<YOUR-RENDER-SERVICE-NAME>.onrender.com/api/:match*"
    }
  ]
}
```

### Step 3.3: Obtain Vercel Credentials for GitHub Actions
To allow GitHub Actions to deploy to Vercel on your behalf, you need 3 items:

1. **`VERCEL_TOKEN`**:
   - Go to [Vercel Account Settings > Tokens](https://vercel.com/account/tokens).
   - Click **Create**, give it a name (e.g., `github-actions-token`), and copy the generated token.

2. **`VERCEL_ORG_ID` & `VERCEL_PROJECT_ID`**:
   - On your local terminal inside `e:\Advanced_Multimodal_Agentic_RAG`, run:
     ```bash
     npx vercel link
     ```
   - Follow prompts to link to your Vercel project.
   - Once linked, open `.vercel/project.json` inside your project directory to see:
     ```json
     {
       "orgId": "team_...",
       "projectId": "prj_..."
     }
     ```
   - Alternatively, on the Vercel Dashboard, go to your Project -> **Settings** -> **General** to copy the **Project ID**, and Team/Account Settings to copy **Team ID** (`orgId`).

---

## 4. Configure GitHub Actions Secrets

Go to your GitHub Repository:
`https://github.com/panchanansahoo/Advance_RAG_Project/settings/secrets/actions`

Click **New repository secret** and add the following:

| Secret Name | Description | Example / Format |
|---|---|---|
| `RENDER_DEPLOY_HOOK_URL` | Webhook URL to trigger Render deployment | `https://api.render.com/deploy/srv-xxx?key=yyy` |
| `VERCEL_TOKEN` | Vercel Personal Access Token | `xyz123...` |
| `VERCEL_ORG_ID` | Vercel Team / User ID | `team_abc123...` |
| `VERCEL_PROJECT_ID` | Vercel Project ID | `prj_xyz456...` |

---

## 5. How Continuous Deployment Works

### Workflow 1: `CI - Tests & Lint` (`.github/workflows/ci.yml`)
- Triggers on every **Pull Request** and **Push** across all branches.
- Sets up Python 3.11 with cached pip dependencies.
- Runs compilation check and pytest suite (`pytest tests/`).

### Workflow 2: `CI/CD - Deploy to Render & Vercel` (`.github/workflows/deploy.yml`)
- Triggers automatically on push to the `main` branch, or manually via **Run workflow** in the GitHub Actions tab.
- Uses path-filtering:
  - When backend files (`backend/`, `Dockerfile`, `requirements.txt`, `render.yaml`) change -> Triggers Render deployment via the deploy hook.
  - When frontend files (`vanilla_frontend/`, `vercel.json`) change -> Pulls config and deploys frontend directly to Vercel Production.

---

## 6. Verification Checklist

- [ ] Render service created and showing **Healthy** on `/api/v1/health`.
- [ ] Render deploy hook tested via `curl -X POST <RENDER_DEPLOY_HOOK_URL>`.
- [ ] Vercel deployment accessible and loads the ChatGPT-style interface.
- [ ] Four GitHub repository secrets configured.
- [ ] Pushing a commit to `main` triggers `.github/workflows/deploy.yml` with green checks.
