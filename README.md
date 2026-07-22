# Taslow Email Extraction Agent

Python hosted-agent service that replaces the retired PromptFlow-based
`TaslowEmailExtractML` implementation.

The service exposes `POST /email-extractions`, accepts the tenant email ingestion payload,
and runs a deterministic workflow:

1. Detect task candidates.
2. Retrieve tenant-scoped project context.
3. Score project confidence.
4. Match optional project scope areas.
5. Resolve assignees.
6. Normalize due dates.
7. Validate the final result and return a normalized response.

The implementation keeps tenant, project, and task ownership outside this repository. This
repository owns only extraction workflow code, prompts, scoring, service clients, tests, and
deployment scaffolding.

## Local Setup

Install VS Code extensions:

- Foundry Toolkit for Visual Studio Code
- Python
- Pylance
- Docker, if containerizing locally
- Azure Resources / Azure Account, if browsing Azure resources from VS Code

Install Python dependencies. Python 3.12+ is recommended for the Foundry Toolkit hosted
agent workflow path; the service code also runs on Python 3.11 for local development and tests.

```powershell
cd C:\Users\jgora\OneDrive\Documents\taslow-email-extraction-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Run tests:

```powershell
pytest
```

Run the local API:

```powershell
uvicorn taslow_email_extraction_agent.app:app --reload --port 8087
```

The API docs are available at `http://localhost:8087/docs`.

## View the Workflow in Agent Inspector

Agent Inspector does not discover the FastAPI endpoint directly. It launches an Agent
Framework entrypoint with `agentdev`, then discovers workflows from `/agentdev/entities`.
This repo includes `inspector_entrypoint.py` and VS Code debug configuration for that path.

Install the inspector-specific packages:

```powershell
python -m pip install -r requirements-inspector.txt
```

Open the repo in VS Code:

```powershell
code C:\Users\jgora\OneDrive\Documents\taslow-email-extraction-agent
```

Then use one of these paths:

1. Press `F5`.
2. Choose `Debug Taslow Agent Inspector`.
3. Agent Inspector should open on port `8087`.
4. Send a sample message such as:

```text
Tessa, please update the electrical scope by next Friday at 5.
```

The workflow graph should show the Taslow workflow steps. Double-click a node to jump back
to the related Python code.

Manual command if you want to test without VS Code debugging:

```powershell
$env:PYTHONPATH="C:\Users\jgora\OneDrive\Documents\taslow-email-extraction-agent\src"
python -m agentdev run inspector_entrypoint.py --port 8087
```

Then open Agent Inspector from Foundry Toolkit and point it at port `8087`.

## Microsoft Agent Framework Inspector Notes

The production Foundry runtime uses the framework-agnostic Invocations protocol adapter and does
not install Microsoft Agent Framework. The optional Inspector dependencies enable the Python
functional-workflow decorators for local visual authoring. Without that optional extra, the pure
async functions run directly, which keeps business rules and scoring testable without model calls.

The current implementation includes a deterministic fallback and a Foundry model-backed task
extractor. In Azure Test it uses Microsoft Entra managed identity rather than an Azure OpenAI
API key. The hosted-agent entry point is `main.py`, which exposes Foundry Agent Service's
`/invocations` protocol for the existing structured email payload. The image supports keyless
Azure AI Search retrieval and managed-identity Project hydration through APIM. Mailbox access,
index population, APIM identity policy, remote invocation, and Task writes remain separately
governed activation phases.

The Azure AI Search project/scope client is also keyless. With
`AGENT_PROJECT_SEARCH_PROVIDER=azure-ai-search`, it obtains bearer tokens for Search and the
embedding model through `DefaultAzureCredential`. Project hydration uses
`PROJECT_SERVICE_BASE_URL` and `PROJECT_SERVICE_TOKEN_SCOPE`; APIM must validate the token and
the dedicated agent identity before forwarding an internal request. In a Hosted agent, Azure
resolves the credential to the agent's instance identity; locally it can use the developer's
Azure credential.

Foundry's direct-code remote build installs `requirements.txt` in a temporary build workspace and
then starts the copied source at `/app/main.py`. The runtime requirement therefore installs this
project as a wheel (`.`), not as an editable checkout (`-e .`), so the `src` package remains
importable after the temporary build workspace is removed. Component CI verifies that behavior in
an isolated virtual environment from outside the repository checkout.
