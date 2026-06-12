# Taslow Email Extraction Agent

Python Microsoft Agent Framework service that replaces the retired PromptFlow-based
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

## Microsoft Agent Framework Notes

Workflow steps use Microsoft Agent Framework's Python functional workflow decorators when
the `agent-framework` package is installed. The pure async functions remain testable directly,
which lets business rules and scoring be verified without making model calls.

The current implementation includes deterministic task detection as a safe baseline. A Foundry
model-backed extractor can be enabled by implementing `FoundryTaskExtractor` without changing
the workflow contract.
