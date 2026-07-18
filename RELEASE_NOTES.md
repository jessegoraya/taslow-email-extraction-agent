# Email Extraction Agent Release Notes

## v0.1.0-dev.2

- Adds the Azure AI Foundry hosted-agent entry point using the Invocations protocol and a `/readiness` health endpoint.
- Uses Microsoft Entra workload identity by default for Foundry model calls; the Test hosted deployment does not require an Azure OpenAI key.
- Packages a checksum-pinned hosted-agent source archive; Foundry registers an immutable agent version when that approved archive is deployed.
- Moves component CI to Python 3.13 and adds hosted-agent contract coverage.

## v0.1.0-dev.1

- Provides the Taslow email-to-task extraction workflow, project/scope matching, assignee resolution, due-date handling, validation, and retry behavior.
- Includes synthetic evaluation and controlled Task-write smoke tooling.
- Adds Windows-safe IANA timezone data, Ruff enforcement, 72 automated tests, coverage, dependency auditing, and immutable Python package artifacts.
- Foundry hosted-agent deployment and the blocking 90 percent per-story evaluation remain separate release gates.
