# Email Extraction Agent Release Notes

## v0.1.0-dev.4

- Fixes Foundry session startup by installing the `src`-layout runtime package as a portable wheel instead of an editable build-workspace link.
- Adds an isolated CI installation that imports the packaged agent from outside the repository checkout before publishing the immutable source artifact.
- Retains the custom Invocations protocol and all existing activation boundaries; this release does not enable Search retrieval, BloomSky, or downstream writes.

## v0.1.0-dev.3

- Replaces the hosted runtime's Azure AI Search API-key requirement with Microsoft Entra bearer-token authentication.
- Uses the dedicated Hosted agent instance identity for Search and embedding access in Azure while retaining developer credentials for local validation.
- Adds fail-closed tests for Search and embedding token acquisition, bearer headers, and tenant-prefiltered vector queries.
- Keeps Search retrieval disabled until the agent identity RBAC and synthetic index-ingestion releases are separately approved.

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
