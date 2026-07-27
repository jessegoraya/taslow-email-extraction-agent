# Email Extraction Agent Release Notes

## v0.1.0-dev.7

- Recognizes a bare forwarded delivery as a current handoff only when a deterministic guard finds
  a concrete direct action request. Completed, FYI, conditional, and status-only forwards remain
  non-actionable.
- Adds `document` and `incorporate` to the bounded action vocabulary used by recovery.
- Resolves otherwise tied Project candidates with the lexical, Search, client-domain, and
  Search-margin evidence already used by Project scoring.
- Publishes all hydrated Project candidate scores in diagnostics so governed evaluation can
  distinguish Search absence from a below-threshold candidate.
- Keeps Project confidence, tenant isolation, no-write behavior, and stable hosted routing
  unchanged.

## v0.1.0-dev.6

- Expands tenant-filtered Project retrieval from eight to twenty-five Search candidates so the
  21-Project BloomSky Test workspace can be evaluated without dropping relevant Projects before
  scoring.
- Resolves equal-confidence Project candidates with stronger participant overlap and then stronger
  associated-people context while preserving Search order when those signals are equal.
- Adds `project_selection_participant_tiebreak` evidence whenever the deterministic tie-break
  changes the selected Project.
- Leaves Scope selection, due-date normalization, and downstream-write policy unchanged.

## v0.1.0-dev.5

- Adds a fail-closed second-pass task-detection recovery for strong action signals that the first
  model pass classified as `no_task_found`.
- Limits model input to newest authored content and explicitly delegated forwarded context so stale
  quoted requests cannot create new tasks.
- Rejects completed work, informational/status-only mail, courtesy closings, and conditional future
  needs before recovery.
- Adds recovery attempt, outcome, and reason diagnostics plus a sealed-tranche audit tool.
- Preserves all existing Project, Scope, assignee, validation, and write gates; this release does
  not weaken project-selection thresholds or activate downstream writes.

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
