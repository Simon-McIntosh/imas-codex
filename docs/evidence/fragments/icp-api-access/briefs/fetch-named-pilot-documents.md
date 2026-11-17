# Coordinator instructions for fetching the named pilot documents

The plan section is the authority for what is delivered. These instructions add
the reuse map, the private inputs and the operating rules.

## Reuse map

- The read-only client in `imas_codex/discovery/icp/` already reads more than
  one host under one method and origin policy, with the sign-in exchange and
  its guardrails. The document host is one more listed host. Extend the
  client; do not open a second session, transport or sign-in path.
- Downloading through an authenticated session, with a size probe and type
  validation, is owned by the transfer client in
  `imas_codex/discovery/base/transfer.py`, which the wiki pipeline uses for
  document content. Use it for the bytes.
- Spreadsheet rows are read by the project's existing spreadsheet extraction.
- Private storage sits under the project's existing private data root; write
  the documents beneath it or under the private cache named below, and record
  which.

## Private inputs

- The document identities, versions and titles are in
  `~/.cache/imas-codex-codac-uda/pilot-documents/assistant-findings-2026-11-17.md`.
  Use the versions it states; where it notes a draft newer than the current
  version, fetch the current one.
- The download form is a GET on the document host that takes the identity, the
  version and a get-document action as query parameters; the private notes
  show the form in the assistant's links. Confirm it against one document
  before fetching the rest.
- The host and the identity list go into the gitignored private facility
  configuration. Append to that file; do not re-serialise it.

## Rules

- GET and HEAD only, enforced in code before any request. Non-read requests
  exist only inside the sign-in exchange, to listed sign-in origins.
- On the document host, an identity not on the private list is refused before
  any request is sent.
- No real host, document identity, title or content goes into a tracked file,
  test, report or commit message. Tests use invented hosts and identities.
- Keep the request count small and state it. Pause between requests through
  the shared pacer. Stop at the first 401, 403 or 5xx and report it.
- A response that is a sign-in page or an error body is a failed fetch.
- Never print or store credentials, cookies or tokens.
- Commit locally with a conventional subject and a body, and give `commits` in
  the manifest as a JSON list of full shas. Do not push or merge.
