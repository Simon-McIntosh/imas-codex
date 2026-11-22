# Let the read-only client fetch any IDM document by identity

## Goal

The project's ICP client fetches any IDM document by identity and version,
read-only, without a per-document allowlist.

## Why

The lead decided on 2026-11-22: "Any IDM document, read-only". Until now the
document gate (`_DocumentAccess` in `iter_codex/discovery/icp/client.py`)
refused any identity not listed in `document_identities` in the private
configuration, which held seven pilot documents. Signal lists and
self-description documents for every plant system are about to be catalogued
and fetched, and listing each by hand in private configuration does not scale.
The standing rule is unchanged: every request is GET or HEAD, refused in code
at the one transport chokepoint before any network call, and signs in with the
lead's account.

## What must be true afterwards

- A GET of the configured document form (path, `uid`, `version`, `action`) for
  any well-formed identity is allowed; every other method, path, query shape,
  fragment or origin is still refused before the network, as today.
- Requests to the document host are rate-limited, with the limit in private
  configuration and a safe default when it is missing; say in the evidence what
  the default is and why.
- Fetched bytes are written only under a private directory outside the
  repository, as today.
- `document_identities` is no longer required; if it is still present it is
  ignored or removed, and the evidence says which.

## Done when

- A test shows a GET of an unlisted, well-formed identity is allowed, while
  POST, PUT, PATCH and DELETE to the same URL, a malformed identity, an extra
  query key and a different path are each refused before any network call.
- The existing write-refusal tests still pass, unchanged.
- One live fetch of a document that is not one of the seven pilot identities
  succeeds through the client, with its digest recorded privately; the
  identity is not written to any tracked file.
- The tests covering the client (`tests/discovery` for the ICP package) add no
  failure against the base, both arms recorded in the manifest as single-line
  JSON objects with whole test ids.

Negative control: with the identity check restored, the unlisted-identity test
fails.

## Rules

- Change only the document access check, its configuration reading and their
  tests. Do not widen any other rule of the request policy.
- No document identity, title or host name in a tracked file; tests use
  invented identities and hosts.
- Private data goes under `~/.cache/imas-codex-codac-uda/`, never only in your
  worktree's ignored files.
- Copy `iter_codex/config/facilities/iter_private.yaml` from the main checkout
  into your worktree before the live fetch; do not edit the main checkout's
  copy. Say in the manifest which private keys the main checkout needs.
- Run `ruff check --fix` and `ruff format` on exactly the files you change.
- Commit locally with a conventional subject and a body, and give `commits` as
  a JSON list of full shas. Do not push or merge.

## Scope

Write paths: `iter_codex/discovery/icp`, `tests/discovery`, and
`docs/evidence/fragments/iter-plant-signal-ontology/any-idm-document-read-only.html`.
