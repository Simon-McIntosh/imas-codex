# Offsite graph package inventory

Measured at `2026-09-01T18:10:26Z` against GitHub Packages for the
`simon-mcintosh` account. GitHub's package and package-version REST records are
the inventory and push-time authority; OCI manifests independently prove that
the selected versions still resolve to archive layers.

## Result

**A full-graph offsite copy does exist.** It is
`ghcr.io/simon-mcintosh/imas-codex-graph:v5.3.0-rc6`, GitHub Packages version
ID `1008109688`, pushed at `2026-07-07T12:27:23Z`. At the measurement time it
was **4,858,983 seconds old: 56 days, 5:43:03, or 56.238229167 days**.

The account holds four imas-codex graph archive packages. Only one is full
scope; the others are two generations of DD-only packaging and one
per-facility TCV subset.

| Registry package | Newest version tag | Registry push time | Scope | Version ID | Newest layer |
|---|---|---|---|---:|---|
| `imas-codex-graph` | `v5.3.0-rc6` (also `latest`) | `2026-07-07T12:27:23Z` | **Full graph: all facilities plus DD** | `1008109688` | 2,434,869,050 bytes, `sha256:11bd24d…` |
| `imas-codex-graph-dd` | `v5.3.0-rc6` (also `latest`) | `2026-07-07T12:36:25Z` | **DD-only**, current package name | `1008139372` | 2,096,263,875 bytes, `sha256:e7111409…` |
| `imas-codex-graph-imas` | `4.0.1.dev1921-gb863cb4aa-r1` | `2026-03-19T11:39:46Z` | **DD-only**, legacy `imas_only` package name | `745647988` | 2,029,533,924 bytes, `sha256:b1a6ba03…` |
| `imas-codex-graph-tcv` | `5.3.0rc6.dev91-g2479801d7.d20260706-r1` | `2026-07-07T12:57:57Z` | **Per-facility: TCV plus DD** | `1008212308` | 2,810,090,518 bytes, `sha256:0cb92406…` |

“Newest” means the version record with the greatest registry-owned
`created_at` value after paginating every version record. It does not mean the
result of `resolve_latest_tag()`, whose ordering is tag grammar and revision,
not registry time. For all four newest records, `updated_at` equals
`created_at`.

The account query returned 12 container packages in total. Five package names
contain `graph`; the fifth is `tessera-graph`. Its newest artifact is
`0.1.0-dev-432b303`, pushed `2026-07-09T09:52:57Z`, with a 10,849-byte layer of
media type `application/vnd.tessera.graph.tar+gzip`. It is a different
product's graph artifact, not an imas-codex archive and not a restore point for
this store. Keeping it in the census makes the account-wide search boundary
explicit rather than silently filtering it away.

## Scope classification

The package-name classification is not inferred from layer size:

- At the exact source revision embedded in the three newest imas-codex OCI
  manifests, `get_package_name()` maps no facility arguments to
  `imas-codex-graph`, `dd_only=True` to `imas-codex-graph-dd`, and a facility
  list to a suffixed package such as `imas-codex-graph-tcv`.
- The release implementation at that revision names `imas-codex-graph` as the
  full variant, `imas-codex-graph-dd` as the DD-only variant, and facility
  suffixes as per-facility variants.
- Repository history immediately before the `imas-only` to `dd-only` rename
  maps `imas_only=True` to `imas-codex-graph-imas` and explicitly describes it
  as the DD-only graph. The renamed implementation maps the same role to
  `imas-codex-graph-dd`.
- The TCV OCI description independently says `TCV + DD read-access share`.

The registry therefore holds a real full-graph package, not only facility
subsets. The earlier 55.996-day TCV currency measurement was accurate for its
package but did not answer full-graph recoverability; the account-wide census
does.

## Disaster-recovery verdict

**NO — release-pinned offsite copies are not an adequate disaster-recovery
floor for the 15,058,189,845-byte live store.**

The positive result is that catastrophic loss of the shared GPFS filesystem
would not leave zero full-graph state: the `v5.3.0-rc6` full archive still
resolves in GHCR. The unacceptable bound is its age. At measurement it permits
up to **56.238 days of full-graph data loss**. The newer TCV package is only 30
minutes newer, and the DD-only packages cannot reconstruct non-TCV facility
data, source code discoveries, mappings, reviews, costs, or any other
full-graph content absent from their filtered scopes. Release cadence is
therefore acting as the backup schedule, and that schedule has left the only
full recovery point more than eight weeks behind the live store.

This inventory proves registry presence and OCI-manifest fetchability. It does
not claim a trial restore or content-equivalence check for the July archive;
those would be additional recovery-confidence gates. Even granting that
archive as restorable, its recovery-point objective is already inadequate for
the live store.

## Registry measurement commands

This is the complete command log for the registry inventory and age
measurement. Source and plan reads used to establish semantics are recorded in
the next section. No command mutated a package or downloaded an archive layer.

1. Enumerate imas-codex graph packages directly from the account package list:

   ```text
   gh api --paginate '/users/simon-mcintosh/packages?package_type=container&per_page=100' --jq '.[] | select(.name | startswith("imas-codex-graph")) | [.name, .visibility, .created_at, .updated_at] | @tsv'
   exit: 0
   result: 4 packages — imas-codex-graph, imas-codex-graph-imas, imas-codex-graph-dd, imas-codex-graph-tcv
   ```

2. Attempt to combine paginated pages inside `gh`; the installed CLI refuses
   `--slurp` together with `--jq`. Each package command failed before making a
   measurement, and the corrected commands are item 3:

   ```text
   gh api --paginate --slurp '/users/simon-mcintosh/packages/container/imas-codex-graph/versions?per_page=100' --jq 'add | sort_by(.created_at) | reverse | .[] | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 1 — the --slurp option is not supported with --jq or --template

   gh api --paginate --slurp '/users/simon-mcintosh/packages/container/imas-codex-graph-imas/versions?per_page=100' --jq 'add | sort_by(.created_at) | reverse | .[] | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 1 — the --slurp option is not supported with --jq or --template

   gh api --paginate --slurp '/users/simon-mcintosh/packages/container/imas-codex-graph-dd/versions?per_page=100' --jq 'add | sort_by(.created_at) | reverse | .[] | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 1 — the --slurp option is not supported with --jq or --template

   gh api --paginate --slurp '/users/simon-mcintosh/packages/container/imas-codex-graph-tcv/versions?per_page=100' --jq 'add | sort_by(.created_at) | reverse | .[] | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 1 — the --slurp option is not supported with --jq or --template
   ```

3. Enumerate every version record for each of the four packages. The endpoint
   returns newest first; every returned record was inspected, and the first
   record in each output is the table row above:

   ```text
   gh api --paginate '/users/simon-mcintosh/packages/container/imas-codex-graph/versions?per_page=100' --jq '.[] | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 0 — newest: v5.3.0-rc6/latest, ID 1008109688, 2026-07-07T12:27:23Z

   gh api --paginate '/users/simon-mcintosh/packages/container/imas-codex-graph-imas/versions?per_page=100' --jq '.[] | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 0 — newest: 4.0.1.dev1921-gb863cb4aa-r1, ID 745647988, 2026-03-19T11:39:46Z

   gh api --paginate '/users/simon-mcintosh/packages/container/imas-codex-graph-dd/versions?per_page=100' --jq '.[] | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 0 — newest: v5.3.0-rc6/latest, ID 1008139372, 2026-07-07T12:36:25Z

   gh api --paginate '/users/simon-mcintosh/packages/container/imas-codex-graph-tcv/versions?per_page=100' --jq '.[] | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 0 — newest: 5.3.0rc6.dev91-g2479801d7.d20260706-r1, ID 1008212308, 2026-07-07T12:57:57Z
   ```

4. Independently select the maximum `created_at` record after collecting every
   paginated page. This avoids depending on API output order:

   ```text
   set -o pipefail; gh api --paginate --slurp '/users/simon-mcintosh/packages/container/imas-codex-graph/versions?per_page=100' | jq 'add | max_by(.created_at) | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 0 — v5.3.0-rc6/latest, ID 1008109688, 2026-07-07T12:27:23Z

   set -o pipefail; gh api --paginate --slurp '/users/simon-mcintosh/packages/container/imas-codex-graph-imas/versions?per_page=100' | jq 'add | max_by(.created_at) | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 0 — 4.0.1.dev1921-gb863cb4aa-r1, ID 745647988, 2026-03-19T11:39:46Z

   set -o pipefail; gh api --paginate --slurp '/users/simon-mcintosh/packages/container/imas-codex-graph-dd/versions?per_page=100' | jq 'add | max_by(.created_at) | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 0 — v5.3.0-rc6/latest, ID 1008139372, 2026-07-07T12:36:25Z

   set -o pipefail; gh api --paginate --slurp '/users/simon-mcintosh/packages/container/imas-codex-graph-tcv/versions?per_page=100' | jq 'add | max_by(.created_at) | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 0 — 5.3.0rc6.dev91-g2479801d7.d20260706-r1, ID 1008212308, 2026-07-07T12:57:57Z
   ```

5. Enumerate all account container-package names to prove the filter boundary:

   ```text
   gh api --paginate '/users/simon-mcintosh/packages?package_type=container&per_page=100' --jq '.[] | .name'
   exit: 0
   result: 12 packages total; graph-named packages were the four imas-codex packages plus tessera-graph
   ```

6. Fetch the newest OCI manifest for every imas-codex graph package:

   ```text
   oras manifest fetch 'ghcr.io/simon-mcintosh/imas-codex-graph:v5.3.0-rc6'
   exit: 0 — layer 2,434,869,050 bytes; version v5.3.0-rc6; source commit e4bced8af5e7d3d7ddaacc0fe7a5cd01b56afc9d

   oras manifest fetch 'ghcr.io/simon-mcintosh/imas-codex-graph-imas:4.0.1.dev1921-gb863cb4aa-r1'
   exit: 0 — layer 2,029,533,924 bytes; version 4.0.1.dev1921-gb863cb4aa-r1; source commit b863cb4aaa720a467c918c982d80ef1c376f8f12

   oras manifest fetch 'ghcr.io/simon-mcintosh/imas-codex-graph-dd:v5.3.0-rc6'
   exit: 0 — layer 2,096,263,875 bytes; version v5.3.0-rc6; source commit e4bced8af5e7d3d7ddaacc0fe7a5cd01b56afc9d

   oras manifest fetch 'ghcr.io/simon-mcintosh/imas-codex-graph-tcv:5.3.0rc6.dev91-g2479801d7.d20260706-r1'
   exit: 0 — layer 2,810,090,518 bytes; version 5.3.0rc6.dev91-g2479801d7.d20260706-r1; description TCV + DD read-access share
   ```

7. Enumerate and inspect the other graph-named account package:

   ```text
   gh api --paginate '/users/simon-mcintosh/packages/container/tessera-graph/versions?per_page=100' --jq '.[] | {id, name, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 0 — newest: 0.1.0-dev-432b303/latest, ID 1014577800, 2026-07-09T09:52:57Z

   oras manifest fetch 'ghcr.io/simon-mcintosh/tessera-graph:0.1.0-dev-432b303'
   exit: 0 — one 10,849-byte application/vnd.tessera.graph.tar+gzip layer
   ```

8. Fix the measurement time and calculate the age of the newest full package:

   ```text
   date --utc +%Y-%m-%dT%H:%M:%SZ
   exit: 0 — 2026-09-01T18:10:26Z

   UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync python -c 'from datetime import datetime; measured=datetime.fromisoformat("2026-09-01T18:10:26+00:00"); full=datetime.fromisoformat("2026-07-07T12:27:23+00:00"); delta=measured-full; print(f"seconds={delta.total_seconds():.0f}"); print(f"days={delta.total_seconds()/86400:.9f}"); print(f"duration={delta.days}d {delta.seconds//3600:02d}:{(delta.seconds%3600)//60:02d}:{delta.seconds%60:02d}")'
   exit: 0 — seconds=4858983; days=56.238229167; duration=56d 05:43:03
   ```

## Semantic-authority and classification commands

These commands were read-only and supplied context or scope classification;
they did not supply package presence or push timestamps.

| Command | Exit | Use |
|---|---:|---|
| `sed -n '1,999p' docs/plans/graph-destructive-command-safety.html` (preceded by `wc -l`) | 0 | Read the complete live plan from this checkout. |
| `sed -n '230,340p' docs/plans/graph-destructive-command-safety.html` | 0 | Re-read the untruncated deliverable and decision section. |
| `sed -n '1,260p' docs/evidence/graph-operational-safety/offsite-currency.md` | 0 | Read the prior TCV-only measurement and its exact authority boundary. |
| `sed`/`rg` reads of `imas_codex/graph/ghcr.py`, `imas_codex/cli/graph/registry.py`, `imas_codex/cli/release.py`, `docs/architecture/graph.md`, and related tests | 0 | Locate the canonical package-name and release-variant contracts. |
| `git log --all --oneline -S 'imas-codex-graph-imas' -- imas_codex docs tests` plus `git show` of the historical package-name implementation | 0 | Prove that `-imas` is the legacy DD-only package. |
| `git show e4bced8a…:imas_codex/graph/ghcr.py`, `git show e4bced8a…:imas_codex/cli/release.py`, and before/after `git show` around `3d14b9ef` | 0 | Classify full, current DD-only, legacy DD-only, and facility-suffixed packages at their source revisions. |
| `git status --short` and `git stash list` | 0 | Confirm the worker began from a clean scoped checkout with no stash. |
| `test ! -e docs/evidence/graph-operational-safety/offsite-inventory.md` | 0 | Confirm the evidence path was new before writing. |

The live plan was also read through Reckon's typed `read_plan` interface at
version 19 before the filesystem read; that tool call succeeded and reported
this node in flight. It is not a shell command and therefore has no process exit
code.
