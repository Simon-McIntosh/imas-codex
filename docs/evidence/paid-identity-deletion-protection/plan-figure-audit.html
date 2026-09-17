<meta name="docs-project" content="imas-codex">
<meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="paid-identity-deletion-protection">
<meta name="plan-status" content="active">
<meta name="plan-title" content="Paid identity deletion protection &mdash; headline-figure audit">
<meta name="plan-evidence-for" content="paid-identity-deletion-protection">

# Headline-figure audit — `paid-identity-deletion-protection`

Every quantity below is quoted verbatim from the plan's own HTML and re-measured
against the live graph (`codex`) and the current worktree source. Read-only: no
graph mutation was performed, and no query in this audit writes.

Base revision: `3f8c06fcef82b8a769ee5ec5d81ae83b4fe3c5c5`.

## Verdict summary

| Verdict | Rows |
|---|---|
| current | 4 |
| drifted | 11 |
| stale | 1 |
| **total rows** | **16** |

Four of sixteen headline quantities still read true. Eleven have drifted — and
by no small amount: the cost ledger only augments, so every spend and row-count
figure in the plan is now a floor, understated by 23–40%. The single `stale`
row is a code pointer that has rotted by 64 lines.

## The audit

| # | Plan § | Asserted (verbatim from the plan) | Re-measured (2026-09-17) | Verdict | Instrument |
|---|---|---|---|---|---|
| 1 | §1 | "Between 99 and 107 seconds later, a structural cleanup pass deleted 78 of the identities … and 93 in total" | 93 `StandardNameChange` rows / 93 distinct `to_name` in the typed window `2026-09-08T11:57:00Z`–`11:57:37Z` | current | `D_datetime_typed_window` (typed bound; see the method note) |
| 2 | §1 | "93" identities cost "$43.12" to destroy (cost pass E) | 1,003 `LLMCost` rows / $50.761691 touching those 93 names | drifted (+28% rows, +18% spend) | `C_spend93` | 
| 3 | §2 | "protects 75 of 93" — the count that did *not* pay | 82 of the 93 names carry at least one `LLMCost` row; 11 carry none | drifted | `R_spend_rows_restore_set` |
| 4 | §2b | "Every still-absent deletion by this path, all time: 491 identities, 1,894 cost rows, $100.47" | 453 absent of 539; 2,660 cost rows / $135.896586 | drifted (+40% rows, +35% spend) | `C_spend539` |
| 5 | §5 | "This path has removed 539 identities all time" | 539 distinct `to_name`, 2,644 rows, operation `remove_derived_parent` | current | `S1_remove_derived_parent_alltime` |
| 6 | §7 | "412 standard names are alive at `origin='derived'`" | 428 | drifted (+3.9%) | `S7_origin_distribution` |
| 7 | §7 | derived: "397 sit at `name_stage` accepted or approved" | 401 accepted, 0 approved | drifted | `S7_origin_by_name_stage` |
| 8 | §7 | derived: "334 carry `docs_stage='accepted'` with non-empty documentation" | 412 | drifted (+23%) | `S7_derived_docs_accepted_nonempty` |
| 9 | §7 | derived: "$116.91 of paid work across 2,339 `LLMCost` rows" | 3,189 rows / $160.710449 over 361 derived names (edge measure) | drifted (+36% rows, +38% spend) | `C_derived_spend_edges` |
| 10 | §6 | "zero are present in the graph; of the 67-name RESTORE set, zero are present" (2026-09-08T19:00Z) | 33 of the 93 resolve by identity; 29 of the 78-name restore set resolve in the graph (lower bound — spellings drift) | drifted | `C_present93`, `R_present_in_restore_set` |
| 11 | §8 | "zero rows hold `name_stage='approved'` … `filter_protected` returns the empty set" | 0 approved rows live; `filter_protected` (`protection.py:238`) still keys `protected_names` on `'approved'` | current | `S8_name_stage_approved` |
| 12 | §3a | "the guard is called at `graph_ops.py:3672`" | the call is at `graph_ops.py:3736` (+64 lines); `graph_ops.py` now carries two guarded call sites (`:3736`, `:5846`) | **stale** | code read (`/tmp/pidp-audit/code-read.log`) |
| 13 | §2b | "no code path deletes an `LLMCost` node" | zero `DETACH DELETE` statements naming cost/LLMCost under `imas_codex/standard_names/` | current | code read |
| 14 | §3 | "32,203 `LLMCost` `FOR_STANDARD_NAME` edges were materialised" | 34,547 edges / 34,131 cost rows | drifted (+7.3%) | `D_for_standard_name_edges` |
| 15 | §4 | archive table: "StandardName 5,078 (live now 5,048)" | live names 5,130 | drifted (live half; archive half unmeasured) | `CO_standard_names` |
| 16 | §8 | source distribution "bound 5,441 … unbound 4,459" | bound 5,393 / unbound 4,626 of 10,019 | drifted | `S8_source_bound_unbound` |

### Unmeasured quantities

| Plan § | Asserted | Why unmeasured |
|---|---|---|
| §3a | "a gap-only run leaves 21 sources with no producing name" | requires a live pipeline run against the paid fleet; the node's fence forbids mutation and the time fence forbids a paid run |
| §4 | archive dump label counts (the tarball side of the table) | the tarball read is heavy local work and does not fit the time fence; the live half was measured instead (row 15) |
| §7 | the WEST cut "161 / 42 / 4" | needs the 208-name manifest read; out of budget |

## Drift, plotted

Every quantity here is a *population* or a *ledger sum*, so the direction of
drift is one of two shapes and the shapes are diagnostic. The populations are
roughly stable because the name surface is roughly stable; the ledger sums climb
because nothing ever subtracts from them. That asymmetry is why the plan's cost
figures rot faster than its count figures, and it is the reason a rescope should
re-derive spend but may trust the population counts to within a few percent.

<figure>
  <svg viewBox="0 0 780 300" width="100%" role="img"
       aria-label="Percent change from asserted to re-measured for eleven headline quantities. Ledger sums and cost row counts grew 23 to 40 percent; the absent populations fell.">
    <g font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#1a1a1c">
      <line x1="150" y1="18" x2="150" y2="250" stroke="#8a8a8f" stroke-width="1"/>
      <text x="150" y="12" text-anchor="middle" font-size="10" fill="#54545a">0%</text>

      <rect x="150" y="24" width="7" height="12" fill="#4a6fa5"/>
      <text x="163" y="34">+2% live names 5048&#8594;5130</text>

      <rect x="150" y="44" width="182" height="12" fill="#b3261e"/>
      <text x="338" y="54">+40% all-time `LLMCost` rows 1894&#8594;2660</text>

      <rect x="150" y="64" width="159" height="12" fill="#b3261e"/>
      <text x="315" y="74">+35% destroyed spend $100.47&#8594;$135.90</text>

      <rect x="150" y="84" width="18" height="12" fill="#4a6fa5"/>
      <text x="174" y="94">+4% live derived 412&#8594;428</text>

      <rect x="150" y="104" width="105" height="12" fill="#b3261e"/>
      <text x="261" y="114">+23% derived docs-accepted 334&#8594;412</text>

      <rect x="150" y="124" width="169" height="12" fill="#b3261e"/>
      <text x="325" y="134">+38% derived spend $116.91&#8594;$160.71</text>

      <rect x="150" y="144" width="163" height="12" fill="#b3261e"/>
      <text x="319" y="154">+36% derived cost rows 2339&#8594;3189</text>

      <rect x="150" y="164" width="125" height="12" fill="#b3261e"/>
      <text x="281" y="174">+28% 93-cohort cost rows 785&#8594;1003</text>

      <rect x="150" y="184" width="33" height="12" fill="#4a6fa5"/>
      <text x="189" y="194">+7% `FOR_STANDARD_NAME` edges 32203&#8594;34547</text>

      <rect x="11" y="204" width="159" height="12" fill="#2e7d32"/>
      <text x="5" y="214" text-anchor="end" fill="#2e7d32">&#8722;36% still absent of 93 &#8594; 60 of 93</text>

      <rect x="116" y="224" width="34" height="12" fill="#2e7d32"/>
      <text x="110" y="234" text-anchor="end" fill="#2e7d32">&#8722;8% all-time absent 491&#8594;453</text>

      <text x="150" y="262" font-size="10" fill="#54545a">red — the asserted figure understates today's surface (ledger growth)</text>
      <text x="150" y="276" font-size="10" fill="#54545a">green — the asserted figure overstates it (recovery that has landed; bar drawn left of zero)</text>
      <text x="150" y="290" font-size="10" fill="#54545a">blue — within 10% of the asserted value</text>
    </g>
  </svg>
  <figcaption>Percent drift of the plan's headline quantities from asserted to
  re-measured. Bars left of the axis are quantities the plan overstates.</figcaption>
</figure>

## The first unstarted beat

The plan's stages run §1 (the incident) → §3 (the guard) → §4 (restore) → §5/§6
(disposition) → §7 (the `origin` field) → §8 (the source binding). Classifying
each against the current graph and source:

| Beat | Classification | Basis |
|---|---|---|
| §1 incident + §3 guard | landed |
| §4 restore | partially landed, not unstarted | 33 of 93 identities resolve; 60 absent |
| §5/§6 disposition | landed (answered) | the census plus the spend rule closed at 67/11/0 |
| §7a stage 1 (leaders reclassified) | landed | 0 derived rows hold a direct `dd:PRODUCED_NAME` producer (`S7_derived_with_direct_dd_producer = 0`) |
| **§7a stage 2 — "Repoint every reader at the source binding"** | **still-required** | the `origin` axis is live and load-bearing: 428 names carry `origin='derived'`, and 75 lines under `imas_codex/standard_names/` still read it |
| §7a stages 3–4, §8 | still-required | downstream of stage 2 |

**First unstarted beat: §7a stage 2.**

The test is not "does the field exist" — it does — but "has any reader been
moved off it". Nobody has: the population it labels is non-zero and growing, and
the readers are still reading. The field was never dropped, so the plan's own
drop step cannot yet have run.

## Remaining effort, restated

The plan declares 14.0 worker-hours. Restated against the measured surface:

| Remaining work | Worker-hours | Basis |
|---|---|---|
| §4 restore of the ~49 still-absent identities | 6–8 | 23 identities were restored for $3.59; the remainder is dominated by source-less identities, one route per identity class |
| §7a stages 2–4: repoint readers, gate, drop `origin` | 4–6 | a 34-file reader census, a protection gate, and a field removal that must land together |
| defender: the two absent `HAS_PARENT` edges (separate authorisation) | 1–2 | needs a signed apply on a quiet graph |
| §7 reclassification remainder | 0 | measured: 0 derived rows with a direct `dd:` producer |
| **total** | **11–16** | against 14.0 declared |

The 14.0 figure survives, but its centre of mass has moved: less is owed to the
restore and more to the `origin` fold than the plan's budget implies.

## Method and logs

Each measurement is a bounded read against `bolt://98dci4-gpu-0002:7687`
(`graph_name=codex`) reached through the login-node-local tunnel — the placement
the repo's AGENTS.md permits for live-graph work, with a named row set and a
typed bound, and no heavy local compute alongside.

**The instrument trap, recorded because it nearly produced a false `stale`.**
The 2026-09-08 window queries first used a string bound
(`c.changed_at >= '2026-09-08'`) against a native Cypher datetime property,
which returned **0 rows for the whole day**. Read naively that says the incident
never happened. Re-issued with a typed bound the same window returns exactly 5
rows / 93 names. Every window query in this audit therefore carries a typed
`datetime(...)` bound, and row 1 is validated by a positive control: the same
predicate over all time returns 2,644 rows / 539 distinct names, which matches
row 5 independently.

| Log | Contents | Exit |
|---|---|---|
| `/tmp/pidp-audit/discovery.log` | property-key discovery | 0 |
| `/tmp/pidp-audit/main.log` | populations, distributions, delete-path census | 0 |
| `/tmp/pidp-audit/diag.log` | window-instrument diagnosis and positive control | 0 |
| `/tmp/pidp-audit/spend.log` | spend attribution over both cohorts | 0 |
| `/tmp/pidp-audit/restore.log` | restore-set presence | 0 |
| `/tmp/pidp-audit/code-read.log` | source reads for rows 12, 13 and the beat census | 0 |

Exact query text: `audit_main.py`, `audit_diag.py`, `audit_spend.py` and
`audit_restore.py` alongside those logs.

## Follow-ons (outside this node's fence)

1. `docs/evidence/sn-lifecycle-integrity/unshielded-identity-deletions.md` is
   internally inconsistent: its top `## Result` block still reads
   **32 RESTORE / 24 CORRECTLY REMOVED / 22 UNDETERMINED** while its later
   sections state the authoritative **67 / 11 / 0**. The plan quotes the 67/11/0
   disposition, which the file's own later text supports — the summary block is
   the stale half and that file owns the fix.
2. The plan's §3a line pointer `graph_ops.py:3672` is stale (row 12).
3. The plan's §6a prose says "nine of the fifteen" carry spend; the
   undispositioned census records 11 RESTORE of fifteen. One of the two is wrong.
4. The plan carries both $100.47 and $100.31 for the same all-time spend.
5. The figure for this evidence had to be embedded inline because the write fence
   is three paths and `docs/figures/` is not among them; a later node with a wider
   fence should lift it out.