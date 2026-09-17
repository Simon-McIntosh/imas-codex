<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="docs-project"       content="imas-codex">
  <meta name="reckon-type"        content="evidence">
  <meta name="plan-slug"          content="catalog-review-surface-plan-figure-audit">
  <meta name="plan-title"         content="Catalog Review Surface: Headline Figures Re-measured">
  <meta name="plan-summary"       content="The plan's asserted quantities re-taken against the live graph and the current tree, each with a verdict and the instrument that produced it.">
  <meta name="plan-status"        content="active">
  <meta name="plan-owner"         content="Simon McIntosh">
  <meta name="plan-evidence-for"  content="catalog-review-surface">
  <meta name="plan-tags"          content="standard-names,west,catalog-review">
  <title>Catalog Review Surface: Headline Figures Re-measured | imas-codex</title>
  <link rel="stylesheet" href="/_shared/foundation.css">
  <link rel="stylesheet" href="/_shared/dashboard.css">
</head>
<body>
  <main class="plan-doc">
    <h2 id="result">Result</h2>
    <p>
      Twelve quantities asserted by <code>catalog-review-surface</code> were re-taken on
      2026-09-17 against the live graph, the current tree, the catalog fork and the archive
      directory. Five are <strong>current</strong>, five are <strong>stale</strong>, and two are
      <strong>drifted</strong>. Nothing was unmeasurable: every query answered inside the
      ten-second ceiling, all at exit 0.
    </p>
    <p>
      The drift is one-directional and expected &mdash; the graph and the archive kept growing
      after the plan froze its numbers, so counts moved upward. The stale rows are a different
      thing: each is a claim the plan made about a defect or a request state that has since been
      repaired. Two of them (<code>graph_pull</code> taking no destination, and the newest
      archive) describe work the plan's own later comments record as landed, and one (the "all
      zero" census) is half-overtaken because a writer for one of the four fields landed on
      2026-09-08.
    </p>

    <h2 id="table">The asserted quantities</h2>
    <table>
      <thead>
        <tr><th>#</th><th>Asserted quantity, verbatim</th><th>Where</th><th>Re-measured 2026-09-17</th><th>Verdict</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>"a controlled census over 4,666 <code>StandardName</code> rows"</td>
          <td>&sect;4a</td>
          <td><strong>5,130</strong> <code>StandardName</code> nodes</td>
          <td>drifted</td>
        </tr>
        <tr>
          <td>2</td>
          <td>"<code>catalog_approved_at</code>, <code>catalog_pr_number</code>, <code>catalog_merge_commit_sha</code> and <code>exported_at</code> <strong>all zero</strong>"</td>
          <td>&sect;4a</td>
          <td>of 5,130 rows the first three are null on <strong>5,130 of 5,130</strong>; <code>exported_at</code> is set on <strong>221</strong></td>
          <td>stale</td>
        </tr>
        <tr>
          <td>3</td>
          <td>"Fork pull request 3 (<code>review/v0.3.0rc1+west-task-2e</code>) is open"</td>
          <td>&sect;4a</td>
          <td>PR 3 state <strong>MERGED</strong>, merged <code>2026-09-01T20:39:07Z</code></td>
          <td>stale</td>
        </tr>
        <tr>
          <td>4</td>
          <td>"The reviewer guide states the editable-versus-machine-owned distinction at commit <code>e9d3044</code>"</td>
          <td>&sect;4a</td>
          <td><code>e9d3044</code> present: "docs(review): explain reviewer-editable fields"</td>
          <td>current</td>
        </tr>
        <tr>
          <td>5</td>
          <td>"the guard job lives in <code>.github/workflows/validate.yml</code> at commit <code>dd70ba6</code>"</td>
          <td>&sect;4a</td>
          <td><code>dd70ba6</code> present: "ci(catalog): guard machine-owned review fields"; <code>validate.yml</code> tracked</td>
          <td>current</td>
        </tr>
        <tr>
          <td>6</td>
          <td>"fork pull request 4 edits the machine-owned <code>unit</code> and its <code>review-edit-guard</code> check <strong>fails</strong>, while pull request 5 ... the same check <strong>passes</strong>"</td>
          <td>&sect;4a</td>
          <td><code>gh pr checks 4</code>: <code>review-edit-guard</code> <strong>fail</strong>; <code>gh pr checks 5</code>: <code>review-edit-guard</code> <strong>pass</strong>; both PRs CLOSED</td>
          <td>current</td>
        </tr>
        <tr>
          <td>7</td>
          <td>"<code>graph_pull</code> at <code>cli/graph/registry.py:1101-1133</code> takes no destination"</td>
          <td>&sect;4b</td>
          <td><code>registry.py:1102</code> declares <code>@click.argument("target")</code>; <code>:1163</code> calls the shared target guard; <code>:1384</code> forwards the target to the local load</td>
          <td>stale</td>
        </tr>
        <tr>
          <td>8</td>
          <td>"The newest verified full archive is 2,458,442,456 bytes at 2026-09-01T14:52:53+02:00"</td>
          <td>&sect;4b</td>
          <td>newest archive across <code>BACKUPS_DIR</code> and <code>EXPORTS_DIR</code> is <code>imas-codex-graph-dev-6e9f34e-20260914T143909Z.tar.gz</code>, <strong>2,482,184,930 bytes</strong>, mtime 2026-09-14T16:41:49+02:00</td>
          <td>stale</td>
        </tr>
        <tr>
          <td>9</td>
          <td>"2,451,894,300 bytes sealed at 2026-09-05T12:15:36Z"</td>
          <td>f-crs-checkpoint-instrument</td>
          <td><code>backups/imas-codex-graph-dev-5b5faf1-20260905T121249Z.tar.gz</code> is <strong>2,451,894,300 bytes</strong>, mtime 2026-09-05T14:15:36+02:00 = 12:15:36Z</td>
          <td>current</td>
        </tr>
        <tr>
          <td>10</td>
          <td>"a 4,748-byte offsite trial dump"</td>
          <td>&sect;4b</td>
          <td><code>backups/offsite-trial-dev-c987057-20260902T101502Z.dump</code> is <strong>4,748 bytes</strong></td>
          <td>current</td>
        </tr>
        <tr>
          <td>11</td>
          <td>"the same 4,937 <code>StandardName</code> identities ... 1,628,593 nodes"</td>
          <td>f-crs-checkpoint-instrument</td>
          <td><strong>5,130 and 1,639,351</strong>; DDVersion <code>4.1.1</code> still <code>is_current</code></td>
          <td>drifted</td>
        </tr>
        <tr>
          <td>12</td>
          <td>"261 accepted human edits with 0 actors pre-deployment"</td>
          <td>c-run-r-20260901T1231-revieweractor</td>
          <td><code>edit_origin='human'</code> on <strong>572</strong> rows; <code>catalog_reviewer_actor</code> non-null on <strong>1</strong></td>
          <td>stale</td>
        </tr>
      </tbody>
    </table>
    <p>
      <strong>Verdict counts:</strong> current <strong>5</strong>, stale <strong>5</strong>,
      drifted <strong>2</strong>. <strong>5 + 5 + 2 = 12</strong> rows.
    </p>

    <h2 id="first-beat">The plan's first unstarted beat</h2>
    <p>
      <strong>Still required.</strong> The first beat with no completed work is deliverable three,
      "An exercised approval fold-back carrying an accepted edit and a pipeline-refused edit,
      with the refusal visible to the reviewer rather than silent" (&sect;4). The measurement that
      decides it is the graph's own record of the 2026-09-08 rehearsal. The rehearsal request
      exists: PR 17, "Exercise approval name outcomes"
      (<code>rehearsal/approval-fold-back-20260908</code>), merged 2026-09-08T09:36:05Z, carrying
      <code>alfven_time</code> to <code>alfven_transit_time</code> (meant to be accepted) and
      <code>atomic_count</code> to <code>temperature_at_plasma_boundary</code> (meant to be
      refused).
    </p>
    <p>
      On the graph that beat has left nothing at all:
    </p>
    <ul>
      <li><code>alfven_time</code> is <strong>unchanged</strong> &mdash; status <code>draft</code>,
          <code>name_stage accepted</code>, no <code>edit_origin</code>, no
          <code>catalog_reviewer_*</code> fields, <code>origin pipeline</code>;
          <code>alfven_transit_time</code> does not exist as a node;</li>
      <li><strong>no</strong> <code>StandardNameChange</code> row records a fold-back on the
          rehearsal identities after 2026-09-07. Corrected 2026-09-17 after independent review:
          this row originally read &ldquo;no <code>StandardNameChange</code> row exists after
          2026-09-07&rdquo;, which is false graph-wide &mdash; the window holds
          <strong>2,889</strong> rows, led by <code>reconcile_catalog_edit_origin</code> at 2,096
          and <code>remove_derived_parent</code> at 284. The original clause compared
          <code>changed_at</code> against a <em>string</em> literal, and because the property is a
          <code>DateTime</code> the predicate evaluated to null and filtered every row out, so the
          zero was an unaimed instrument rather than a measurement. Scoped correctly, exactly
          <strong>two</strong> post-merge rows touch these identities &mdash; a
          <code>reconcile_catalog_edit_origin</code> bookkeeping row on <code>alfven_time</code> at
          2026-09-08T11:55:49Z and a <code>correct_published_cut_null_origins</code> row on
          <code>atomic_count</code> at 2026-09-09T10:05:42Z &mdash; neither of which is a
          fold-back, against a positive control of 21 rows naming these identities overall. The
          section's conclusion is unchanged: the fold-back did not run;</li>
      <li><code>catalog_approved_at</code>, <code>catalog_pr_number</code> and
          <code>catalog_merge_commit_sha</code> are null on <strong>all 5,130 rows</strong>, so no
          accepted edit has ever been recorded with its pull request.</li>
    </ul>
    <p>
      The who-changed-it half of deliverable five is still owed too, and one row shows exactly
      why: <code>net_power_due_to_ion_cyclotron_heating</code> is the only identity carrying
      <code>catalog_reviewer_actor</code> ("Simon McIntosh"), and its
      <code>catalog_pr_number</code> is <strong>null</strong>. The actor can be written while the
      traceability receipt's pull-request half cannot, so no traceability receipt is readable from
      the graph today. The fold-back did not run on 2026-09-08, which the plan already records
      (c-run-r-20260908T092429674063): the isolated substrate would not start from the verified
      archive. That is the beat's remaining prerequisite, not its remainder.
    </p>

    <h2 id="effort">Effort restated in worker-hours</h2>
    <p>
      Of the plan's five deliverables, one and two are complete and proven live; three, four and
      five are unstarted (above). The plan's own estimate is <code>6.0</code> worker-hours
      (<code>plan-effort-hours</code>), so the remaining three deliverables account for
      <strong>roughly 3.0 to 4.0 worker-hours</strong>: the fold-back exercise and its receipt on
      purpose-made identities, the unwind proof, and the traceability check. This is a derived
      figure from the plan's own number, not a re-measurement of a clock.
    </p>
    <p>
      One boundary is real and should not be budgeted as worker time: opening, commenting on and
      closing a request on the catalog fork is a write outside an <code>imas-codex</code>
      worktree, so the live-fork half belongs to a session with that repository in scope
      (c-run-r-20260908T092515122308, c-run-r-20260908T102450494864).
    </p>

    <h2 id="method">What was read, and how</h2>
    <p>
      Every graph figure comes from a read-only Cypher query over a <code>GraphClient()</code>
      opened against the active symlink (<code>graph_name codex</code>,
      <code>GraphClient()</code> resolved <code>bolt://98dci4-gpu-0002:7687</code>). No command
      in this audit started, stopped, loaded, cleared, pulled or switched a graph, and no node,
      relationship or property was written. Code claims are answered by reading the line the plan
      names at the current revision. Request state is from <code>gh</code> against the fork.
      File claims are from <code>stat</code> / <code>ls</code> on the path the plan names.
      Query output is captured to the log files named in the manifest
      (<code>logs/graph-audit.log</code>, <code>logs/graph-audit-2.log</code>,
      <code>logs/graph-audit-3.log</code>, <code>logs/graph-audit-5.log</code>,
      <code>logs/code-registry.log</code>, <code>logs/files.log</code>,
      <code>logs/catalog-repo.log</code>, <code>logs/artifacts.log</code>,
      <code>logs/gh-prs.log</code>, <code>logs/gh-prs2.log</code>), all at exit 0.
    </p>
  </main>
</body>
</html>