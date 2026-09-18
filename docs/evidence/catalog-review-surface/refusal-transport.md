<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="docs-project"       content="imas-codex">
  <meta name="reckon-type"        content="evidence">
  <meta name="plan-slug"          content="refusal-transport">
  <meta name="plan-title"         content="Carrying a Fold-Back Refusal Onto the Request">
  <meta name="plan-summary"       content="The refusal path, the call site added, the existing transport it reuses, the reproduction at the base revision, and the two sibling tests this change newly breaks.">
  <meta name="plan-status"        content="active">
  <meta name="plan-owner"         content="Simon McIntosh">
  <meta name="plan-tags"          content="standard-names,catalog-review,west">
  <meta name="plan-evidence-for"  content="catalog-review-surface">
  <title>Carrying a Fold-Back Refusal Onto the Request | imas-codex</title>
  <link rel="stylesheet" href="/_shared/foundation.css">
</head>
<body>
  <p>
    The reviewer's only surface is the pull request. When the approval fold-back
    refuses an edit, the catalog has already merged and the reviewer has moved on,
    so a refusal that stays in the run log is indistinguishable from an approval:
    the request reads as accepted while the graph disagrees with the catalog. This
    record states the defect as measured, the call site that closes it, the
    existing transport it reuses, and — in the last section — the two sibling
    tests the change newly breaks.
  </p>

  <h2 id="defect">The defect, reproduced at the base revision</h2>
  <p>
    At base <code>6c90ee1e68edde3d57a7e08aadb3c79900875f59</code>, nothing in the
    package posts anything to a pull request. The only references to the comments
    endpoint are reads:
  </p>
  <pre><code>6c90ee1e… imas_codex/standard_names/promote.py:1917   GET /issues/{n}/comments?per_page=100
6c90ee1e… imas_codex/standard_names/review_triage.py:167  GET /issues/{n}/comments
=== POST call sites at base === (none; exit 1)</code></pre>
  <p>
    So a refusal raised above is recorded in <code>ApprovalReport.promotion_refused</code>
    and <code>ApprovalReport.contested</code> and reaches no reviewer.
  </p>

  <h2 id="change">The call site, and the transport it reuses</h2>
  <p>
    The notice is written from inside <code>run_approval</code>, after the report is
    complete and before the catalog correction is composed, so it carries every
    refusal the run recorded. It reuses
    <code>imas_codex.graph.ghcr.github_api_call</code> — the same client the
    pre-existing <code>_pull_request_call</code> (<code>promote.py:1459</code>, GET) uses —
    with <code>POST /repos/{repo}/issues/{number}/comments</code>.
  </p>
  <svg viewBox="0 0 760 190" width="100%" style="max-width:760px" role="img"
       aria-label="The refusal path: run_approval records a refusal, notify_refusal_on_pull_request builds a body, and github_api_call posts it to the request.">
    <rect x="8" y="20" width="170" height="60" rx="8" fill="#eef2f7" stroke="#5b6b7f" stroke-width="1.5"/>
    <text x="93" y="44" text-anchor="middle" font-size="12" font-weight="600">run_approval</text>
    <text x="93" y="62" text-anchor="middle" font-size="11">records the refusal</text>

    <rect x="215" y="20" width="200" height="60" rx="8" fill="#eef7ee" stroke="#4f7a4f" stroke-width="1.5"/>
    <text x="315" y="44" text-anchor="middle" font-size="12" font-weight="600">notify_refusal_on_pull_request</text>
    <text x="315" y="62" text-anchor="middle" font-size="11">promote.py:1524</text>

    <rect x="452" y="20" width="150" height="60" rx="8" fill="#f7f0ee" stroke="#8a5b4f" stroke-width="1.5"/>
    <text x="527" y="44" text-anchor="middle" font-size="12" font-weight="600">github_api_call</text>
    <text x="527" y="62" text-anchor="middle" font-size="11">existing client</text>

    <rect x="639" y="20" width="113" height="60" rx="8" fill="#f2eef7" stroke="#6b4f8a" stroke-width="1.5"/>
    <text x="695" y="44" text-anchor="middle" font-size="12" font-weight="600">the request</text>
    <text x="695" y="62" text-anchor="middle" font-size="11">POST comment</text>

    <line x1="178" y1="50" x2="213" y2="50" stroke="#5b6b7f" stroke-width="2" marker-end="url(#a)"/>
    <line x1="415" y1="50" x2="450" y2="50" stroke="#5b6b7f" stroke-width="2" marker-end="url(#a)"/>
    <line x1="602" y1="50" x2="637" y2="50" stroke="#5b6b7f" stroke-width="2" marker-end="url(#a)"/>
    <defs>
      <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path d="M0,0 L10,5 L0,10 z" fill="#5b6b7f"/>
      </marker>
    </defs>
    <text x="8" y="120" font-size="11" fill="#444">absent at base — every refusal was silent to the reviewer</text>
  </svg>

  <h2 id="evidence">The evidence that could have failed</h2>
  <ul>
    <li>
      <strong>Named gate, green.</strong>
    <code>tests/standard_names/test_promotion_refusal_reaches_the_request.py</code>
      — 2 passed, exit 0 (<code>logs/refusal-gate.log</code>). The first case drives one
      refused promotion and asserts exactly one POST, to
      <code>/repos/x/y/issues/7/comments</code>, whose body carries the refusal reason
      and the refused identity. The second drives a successful promotion and asserts the
      transport was never called, so the writer is not unconditional. Both mock
      <code>github_api_call</code>; neither needs a live graph or a network.
    </li>
    <li>
      <strong>The call-site-absent probe.</strong> With
      <code>notify_refusal_on_pull_request</code> suppressed, the same assertions measure
      <em>refused rows: 1 / transport calls: 0</em> and fail — the gate is not vacuous
      (<code>logs/callsite-absent-probe.log</code>).
    </li>
    <li>
      <strong>What the change touches.</strong> <code>git diff --stat</code> on
      <code>promote.py</code>: 96 insertions, 0 deletions.
    </li>
    <li>
      <strong>Exit status recorded.</strong> The gate run above is the node's own
      measure; the surrounding subtree run is in the next section, and it is not green.
    </li>
  </ul>

  <h2 id="delta">The subtree run, and the two failures this change added</h2>
  <p>
    The subtree covering the changed module
    (<code>tests/standard_names/</code>, default markers) ran on a debug partition:
    <strong>3 failed, 7443 passed, 11 skipped, 326 deselected</strong>, exit 1
    (<code>logs/focused-suite.log</code>).
  </p>
  <p>
    Judged by delta, not by absolute green. The same three tests were re-measured at
    base <code>6c90ee1e</code> from a scratch worktree at that revision:
    <strong>1 failed, 5 passed</strong> (<code>logs/base-three-failures-revalidated.log</code>,
    which prints the module path it resolved so the reader can see the base tree was
    the one exercised.
  </p>
  <table>
    <thead>
      <tr><th>Test</th><th>At base</th><th>With the change</th><th>Attribution</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>test_error_siblings …orphans_error_siblings</td>
        <td>failed</td><td>failed</td><td>pre-existing</td>
      </tr>
      <tr>
        <td>test_sn_approve_tag …contested_entry_is_removed…</td>
        <td>passed</td><td>failed</td><td><strong>added by this change</strong></td>
      </tr>
      <tr>
        <td>test_sn_approve_tag …undo_restores_catalog_main…</td>
        <td>passed</td><td>failed</td><td><strong>added by this change</strong></td>
      </tr>
    </tbody>
  </table>

  <h2 id="required">The fix that is outside this node's fence</h2>
  <p>
    Both added failures are the same defect in a sibling test harness, and the failure
    text names it exactly:
  </p>
  <pre><code>standard_names.test_sn_approve_tag.RealTransportAttempted:
test opened a real connection to https://api.github.com/repos/fork/catalog/issues/7/comments</code></pre>
  <p>
    <code>tests/standard_names/test_sn_approve_tag.py</code> guards every test with a
    fixture that raises on any real HTTP (<code>no_real_transport</code>, line 64). Its
    <code>_fold_additive_batch</code> helper (line 185) drives <code>run_approval</code>
    past a refusal and patches the readers, the scorer, the applier, and
    <code>mark_catalog_name_approved</code> — but never the write transport, because
    until this change the fold-back made no write. The guard is doing its job: it
    caught a network write the test did not know about.
  </p>
  <p>
    The repair is one mocked client in that helper, for which the file already carries
    the house pattern at line 493
    (<code>patch("imas_codex.graph.ghcr.github_api_call", fake_call)</code>). That file
    is <strong>not</strong> in this node's write fence, so it is reported here and in
    the manifest rather than edited. Until it is applied, merging this change turns
    two green tests red.
  </p>
  <p>
    Nothing else in the subtree moved: the third failure is a stale assertion in
    <code>test_error_siblings.py</code> about a Cypher string that no longer carries
    <code>sn.quarantine_reason</code>, and it fails identically at base.
  </p>
</body>
</html>
  </body>
</html>