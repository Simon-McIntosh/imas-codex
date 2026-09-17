<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="docs-project" content="imas-codex">
  <meta name="reckon-type" content="evidence">
  <meta name="plan-slug" content="unbound-source-backlog-figure-audit">
  <meta name="plan-evidence-for" content="unbound-source-backlog">
  <meta name="plan-title" content="Unbound source backlog &mdash; headline figures re-measured">
  <title>Unbound source backlog figure audit | imas-codex</title>
  <link rel="stylesheet" href="/_shared/foundation.css">
  <link rel="stylesheet" href="/_shared/dashboard.css">
</head>
<body>
<main class="plan-doc">
  <h1>Unbound source backlog &mdash; headline figures re-measured</h1>

  <p>Every figure below was re-read from the live graph read-only at worktree
  revision <code>5ea0d73b441b6991fc29b9dbb03e13bc5b63b43c</code> on 2026-09-17, with no graph mutation.
  The instrument was <code>imas_codex.graph.client.GraphClient.from_profile()</code>
  running named-cohort Cypher, one query per quantity, output captured to
  <code>census.log</code>, <code>census3.log</code> and <code>census4.log</code> in the
  run directory (paths in the manifest). Positive controls are stated where a zero
  would otherwise be ambiguous: the source census returned 10,019 rows and the
  <code>HAS_STANDARD_NAME</code> relation returned 5,092 edges, so neither the node
  label nor the relationship name was mis-typed into a silent empty result.</p>

  <h2>Verdict table</h2>

  <table>
    <caption>The plan's asserted quantities against their re-measured values.
    Verdicts: <strong>current</strong> (matches), <strong>drifted</strong> (value moved;
    claim still meaningful), <strong>stale</strong> (the asserted mechanism no longer
    holds).</caption>
    <thead>
      <tr><th>#</th><th>Asserted in plan (verbatim)</th><th>Plan</th><th>Re-measured</th><th>Verdict</th><th>Query / code read</th></tr>
    </thead>
    <tbody>
      <tr><td>1</td><td>&ldquo;Of <strong>9,900</strong> such rows&rdquo; &mdash; standard name sources</td><td>9,900</td><td><strong>10,019</strong></td><td>drifted (+119)</td><td><code>MATCH (s:StandardNameSource) RETURN count(s)</code></td></tr>
      <tr><td>2</td><td>&ldquo;<strong>5,441 are bound</strong> to a name through a <code>PRODUCED_NAME</code> edge&rdquo;</td><td>5,441</td><td><strong>5,393 sources</strong> (5,493 edges)</td><td>drifted (&minus;48)</td><td><code>MATCH (s)-[:PRODUCED_NAME]-&gt;() RETURN count(DISTINCT s), count(*)</code></td></tr>
      <tr><td>3</td><td>&ldquo;<strong>4,459 are not</strong>&rdquo; unbound</td><td>4,459</td><td><strong>4,626</strong></td><td>drifted (+167)</td><td>derived: 10,019 &minus; 5,393</td></tr>
      <tr><td>4</td><td><code>composed</code> bound</td><td>3,053</td><td><strong>3,064</strong></td><td>drifted</td><td><code>source_status_split</code></td></tr>
      <tr><td>5</td><td><code>attached</code> bound</td><td>2,249</td><td><strong>2,197</strong></td><td>drifted</td><td><code>source_status_split</code></td></tr>
      <tr><td>6</td><td><code>stale</code> bound</td><td>93</td><td><strong>93</strong></td><td>current</td><td><code>source_status_split</code></td></tr>
      <tr><td>7</td><td><code>stale</code> unbound &mdash; &ldquo;expected residue&rdquo;</td><td>1,652</td><td><strong>1,737</strong></td><td>drifted (+85)</td><td><code>source_status_split</code></td></tr>
      <tr><td>8</td><td><code>extracted</code> unbound &mdash; &ldquo;real work not done&rdquo;</td><td>1,283</td><td><strong>1,359</strong></td><td>drifted (+76)</td><td><code>source_status_split</code></td></tr>
      <tr><td>9</td><td><code>extracted</code> bound</td><td>44</td><td><strong>37</strong></td><td>drifted</td><td><code>source_status_split</code></td></tr>
      <tr><td>10</td><td><code>skipped</code> unbound</td><td>1,413</td><td><strong>1,411</strong></td><td>drifted (&minus;2)</td><td><code>source_status_split</code></td></tr>
      <tr><td>11</td><td><code>failed</code> unbound</td><td>101</td><td><strong>109</strong></td><td>drifted (+8)</td><td><code>source_status_split</code></td></tr>
      <tr><td>12</td><td><code>failed</code> bound</td><td>2</td><td><strong>2</strong></td><td>current</td><td><code>source_status_split</code></td></tr>
      <tr><td>13</td><td><code>not_physical_quantity</code> unbound</td><td>10</td><td><strong>10</strong></td><td>current</td><td><code>source_status_split</code></td></tr>
      <tr><td>14</td><td>&ldquo;<strong>398 rows carry no recorded reason at all</strong>&rdquo;</td><td>398</td><td><strong>398</strong></td><td>current (exact)</td><td><code>MATCH (s {status:'skipped'}) WHERE coalesce(s.skip_reason,'')='' RETURN count</code></td></tr>
      <tr><td>15</td><td>&ldquo;<strong>5,072 of 61,366 <code>IMASNode</code> paths</strong> currently carry a standard name&rdquo;</td><td>5,072 / 61,366</td><td><strong>4,992 distinct nodes</strong> (5,092 edges) / <strong>61,366</strong></td><td>drifted (&minus;80 or &minus;20)</td><td><code>MATCH (n:IMASNode)-[:HAS_STANDARD_NAME]-&gt;() RETURN count(DISTINCT n), count(*)</code></td></tr>
      <tr><td>16</td><td>&ldquo;<code>non_nameable_reason</code> is written only when <code>source_status=='skipped'</code> (<code>graph_ops.py:12324</code>), and it never reads <code>last_error</code>&rdquo;</td><td>mechanism</td><td>line 12324 has moved; the roster at <code>graph_ops.py:12441&ndash;12462</code> now reads <code>last_error</code> and emits the sentinel <code>"cause not recorded"</code></td><td>stale</td><td>code read + <code>git log -S 'cause not recorded'</code> &rarr; e20817e41</td></tr>
    </tbody>
  </table>

  <p><strong>Verdict counts: current 4, drifted 11, stale 1 &mdash; sum 16 of 16 rows.</strong></p>

  <p>The verdict is read against the plan text, not against a tolerance a reader
  would have to guess. A row is <em>current</em> when the re-measured value equals the
  asserted one exactly, <em>drifted</em> when it differs but the quantity is still the
  quantity named, and <em>stale</em> when the asserted mechanism no longer holds at
  all.</p>

  <h2>The first unstarted beat: already-complete</h2>

  <p>The plan's first beat is <code>f-usb-001</code> &mdash; ship &sect;3 before &sect;2:
  make the write path unable to record a skip without a reason, then reconstruct the
  missing 398 or mark them as an honest unknown. The deciding measurement is that the
  reason-less cohort is <strong>frozen</strong>, and the roster mechanism the plan names
  has been replaced:</p>

  <ul>
    <li><strong>No growth.</strong> The 398 reason-less skipped sources carry a
    <code>skipped_at</code> between <code>2026-06-17T17:21:35Z</code> and
    <code>2026-07-28T07:43:24Z</code>. <strong>Zero</strong> were skipped after the plan's
    2026-09-08 write date, out of 1,411 skipped sources in total. Nothing the current
    writers produce lands in this cohort, so the write path is not the source of the 398.</li>
    <li><strong>The mechanism the plan cites is fixed.</strong> Commit
    <code>e20817e41</code> (2026-09-08) changed the roster to
    <code>non_nameable_reason = last_error or skip_cause or "cause not recorded"</code>, so a
    skip that lost its diagnostic now reports an explicit unknown instead of a blank
    field &mdash; the honest-unknown disposition the plan asks for, delivered at read time.
    The plan's line reference <code>graph_ops.py:12324</code> is behind the current
    <code>graph_ops.py</code> layout; the block now sits at lines 12441&ndash;12462.</li>
    <li><strong>Both skip writers require a reason by signature.</strong>
    <code>mark_source_skipped</code> takes <code>reason</code> as a required
    <code>str</code> keyword; <code>write_skipped_sources</code> reads
    <code>r["skip_reason"]</code> as a required record key. Neither has a blank-reason
    route.</li>
  </ul>

  <p>Residual, recorded as a follow-on rather than repaired here: the 398 graph rows
  still carry an empty <code>skip_reason</code> &mdash; only the roster supplies the
  sentinel. Since 397 of the 398 have no surviving evidence on any of
  <code>skip_reason</code>, <code>skip_reason_detail</code> or <code>last_error</code>,
  a data write would add nothing a reader does not already get from the roster.</p>

  <h2>What the drift means for rescoping</h2>

  <p>The unbound population, corrected for the classes the plan itself excludes, is
  no longer 2,797 rows but 2,879: the backlog proper is 1,359 <code>extracted</code>,
  1,411 <code>skipped</code> and 109 <code>failed</code>, against 1,737
  <code>stale</code> and 10 <code>not_physical_quantity</code> that are mechanism, not
  backlog. The plan's central framing survives unchanged: counting all unbound rows as
  one backlog still overstates the real work by roughly a third, and it now overstates
  4,626 as against 2,879.</p>

  <p>The item that moved most is not in the original table: the plan's §3 mechanism
  claim of 2026-09-08 is superseded, and the sprint's live risk has migrated to the
  later followups &mdash; the manifest-generability invariant
  (<code>f-usb-a-manifest-must-be-fully-generable</code>) and the export-side
  producer predicate (<code>f-usb-export-refuses-a-name-with-no-producer</code>).</p>

  <h2>Remaining effort</h2>

  <p>Restated in worker-hours against the plan's <code>plan-effort-hours</code> of 10.0:</p>

  <p>engineering: §3 beat <strong>0 h</strong> (already-complete, above); §2 eligibility
  census on the 1,359 extracted-unbound rows <strong>1 h</strong>; the three later
  followups &mdash; generability instrument, export producer predicate, export-exposure
  audit &mdash; <strong>6 h</strong>. The §2 <code>sn run</code> composition pass is
  pipeline spend sized to the eligibility count (a paid review cohort), not
  worker-hours. <strong>Revised engineering total 7 h</strong>, down from 10.0 h
  because the §3 guard and reconstruction halves are now accounted for.</p>

  <h2>Artifacts and logs</h2>

  <ul>
    <li><code>census.log</code> &mdash; discovery, totals, status split, skip-reason coverage (exit 0).</li>
    <li><code>census3.log</code> &mdash; skipped-node keys, reason-less cohort timing, source types (exit 0).</li>
    <li><code>census4.log</code> &mdash; reason-less <code>skipped_at</code> range, distinct name coverage (exit 0).</li>
  </ul>

  <p>No graph mutation was performed: every statement was a read, and
  <code>census.json</code> / <code>census3.json</code> / <code>census4.json</code> hold the
  raw rows.</p>
</main>
</body>
</html>