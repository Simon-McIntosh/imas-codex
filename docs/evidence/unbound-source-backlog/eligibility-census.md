<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="docs-project" content="imas-codex">
  <meta name="reckon-type" content="evidence">
  <meta name="plan-slug" content="unbound-source-backlog-eligibility-census">
  <meta name="plan-title" content="Unbound source backlog &mdash; eligibility census of the extracted-unbound cohort">
  <meta name="plan-evidence-for" content="unbound-source-backlog">
  <title>Extracted-unbound eligibility census | imas-codex</title>
  <link rel="stylesheet" href="/_shared/foundation.css">
  <link rel="stylesheet" href="/_shared/dashboard.css">
</head>
<body>
<main class="plan-doc">
  <h1>Extracted-unbound eligibility census</h1>

  <p><strong>1,268 of the 1,359 extracted-unbound sources are still admissible under the
  current admission rules; 91 are not.</strong> The two counts come from one pass over one
  cohort and sum to the cohort size measured in that pass
  (<code>1268 + 91 = 1359</code>). Of the 91 inadmissible rows, <strong>65 are
  test/placeholder identities</strong> &mdash; the same string <code>test/path</code> or
  <code>__revreltest__/leaf1</code> &mdash; so the residue attributable to a rule actually
  doing work is <strong>26 rows</strong>. No row in the cohort has reached its attempt cap:
  the highest <code>attempt_count</code> present is 4.</p>

  <p>Measured read-only on <strong>2026-09-17</strong> at worktree revision
  <code>3d80e116f4c219519ddf5ece66c1bed8c8b23e8e</code>, by a read-only census enumerator that
  applies the composition path's admission gates in order and records the first failure per row.
  No graph mutation was performed.</p>

  <h2>What was measured</h2>

  <p>The cohort is every <code>StandardNameSource</code> with <code>status = 'extracted'</code>
  and no outgoing <code>PRODUCED_NAME</code> edge: the sources the Data Dictionary extraction
  produced that no standard name was ever composed for. Each row was classified against the
  gates the composition path applies, in pipeline order, and the <em>first</em> gate it fails
  is the attribution. A row that fails no gate is counted admissible.</p>

  <p>The census enumerator is not the pipeline: it re-implements the gates and reads the graph,
  so its rule attribution is a reconstruction. The positive controls in the last section measure
  how good that reconstruction is.</p>

  <h2>Named queries and instruments</h2>

  <p>Every count below carries the query or callable that produced it. Four instruments were
  used, and no count in this page comes from an unnamed read.</p>

  <h3><code>q-cohort-extracted-unbound</code> &mdash; the cohort and every row attribute</h3>

  <p>Run once, returning 1,359 rows. It resolves each source to its Data Dictionary node by
  <code>FROM_DD_PATH</code> edge first and by <code>coalesce(sns.dd_path, sns.source_id)</code>
  only when no edge exists, reports which route resolved it, and carries the node's
  <code>units</code>, <code>data/</code> child count and <code>quantity</code> child count so the
  container gate can be evaluated in Python rather than guessed in Cypher.</p>

  <pre><code>MATCH (sns:StandardNameSource)
WHERE sns.status = 'extracted'
  AND NOT (sns)-[:PRODUCED_NAME]-&gt;(:StandardName)
CALL (sns) {
  OPTIONAL MATCH (sns)-[:FROM_DD_PATH]-&gt;(edge_node:IMASNode)
  WITH sns, collect(edge_node) AS edge_nodes
  RETURN CASE WHEN size(edge_nodes) = 1 THEN edge_nodes[0] ELSE null END AS edge_node,
         size(edge_nodes) AS edge_node_count }
CALL (sns, edge_node) {
  OPTIONAL MATCH (by_id:IMASNode) WHERE by_id.id = coalesce(sns.dd_path, sns.source_id)
  WITH by_id, edge_node
  RETURN CASE WHEN edge_node IS NOT NULL THEN edge_node ELSE by_id END AS n,
         CASE WHEN edge_node IS NOT NULL THEN 'edge'
              WHEN by_id IS NOT NULL THEN 'path_scalar'
              ELSE 'unresolved' END AS resolved_by }
OPTIONAL MATCH (n)-[:HAS_UNIT]-&gt;(u:Unit)
WITH sns, n, resolved_by, edge_node_count,
     [x IN collect(DISTINCT u.id) WHERE x IS NOT NULL] AS units
OPTIONAL MATCH (n)-[:IN_IDS]-&gt;(ids:IDS)
CALL (n) {
  OPTIONAL MATCH (data_child:IMASNode)-[:HAS_PARENT]-&gt;(n)
  WHERE n IS NOT NULL AND data_child.id = n.id + '/data'
  WITH n, collect(data_child) AS data_children
  OPTIONAL MATCH (q:IMASNode)-[:HAS_PARENT]-&gt;(n)
  WHERE n IS NOT NULL AND q.node_category = 'quantity'
  WITH n, data_children, collect(q) AS quantity_children
  RETURN size(data_children) AS data_child_count,
         size(quantity_children) AS quantity_child_count }
RETURN sns.id, sns.source_type, sns.source_id, sns.skip_reason, sns.dd_path,
       sns.dd_data_type, sns.dd_unit, coalesce(sns.attempt_count,0) AS attempt_count,
       resolved_by, n.id AS node_id, n.node_category, n.data_type, n.description,
       n.documentation, n.unit, n.lifecycle_status, units,
       data_child_count, quantity_child_count, ids.id AS ids_name</code></pre>

  <p>An identical statement with the <code>status</code> predicate and the
  <code>PRODUCED_NAME</code> exclusion replaced by a single <code>$status</code> parameter is
  <code>q-status-population</code>, used only for the positive controls below so that every gate
  is held byte-identical between the census and the control.</p>

  <h3><code>i-dd-admission</code> &mdash; the ordered gate list</h3>

  <p>Applied to each <code>source_type = 'dd'</code> row, first failure wins:
  <code>dd_lifecycle_removed</code> (node <code>lifecycle_status = 'removed'</code>) &rarr;
  <code>node_category_ineligible</code> (<code>n.node_category</code> not in
  <code>imas_codex.core.node_categories.SN_SOURCE_CATEGORIES</code>) &rarr;
  <code>empty_description</code> &rarr; <code>container_node_no_signal_signature</code>
  (<code>data_type</code> is <code>STRUCTURE</code>/<code>STRUCT_ARRAY</code> and the node is not
  a <code>quantity</code> with at least one <code>data/</code> child and no <code>quantity</code>
  child) &rarr; <code>imas_codex.standard_names.sources.dd_qualifier.qualify_dd</code>, whose own
  <code>reason_code</code> is the attribution when it refuses.
  <code>missing_dd_authority</code> is checked before all of them and fires when the row
  resolves to no <code>IMASNode</code> at all.</p>

  <h3><code>i-derived-parent-admission</code></h3>

  <p>Applied to every non-<code>dd</code> row:
  <code>imas_codex.standard_names.parents.is_admissible_parent_name(source_id, graph)</code>.
  A refusal is attributed to <code>derived_parent_inadmissible</code> with the callable's own
  <code>reason</code> as detail.</p>

  <h2>Result &mdash; the cohort and its two halves</h2>

  <table>
    <caption>One pass, <code>q-cohort-extracted-unbound</code>, 2026-09-17. The two rows sum to
    the cohort size.</caption>
    <thead><tr><th>Outcome</th><th>Count</th><th>Share</th><th>Produced by</th></tr></thead>
    <tbody>
      <tr><td>Admissible under the current rules</td><td><strong>1,268</strong></td><td>93.3%</td><td><code>q-cohort-extracted-unbound</code> + <code>i-dd-admission</code> / <code>i-derived-parent-admission</code></td></tr>
      <tr><td>Inadmissible under the current rules</td><td><strong>91</strong></td><td>6.7%</td><td>same pass, first failing gate</td></tr>
      <tr><td><strong>Cohort measured in the same pass</strong></td><td><strong>1,359</strong></td><td>100%</td><td><code>len(rows)</code> from the one query above</td></tr>
    </tbody>
  </table>

  <p>The cohort size is an independent agreement as well as an arithmetic one: earlier the same
  day, a separate instrument covering all 10,019 sources (<code>source_status_split</code>, the
  headline re-measurement) reported the same <strong>1,359</strong> <code>extracted</code>-unbound
  rows. Two instruments, two queries, one figure.</p>

  <table>
    <caption>The same two halves split by source family, with the resolution route each row took.</caption>
    <thead><tr><th>Source family</th><th>Admissible</th><th>Inadmissible</th><th>Total</th><th>Gate applied</th></tr></thead>
    <tbody>
      <tr><td><code>dd</code> (Data Dictionary extraction)</td><td>1,213</td><td><strong>29</strong></td><td>1,242</td><td><code>i-dd-admission</code></td></tr>
      <tr><td><code>derived</code> (parent identity)</td><td>52</td><td><strong>62</strong></td><td>114</td><td><code>i-derived-parent-admission</code></td></tr>
      <tr><td><code>manual</code></td><td>3</td><td>&mdash;</td><td>3</td><td><code>i-derived-parent-admission</code></td></tr>
      <tr><td><strong>Total</strong></td><td><strong>1,268</strong></td><td><strong>91</strong></td><td><strong>1,359</strong></td><td></td></tr>
    </tbody>
  </table>

  <p>Of the 1,242 <code>dd</code> rows, <strong>1,235 resolved through a
  <code>FROM_DD_PATH</code> edge</strong>, <strong>1 through the
  <code>dd_path</code> scalar</strong> and <strong>6 through neither</strong>. That
  <code>resolved_by</code> split is reported by the query itself, so the 6
  <code>missing_dd_authority</code> rows are visible as a resolution failure rather than an
  empty read.</p>

  <h2>Result &mdash; the 91 inadmissible rows by the rule that excludes them</h2>

  <table>
    <caption>First failing gate, in pipeline order. Sums to 91.</caption>
    <thead><tr><th>Rule</th><th>Count</th><th>Source family</th><th>Instrument</th><th>Detail carried by the row</th></tr></thead>
    <tbody>
      <tr><td><code>derived_parent_inadmissible</code></td><td><strong>62</strong></td><td><code>derived</code></td><td><code>i-derived-parent-admission</code></td><td>59 of the 62 are the placeholder identity below; 2 are bare bases; 1 lost a projection token</td></tr>
      <tr><td><code>node_category_ineligible</code></td><td><strong>18</strong></td><td><code>dd</code></td><td rowspan="5"><code>i-dd-admission</code></td><td><code>structural</code> 9, <code>fit_artifact</code> 8, <code>representation</code> 1</td></tr>
      <tr><td><code>missing_dd_authority</code></td><td><strong>6</strong></td><td><code>dd</code></td><td>resolves to no <code>IMASNode</code>: 5 are <code>test/path</code>, 1 is <code>__revreltest__/leaf1</code></td></tr>
      <tr><td><code>container_node_no_signal_signature</code></td><td><strong>3</strong></td><td><code>dd</code></td><td><code>STRUCT_ARRAY</code> 2, <code>STRUCTURE</code> 1</td></tr>
      <tr><td><code>dd_lifecycle_removed</code></td><td><strong>1</strong></td><td><code>dd</code></td><td><code>magnetics/bpol_probe/non_linear_response/b_field_non_linear</code></td></tr>
      <tr><td><code>duplicate_ids</code></td><td><strong>1</strong></td><td><code>dd</code></td><td><code>core_instant_changes/change/profiles_1d/rotation_frequency_tor_sonic</code> &mdash; <code>core_instant_changes</code> duplicates <code>core_profiles</code> quantities with <code>change_in_*</code> prefixes (<code>qualify_dd</code> reason code)</td></tr>
      <tr><td colspan="2"><strong>Total</strong></td><td colspan="3"><strong>91</strong></td></tr>
    </tbody>
  </table>

  <figure>
    <svg viewBox="0 0 700 330" width="700" height="330" role="img"
         aria-label="Stacked bar decomposing the 91 inadmissible rows into 65 placeholder identities and 26 rule-driven rows, and a bar per excluding rule"
         style="max-width:100%;font:12px/1.4 system-ui,sans-serif" fill="currentColor">
      <text x="8" y="20" font-weight="700">91 inadmissible, decomposed</text>
      <text x="8" y="52" text-anchor="start">placeholder</text>
      <text x="8" y="68" text-anchor="start">identity</text>
      <rect x="90" y="40" width="429" height="30" fill="currentColor" fill-opacity="0.25"/>
      <rect x="519" y="40" width="171" height="30" fill="currentColor" fill-opacity="0.7"/>
      <text x="110" y="60" font-weight="700">65 placeholder identities</text>
      <text x="535" y="60" font-weight="700">26 rule-driven</text>
      <text x="690" y="88" text-anchor="end" font-style="italic">test/path &times;64, __revreltest__/leaf1 &times;1</text>

      <text x="8" y="120" font-weight="700">by excluding rule</text>

      <text x="84" y="140" text-anchor="end">derived_parent_inadmissible</text>
      <rect x="90" y="128" width="600" height="16" fill="currentColor" fill-opacity="0.7"/>
      <text x="682" y="140" text-anchor="end" font-weight="700">62</text>

      <text x="84" y="168" text-anchor="end">node_category_ineligible</text>
      <rect x="90" y="156" width="174" height="16" fill="currentColor" fill-opacity="0.7"/>
      <text x="272" y="168" font-weight="700">18</text>

      <text x="84" y="196" text-anchor="end">missing_dd_authority</text>
      <rect x="90" y="184" width="58" height="16" fill="currentColor" fill-opacity="0.25"/>
      <text x="156" y="196" font-weight="700">6</text>

      <text x="84" y="224" text-anchor="end">container_no_signal_signature</text>
      <rect x="90" y="212" width="29" height="16" fill="currentColor" fill-opacity="0.7"/>
      <text x="127" y="224" font-weight="700">3</text>

      <text x="84" y="252" text-anchor="end">dd_lifecycle_removed</text>
      <rect x="90" y="240" width="10" height="16" fill="currentColor" fill-opacity="0.7"/>
      <text x="108" y="252" font-weight="700">1</text>

      <text x="84" y="280" text-anchor="end">duplicate_ids</text>
      <rect x="90" y="268" width="10" height="16" fill="currentColor" fill-opacity="0.7"/>
      <text x="108" y="280" font-weight="700">1</text>

      <rect x="90" y="300" width="14" height="14" fill="currentColor" fill-opacity="0.25"/>
      <text x="112" y="312">placeholder identity (every one sits under <tspan font-style="italic">derived_parent_inadmissible</tspan> or <tspan font-style="italic">missing_dd_authority</tspan>)</text>
    </svg>
    <figcaption>Bars are counts of rows, not shares. The paler segment is the part of the
    91 that no admission rule decided: <code>test/path</code> and <code>__revreltest__/leaf1</code>
    are test scaffolding that reached the graph as sources, and they account for 65 of the 91
    refusals. Only the remaining 26 are a statement about the rules.</figcaption>
  </figure>

  <h3>The 65 rows no rule decided</h3>

  <table>
    <caption>Placeholder identities inside the inadmissible set, by the gate they fail.</caption>
    <thead><tr><th>Identity</th><th>Rows</th><th>Source family</th><th>Fails</th></tr></thead>
    <tbody>
      <tr><td><code>test/path</code></td><td><strong>59</strong></td><td><code>derived</code></td><td><code>derived_parent_inadmissible</code> &mdash; not a valid grammar token</td></tr>
      <tr><td><code>test/path</code></td><td><strong>5</strong></td><td><code>dd</code></td><td><code>missing_dd_authority</code> &mdash; resolves to no node</td></tr>
      <tr><td><code>__revreltest__/leaf1</code></td><td><strong>1</strong></td><td><code>dd</code></td><td><code>missing_dd_authority</code> &mdash; resolves to no node</td></tr>
    </tbody>
  </table>

  <p>These 65 rows are a source-hygiene finding in their own right and they are not what this
  census was asked for: they are test fixtures that were written into the graph as
  <code>StandardNameSource</code> rows and then left there. They inflate the refusal count more than
  threefold &mdash; 65 of the 91 &mdash; and they will inflate any future count of the same kind
  whenever a test run writes to the graph the census reads.</p>

  <h3>The 26 rows a rule decided</h3>

  <table>
    <caption>Rule-driven refusals, named. These are the rows the §2 composition pass cannot
    compose and must not be sized to.</caption>
    <thead><tr><th>Rule</th><th>Rows</th><th>Identities</th></tr></thead>
    <tbody>
      <tr><td><code>node_category_ineligible</code> (<code>structural</code>)</td><td>9</td><td><code>ntms/time_slice/mode</code>; <code>spectrometer_visible/channel/isotope_ratios/isotope</code>; <code>plasma_transport/model</code>; <code>edge_transport/model/ggd/neutral/state/momentum</code>; <code>summary/pedestal_fits</code>; <code>ic_antennas/antenna/module</code>; <code>ic_antennas/antenna/module/strap/geometry/oblique</code>; <code>waves/coherent_wave</code>; <code>ferritic/object/axisymmetric/annulus</code></td></tr>
      <tr><td><code>node_category_ineligible</code> (<code>fit_artifact</code>)</td><td>8</td><td>the eight <code>equilibrium/time_slice/constraints/&hellip;/reconstructed</code> positions: <code>pressure</code>, <code>pressure_rotational</code>, <code>pf_current</code>, <code>b_field_tor_vacuum_r</code>, and <code>strike_point</code>/<code>x_point</code> <code>position_reconstructed</code> in both <code>r</code> and <code>z</code></td></tr>
      <tr><td><code>node_category_ineligible</code> (<code>representation</code>)</td><td>1</td><td><code>plasma_transport/model/ggd</code></td></tr>
      <tr><td><code>derived_parent_inadmissible</code> (bare base, clause A)</td><td>2</td><td><code>power_density</code>; <code>wavelength</code></td></tr>
      <tr><td><code>derived_parent_inadmissible</code> (projection lost token)</td><td>1</td><td><code>root_mean_square_of_spectral_width_of_spectrometer_channel</code> &mdash; token <code>of</code> dropped, so the name has no canonical form</td></tr>
      <tr><td><code>container_node_no_signal_signature</code></td><td>3</td><td><code>edge_sources/source/ggd/electrons/energy</code> and <code>plasma_sources/source/ggd/electrons/energy</code> (<code>STRUCT_ARRAY</code>); <code>mhd_linear/time_slice/toroidal_mode/plasma/psi_potential_perturbed</code> (<code>STRUCTURE</code>)</td></tr>
      <tr><td><code>dd_lifecycle_removed</code></td><td>1</td><td><code>magnetics/bpol_probe/non_linear_response/b_field_non_linear</code></td></tr>
      <tr><td><code>duplicate_ids</code></td><td>1</td><td><code>core_instant_changes/change/profiles_1d/rotation_frequency_tor_sonic</code></td></tr>
      <tr><td colspan="2"><strong>Total</strong></td><td><strong>26</strong> &mdash; 23 <code>dd</code> and 3 <code>derived</code></td></tr>
    </tbody>
  </table>

  <h2>Two side findings that bear on how §2 is sized</h2>

  <p><strong>No row in this cohort has exhausted its attempt budget.</strong> The maximum
  <code>attempt_count</code> across all 1,359 rows is <strong>4</strong>, against a cap of five;
  353 of the 1,242 <code>dd</code> rows and 10 of the 114 <code>derived</code>
  rows have been attempted at least once, so <strong>355 of the 1,268 admissible rows</strong>
  have spent an attempt and <strong>913 have never been attempted at all</strong>; only 8 of the
  91 inadmissible rows were ever attempted. So the 1,268 is a ceiling on work not yet started,
  not a rediscovery of searches that already failed.</p>

  <p><strong>The rules have tightened since these rows were processed.</strong> Applying the
  current gates to populations that were <em>already</em> handled finds
  <strong>18 inadmissible rows inside <code>composed</code></strong> (11
  <code>container_node_no_signal_signature</code>, 6 <code>node_category_ineligible</code>,
  1 <code>duplicate_ids</code>) and <strong>33 inside <code>attached</code></strong> (20
  container, 8 <code>node_category_ineligible</code>, 4 <code>configurable_meaning</code>, 1
  <code>dd_lifecycle_removed</code>). Those rows were admitted when they ran and would be
  refused now. This is the effect the plan anticipated, and it is small: 51 rows across 4,767
  already-processed <code>dd</code> rows.</p>

  <h2>Positive controls &mdash; what the reconstruction is resting on</h2>

  <p>The census enumerator re-implements the gates, so the controls that matter are the ones
  where the graph already records an answer. <code>q-status-population</code> was run over five
  labelled <code>dd</code> populations with every gate held identical.</p>

  <table>
    <caption>Labelled populations classified by the census rules, 2026-09-17. "Labelled" counts
    rows carrying a non-empty <code>skip_reason</code>; "agree" counts those whose recorded
    reason equals the rule this census attributes.</caption>
    <thead><tr><th><code>status</code></th><th>Rows</th><th>Classified inadmissible</th><th>Labelled</th><th>Rule agrees with label</th><th>Rules found</th></tr></thead>
    <tbody>
      <tr><td><code>stale</code></td><td>1,820</td><td><strong>1,820 (100%)</strong></td><td>251</td><td>3</td><td><code>dd_lifecycle_removed</code> 1,530; <code>missing_dd_authority</code> 290</td></tr>
      <tr><td><code>composed</code></td><td>2,589</td><td>18</td><td>18</td><td>0</td><td><code>container_node_no_signal_signature</code> 11; <code>node_category_ineligible</code> 6; <code>duplicate_ids</code> 1</td></tr>
      <tr><td><code>attached</code></td><td>2,178</td><td>33</td><td>29</td><td>4</td><td><code>container_node_no_signal_signature</code> 20; <code>node_category_ineligible</code> 8; <code>configurable_meaning</code> 4; <code>dd_lifecycle_removed</code> 1</td></tr>
      <tr><td><code>skipped</code></td><td>1,411</td><td>1,016</td><td>1,013</td><td>873</td><td><code>temporal_coordinate</code> 398; <code>local_coordinate_frame</code> 390; <code>dd_unit_mixed_non_standard</code> 107; <code>configurable_meaning</code> 80; <code>node_category_ineligible</code> 23; <code>dd_unit_unresolvable</code> 17; <code>container_node_no_signal_signature</code> 1</td></tr>
      <tr><td><code>failed</code></td><td>111</td><td><strong>0</strong></td><td>2</td><td>0</td><td>&mdash;</td></tr>
    </tbody>
  </table>

  <ul>
    <li><strong>The instrument flags the population that is known to be invalid.</strong>
    Every one of the 1,820 <code>stale</code> rows is refused &mdash; 1,530 on
    <code>dd_lifecycle_removed</code> and 290 because the node no longer resolves. A rule set
    that returned a clean bill of health for <code>stale</code> would be measuring nothing.</li>
    <li><strong><code>failed</code> rows are not admission failures.</strong> Zero of the 111
    are inadmissible, which is the expected shape: a source reaches <code>failed</code> after
    being admitted, so the refusal has to come from somewhere else.</li>
    <li><strong>Where the graph records a reason, the reconstruction agrees on 873 of 1,013
    (86%).</strong> The 140 disagreements are the honest limit of this instrument and are
    <em>not</em> resolved here: <code>skip_reason</code> is written by a different point in the
    pipeline and the two vocabularies are not guaranteed to be the same rule set, so a
    disagreement does not establish that either is wrong. It does establish that the attribution
    in the rule table above is a reconstruction, which is why each row names its instrument.</li>
    <li><strong>1,411 <code>skipped</code> rows in the control matches the 1,411 the headline
    re-measurement reported earlier the same day</strong>, from a different query &mdash; a third
    independent agreement on the population this plan's two halves are drawn from.</li>
  </ul>

  <h2>What this does not establish</h2>

  <ul>
    <li><strong>Admissible is not composable.</strong> 1,268 rows pass the admission gates;
    whether each one produces a name the review pool accepts is what the composition pass itself
    will show. This census sizes that pass, it does not pre-empt its result.</li>
    <li><strong>The gate list is read from code, not from a specification.</strong>
    <code>i-dd-admission</code> mirrors the order the extraction path applies its gates; if that
    path changes, this page's rule ordering is stale. The 86% label agreement is the only
    external check on it.</li>
    <li><strong>Row attributes are as at the measurement, not as at extraction time.</strong>
    A row classified here as <code>node_category_ineligible</code> may have been admissible when
    it was extracted.</li>
  </ul>

  <h2>Artifacts</h2>

  <ul>
    <li><code>census.py</code> &mdash; cohort query, gate list, classifier. Run
    <code>census.py</code> for the census and <code>census.py validate</code> for the controls.</li>
    <li><code>census.log</code> &mdash; cohort, totals, by-family split, rule counts (exit 0).</li>
    <li><code>census-rows.json</code> &mdash; all 1,359 classified rows, every attribute the
    query returned plus the verdict and attribution.</li>
    <li><code>detail.py</code> / <code>detail.log</code> &mdash; rule &times; attempted,
    rule &times; family, named identities.</li>
    <li><code>validate.log</code> &mdash; the five labelled populations (exit 0).</li>
  </ul>

  <p>All paths relative to the run directory named in the manifest. No graph mutation was
  performed: every statement was a read.</p>
</main>
</body>
</html>