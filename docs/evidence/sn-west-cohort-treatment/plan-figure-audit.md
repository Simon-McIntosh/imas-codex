<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="docs-project" content="imas-codex">
  <meta name="reckon-type" content="evidence">
  <meta name="plan-slug" content="sn-west-cohort-treatment-plan-figure-audit">
  <meta name="plan-title" content="WEST Cohort Treatment: Headline Figures Re-measured">
  <meta name="plan-summary" content="The WEST plan's asserted cohort quantities re-taken against the committed manifest and live graph on 2026-09-17.">
  <meta name="plan-status" content="active">
  <meta name="plan-owner" content="Simon McIntosh">
  <meta name="plan-evidence-for" content="sn-west-cohort-treatment">
  <meta name="plan-tags" content="standard-names,west,census,release-readiness">
  <title>WEST Cohort Treatment: Headline Figures Re-measured | imas-codex</title>
  <link rel="stylesheet" href="/_shared/foundation.css">
  <link rel="stylesheet" href="/_shared/dashboard.css">
</head>
<body>
  <main class="plan-doc">
    <h2 id="result">Result</h2>
    <p>
      Five quantitative claims from the live plan were re-measured on
      <strong>2026-09-17</strong> against the committed
      <code>west_production_dd_paths.yaml</code> manifest and the live graph. One is
      <strong>current</strong>, one is <strong>drifted</strong>, and three are
      <strong>stale</strong>. The current manifest contains <strong>342 unique source
      paths</strong>; <strong>340</strong> are carried by a name accepted on both the name and
      documentation axes, and the two residual paths are explicit rather than hidden.
    </p>
    <p>
      The release projection resolves those 340 carried source rows to
      <strong>230 prospective candidate identities</strong>. Every one has a direct binding
      from the committed manifest, and <strong>zero</strong> are bound only to sources outside
      it. The original 150-name scoping defect is therefore <strong>closed</strong>. The
      original 37-path untreated tail is not closed completely, but its recorded size is stale:
      the current residual is two paths, not 37.
    </p>

    <h2 id="quantities">Asserted quantities and current measurements</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Asserted quantity, verbatim</th>
          <th>Re-measured value</th>
          <th>Instrument</th>
          <th>Verdict</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>Hero: “leaves <strong>37 manifest paths untreated</strong>”</td>
          <td>
            <strong>2 of 342</strong> are not carried:
            <code>calorimetry/group/component/energy_total/data</code> is
            <code>extracted</code>; <code>camera_x_rays/camera/camera_dimensions</code>
            is <code>failed</code> at the attempt cap.
          </td>
          <td><a href="#carried-query">Per-source carried census</a></td>
          <td>stale</td>
        </tr>
        <tr>
          <td>2</td>
          <td>Hero: “publishes <strong>150 names</strong> that are not in its own batch manifest”</td>
          <td>
            <strong>0 of 230</strong> prospective candidates are bound only outside the
            committed manifest; <strong>0</strong> lack a direct manifest binding. The defect
            is closed.
          </td>
          <td>
            <a href="#scope-query">Prospective scope census</a> plus the
            <a href="#scope-control">outside-scope positive control</a>
          </td>
          <td>stale</td>
        </tr>
        <tr>
          <td>3</td>
          <td>§9a: “Of <strong>355</strong> <code>manifest_sources</code> rows, <strong>324</strong> carry a name and <strong>31</strong> do not”</td>
          <td>
            The committed manifest now has <strong>342</strong> rows; <strong>340</strong>
            carry an accepted-on-both-axes name and <strong>2</strong> do not.
          </td>
          <td><a href="#carried-query">Per-source carried census</a></td>
          <td>drifted</td>
        </tr>
        <tr>
          <td>4</td>
          <td>2026-09-15 landing record: “a per-source census of the committed manifest reads <strong>340 of 342 carried</strong>”</td>
          <td><strong>340 of 342 carried</strong>, exactly reproducing the recorded bound</td>
          <td><a href="#carried-query">Per-source carried census</a></td>
          <td>current</td>
        </tr>
        <tr>
          <td>5</td>
          <td>First ordered beat: “<strong>Five withheld names</strong> are accepted, valid and scored 0.844 to 0.906, held back only by a stored verdict”</td>
          <td>
            <strong>0 of 230</strong> prospective candidates are quarantined. The same
            predicate finds <strong>660</strong> quarantined names graph-wide, so the zero is
            not a blind instrument.
          </td>
          <td>
            <a href="#quarantine-query">Candidate quarantine census</a> plus the
            <a href="#quarantine-control">global positive control</a>
          </td>
          <td>stale</td>
        </tr>
      </tbody>
    </table>
    <p>
      <strong>Verdict counts:</strong> current <strong>1</strong>, drifted
      <strong>1</strong>, stale <strong>3</strong>. <strong>1 + 1 + 3 = 5</strong>
      rows.
    </p>

    <h2 id="residual">The two residual sources</h2>
    <table>
      <thead>
        <tr><th>Committed source path</th><th>Current state</th><th>Measured reading</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><code>calorimetry/group/component/energy_total/data</code></td>
          <td><code>extracted</code></td>
          <td>
            No <code>PRODUCED_NAME</code> edge and no <code>produced_sn_id</code>. The
            release projection reports it unmatched. It needs fresh composition; it is not
            safely recoverable by attaching it to a terminal identity.
          </td>
        </tr>
        <tr>
          <td><code>camera_x_rays/camera/camera_dimensions</code></td>
          <td><code>failed</code></td>
          <td>
            No <code>PRODUCED_NAME</code> edge and no <code>produced_sn_id</code>;
            <code>last_error</code> is “compose claim-attempt cap reached”. The release
            projection reports it unmatched.
          </td>
        </tr>
      </tbody>
    </table>
    <p>
      These two rows are the complete current release-critical source tail. The census found all
      <strong>342 of 342</strong> source nodes, so neither absence is caused by a missing manifest
      node. A known-present control also resolves
      <code>spectrometer_visible/channel/isotope_ratios/signal_to_noise</code> to the accepted,
      documentation-accepted, validation-valid
      <code>spectral_signal_to_noise_ratio_of_spectrometer_channel</code>, proving the relationship
      traversal sees a carried row that is known to exist.
    </p>

    <h2 id="first-beat">The plan's first unstarted beat</h2>
    <p>
      <strong>Already complete for the current release cohort.</strong> The driving followup's
      first ordered beat is “reconcile the quarantine instruments”. The current manifest-derived
      projection contains 230 identities and <strong>zero quarantined identities</strong>, so no
      WEST candidate remains held on the disagreement that motivated that beat. This is not an
      untested zero: the graph has 5,130 <code>StandardName</code> nodes, 5,128 with a
      <code>validation_status</code>, and the identical predicate sees 660 quarantined names
      globally. The release cohort is therefore empty on the condition the beat existed to
      resolve, while the instrument demonstrably fires elsewhere.
    </p>
    <p>
      This classification does <em>not</em> close the whole plan. The two-source carried deficit
      remains non-zero and belongs to later source-generation work. It only says the first ordered
      quarantine beat must not be dispatched again as though five WEST candidates were still
      waiting on it.
    </p>

    <h2 id="queries">Exact read instruments</h2>
    <p>
      Every query below ran through <code>GraphClient()</code> on the login node because the graph
      endpoint is login-local. Each completed in less than 0.16 seconds. The complete output,
      parameters-derived candidate list, elapsed time and process exit are retained in
      <code>current_cohort_census-r2.log</code>; it ends <code>EXIT=0</code>. The driver records
      <code>graph_mutations: 0</code> and contains no write clause.
    </p>

    <h3 id="carried-query">Per-source carried census</h3>
    <pre>UNWIND $paths AS source_path
OPTIONAL MATCH (source:StandardNameSource {id: 'dd:' + source_path})
OPTIONAL MATCH (source)-[:PRODUCED_NAME]-&gt;(name:StandardName)
WITH source_path, source,
     [candidate IN collect(DISTINCT {
       id: name.id,
       name_stage: name.name_stage,
       docs_stage: name.docs_stage,
       validation_status: name.validation_status,
       status: name.status
     })
      WHERE candidate.id IS NOT NULL] AS produced_names
WITH source_path, source, produced_names,
     [candidate IN produced_names
      WHERE candidate.name_stage IN ['accepted', 'approved']
        AND candidate.docs_stage IN ['accepted', 'approved']] AS accepted_on_both_axes
RETURN count(*) AS manifest_sources,
       sum(CASE WHEN source IS NOT NULL THEN 1 ELSE 0 END) AS source_nodes,
       sum(CASE WHEN size(accepted_on_both_axes) &gt; 0 THEN 1 ELSE 0 END) AS carried_sources,
       sum(CASE WHEN size(accepted_on_both_axes) = 0 THEN 1 ELSE 0 END) AS untreated_sources,
       collect(CASE WHEN size(accepted_on_both_axes) = 0 THEN {
         source_path: source_path,
         source_status: source.status,
         produced_sn_id: source.produced_sn_id,
         last_error: source.last_error,
         produced_names: produced_names
       } END) AS untreated_rows</pre>
    <p>
      Parameters: <code>$paths</code> is the 342-entry list returned by
      <code>load_sources_file</code> from the committed manifest. Returned:
      <code>manifest_sources=342</code>, <code>source_nodes=342</code>,
      <code>carried_sources=340</code>, <code>untreated_sources=2</code>.
    </p>

    <h3 id="scope-query">Prospective scope census</h3>
    <pre>UNWIND $candidate_ids AS candidate_id
MATCH (name:StandardName {id: candidate_id})
OPTIONAL MATCH (manifest_source:StandardNameSource)-[:PRODUCED_NAME]-&gt;(name)
WHERE manifest_source.id IN $manifest_source_ids
OPTIONAL MATCH (outside_source:StandardNameSource)-[:PRODUCED_NAME]-&gt;(name)
WHERE NOT (outside_source.id IN $manifest_source_ids)
WITH name,
     collect(DISTINCT manifest_source.id) AS manifest_source_ids,
     collect(DISTINCT outside_source.id) AS outside_source_ids
RETURN count(name) AS prospective_candidate_names,
       sum(CASE WHEN size(manifest_source_ids) = 0 THEN 1 ELSE 0 END)
         AS without_direct_manifest_binding,
       sum(CASE WHEN size(manifest_source_ids) = 0 AND size(outside_source_ids) &gt; 0
                THEN 1 ELSE 0 END) AS bound_only_outside_manifest,
       collect(CASE WHEN size(manifest_source_ids) = 0 THEN {
         id: name.id,
         outside_source_ids: outside_source_ids
       } END) AS candidates_without_direct_manifest_binding</pre>
    <p>
      Parameters: <code>$candidate_ids</code> is the 230-identity ordered set returned by the
      current <code>fetch_manifest_source_release_rows</code> projection over the committed
      manifest; <code>$manifest_source_ids</code> is the same 342 paths prefixed by
      <code>dd:</code>. Returned: 230 candidates, 0 without a direct manifest binding, 0 bound
      only outside the manifest.
    </p>

    <h3 id="scope-control">Outside-scope positive control</h3>
    <pre>MATCH (source:StandardNameSource)-[:PRODUCED_NAME]-&gt;(name:StandardName)
WITH name, collect(DISTINCT source.id) AS source_ids
WHERE none(source_id IN source_ids WHERE source_id IN $manifest_source_ids)
RETURN count(name) AS names_bound_only_outside_manifest,
       collect(name.id)[0..5] AS outside_only_examples</pre>
    <p>
      Returned <strong>2,556</strong> names bound only outside the WEST manifest, including
      <code>vacuum_magnetic_vector_potential</code> and
      <code>total_ion_energy_diffusion_coefficient</code>. The prospective-cohort zero therefore
      means the selection is aimed at the manifest, not that the predicate cannot see an outside
      binding.
    </p>

    <h3 id="quarantine-query">Candidate quarantine census</h3>
    <pre>UNWIND $candidate_ids AS candidate_id
MATCH (name:StandardName {id: candidate_id})
RETURN count(name) AS prospective_candidate_names,
       sum(CASE WHEN name.validation_status = 'quarantined' THEN 1 ELSE 0 END)
         AS quarantined,
       sum(CASE WHEN name.validation_status = 'quarantined'
                 AND name.validated_at IS NOT NULL THEN 1 ELSE 0 END)
         AS quarantined_with_observation_time,
       sum(CASE WHEN name.validation_status = 'quarantined'
                 AND name.validated_at IS NULL THEN 1 ELSE 0 END)
         AS quarantined_without_observation_time,
       collect(CASE WHEN name.validation_status = 'quarantined' THEN {
         id: name.id,
         name_stage: name.name_stage,
         docs_stage: name.docs_stage,
         validated_at: name.validated_at,
         validation_issues: name.validation_issues
       } END) AS quarantined_rows</pre>
    <p>
      Returned: 230 candidates, 0 quarantined, 0 dated quarantines and 0 undated
      quarantines.
    </p>

    <h3 id="quarantine-control">Quarantine positive control</h3>
    <pre>MATCH (name:StandardName)
RETURN count(name) AS name_nodes,
       count(name.validation_status) AS names_with_validation_status,
       sum(CASE WHEN name.validation_status = 'quarantined' THEN 1 ELSE 0 END)
         AS quarantined_names,
       collect(CASE WHEN name.validation_status = 'quarantined' THEN name.id END)[0..5]
         AS quarantined_examples</pre>
    <p>
      Returned: 5,130 name nodes, 5,128 with the queried property and 660 quarantined
      names. The first examples include <code>total_prefill_lower_count</code> and
      <code>toroidal_particle_current</code>.
    </p>

    <h2 id="boundary">Boundary and interpretation</h2>
    <p>
      This is a read-only currency audit. It did not run a release, create a candidate, mutate the
      graph, change the manifest, or infer a published count from the 230-name prospective set.
      “Prospective candidate” means the unique current terminal identities returned by the
      committed manifest's release projection. The actual batch size and published count remain
      release dry-run outputs and must be taken at the time of a separately authorised cut.
    </p>
  </main>
</body>
</html>
