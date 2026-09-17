<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="docs-project" content="imas-codex">
  <meta name="reckon-type" content="evidence">
  <meta name="plan-slug" content="unbound-source-backlog-refusal-adjudication">
  <meta name="plan-title" content="Unbound source backlog &mdash; the 91 refusals adjudicated">
  <meta name="plan-evidence-for" content="unbound-source-backlog">
  <title>Refusal adjudication | imas-codex</title>
  <link rel="stylesheet" href="/_shared/foundation.css">
  <link rel="stylesheet" href="/_shared/dashboard.css">
</head>
<body>
<main class="plan-doc">
  <h1>Unbound source backlog &mdash; the 91 refusals adjudicated</h1>

  <p><strong>Of the 91 inadmissible rows of the extracted-unbound cohort, 65 are test-fixture
  identities no admission rule ever decided and 26 are refusals a rule did decide</strong>
  (<code>65 + 26 = 91</code>). Of the 26, <strong>3 have their failing condition read from the
  parent identity and are returned by a parent repair</strong>, and <strong>23 have it read from
  the row's own node and are not. </strong></p>

  <p>This page is the disposition the composition pass needs: it names the 26 rows it cannot
  compose and says which of them a parent-side repair would return, so the pass can be sized
  against the rest and the two bare bases can be routed as a grammar question rather than
  re-run.</p>

  <p>Read-only throughout: <strong>no graph write was performed</strong>, before or after the
  measurement. Row set and per-row attributes are the cohort the eligibility census captured at
  revision <code>3d80e116f4c219519ddf5ece66c1bed8c8b23e8e</code> on <strong>2026-09-17</strong>;
  the adjudication instrument re-read the graph for the parent identities only.</p>

  <h2>The two groups, and the 26 by who owns the refusal</h2>

  <figure>
    <img src="/imas-codex/figures/unbound-source-backlog/refusal-adjudication.svg"
         alt="Flow diagram decomposing the 91 inadmissible rows into 65 test-fixture identities
         and 26 rule-driven refusals, and the 26 further into 3 whose parent identity owns the
         failing condition and 23 whose own node does"
         style="max-width:100%">
    <figcaption>The 91 decompose first by whether a rule decided the refusal, then by which
    entity the failing gate reads. Only the 3 rows in the pale box are returned by a repair to
    something other than the refused row itself.</figcaption>
  </figure>

  <table>
    <caption>Instrument <code>adjudicate.py</code>, run 2026-09-17. The three rows sum as stated
    by construction: every inadmissible row is classified once.</caption>
    <thead><tr><th>Group</th><th>Rows</th><th>Decided by</th></tr></thead>
    <tbody>
      <tr><td>Test-fixture identities</td><td><strong>65</strong></td><td>no admission rule; the row's own id and <code>source_id</code></td></tr>
      <tr><td>Rule-driven refusals</td><td><strong>26</strong></td><td>the first failing admission gate</td></tr>
      <tr><td>Recoverable by a parent repair</td><td><strong>3</strong></td><td>gate reads the parent identity</td></tr>
      <tr><td>Not recoverable by a parent repair</td><td><strong>23</strong></td><td>gate reads the row's own node</td></tr>
      <tr><td><strong>Inadmissible</strong></td><td><strong>91</strong></td><td><code>65 + 26</code>, and <code>3 + 23</code></td></tr>
    </tbody>
  </table>

  <h2>Part 1 &mdash; the 65 test-fixture identities, individually</h2>

  <p>The discriminator is not a guess about a name: it is the row's own identity against
  prefixes the codebase <em>declares</em> as fixture identities. Every one of the 65 is listed
  below.</p>

  <h3>The predicate</h3>

  <p><code>imas_codex/standard_names/fixture_sources.py</code>, read at HEAD,
  declares the fixture namespace and says in its own docstring that such rows
  "live in the graph beside real data":</p>

  <pre><code>FIXTURE_SOURCE_ID_PREFIX = "dd:test_review_entry__"
FIXTURE_SOURCE_PATH_PREFIX = "test/"</code></pre>

  <p>Three facts are true of every row below, and any one of them alone is sufficient:
  its node id begins with <code>dd:test_review_entry__</code> (64 rows) or is exactly
  <code>dd:__revreltest__/leaf1</code> (1 row); its <code>source_id</code> is the literal
  <code>test/path</code> (64 rows) or <code>__revreltest__/leaf1</code> (1 row); and it
  resolves to no <code>IMASNode</code> at all, so no Data Dictionary quantity stands behind
  it.</p>

  <p>They are not uniform in source family: <strong>59 are <code>derived</code></strong> (so
  the census applies the parent-admission gate to them and refuses them for it) and
  <strong>6 are <code>dd</code></strong> (refused with <code>missing_dd_authority</code>,
  which is what a source with no resolvable Data Dictionary path is refused for). Neither
  family is a real row: the two share the identity literals, and the family is decided at the
  seeding call site that used them.</p>

  <h3>Where they come from (code read, not inference)</h3>

  <p>Both literals are seeded by graph-marked tests, and each seeding site is readable:</p>

  <ul>
    <li><code>tests/standard_names/test_reviewable_name_stage.py:105-110</code> merges
    <code>StandardNameSource {id: $src_id}</code> and sets
    <code>sns.source_id = 'test/path'</code>; the id is formed at
    <code>:182,213,323,338</code> as <code>f"dd:{_uid('src')}"</code> where
    <code>_uid</code> prefixes <code>test_review_entry__</code>.</li>
    <li><code>tests/standard_names/test_review_release.py:1005</code> sets
    <code>PREFIX = "__revreltest__"</code>, <code>LEAF = f"{PREFIX}/leaf1"</code> and merges
    <code>StandardNameSource {id: 'dd:' + $leaf}</code> &mdash; exactly
    <code>dd:__revreltest__/leaf1</code>.</li>
  </ul>

  <p><strong>The rows survive their own teardown, and the reason is a prefix mismatch.</strong>
  The cleanup fixture at <code>test_reviewable_name_stage.py:52-63</code> deletes
  <code>StandardNameSource</code> rows by <code>n.id STARTS WITH $p</code> with
  <code>$p = "test_review_entry__"</code>, and the fixture at
  <code>test_review_release.py:1024</code> deletes by <code>n.id STARTS WITH "__revreltest__"</code>.
  The seeded source ids both carry the <code>dd:</code> prefix, so neither predicate matches them:
  the name each source was bound to is deleted, the source row is not. What is left behind is a
  <code>StandardNameSource</code> with no producer and no Data Dictionary node &mdash; the exact
  shape the census counted as unbound and this page disposes of.</p>

  <table>
    <caption>All 65 fixture rows, each with its id, its <code>source_id</code>, its source
    family, the gate it fails and its resolution. Produced by <code>adjudicate.py</code> from the census
    cohort; the fixture predicate is stated above.</caption>
    <thead><tr><th>Node id</th><th><code>source_id</code></th><th>Family</th><th>Fails</th><th>Evidence</th></tr></thead>
    <tbody>
      <tr><td><code>dd:__revreltest__/leaf1</code></td><td><code>__revreltest__/leaf1</code></td><td>dd</td><td><code>missing_dd_authority</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_0b289e0c</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_1cadb8eb</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_1f846004</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_221d35d4</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_241fc8f6</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_255716f2</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_2fb0d37c</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_324c3237</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_329cbaf6</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_367c5bf9</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_386115d9</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_392119ec</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_45dfce56</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_474635ba</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_4b1958f4</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_526d9cf1</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_58035398</code></td><td><code>test/path</code></td><td>dd</td><td><code>missing_dd_authority</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_596e54cc</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_5be77702</code></td><td><code>test/path</code></td><td>dd</td><td><code>missing_dd_authority</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_5beef582</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_6627852d</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_66aab77a</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_6e521dfb</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_6fbb5b84</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_70409727</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_70f4e2aa</code></td><td><code>test/path</code></td><td>dd</td><td><code>missing_dd_authority</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_71cfab89</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_76d1205f</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_79221aab</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_7b687093</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_7d1566d3</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_7e584d43</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_7fa0eade</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_8549e34e</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_857ad820</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_877c50b7</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_8eef1670</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_91998248</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_9963657a</code></td><td><code>test/path</code></td><td>dd</td><td><code>missing_dd_authority</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_9ade5078</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_9bfd9176</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_9ddaa9ee</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_aa7db75a</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_b06c1eed</code></td><td><code>test/path</code></td><td>dd</td><td><code>missing_dd_authority</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_b0a24ec8</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_b2f05ddc</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_b5cd0829</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_bbf2f3de</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_bc02515a</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_c430e19f</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_c5eb7f3a</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_ca1fbfb1</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_ca5ad00e</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_cd95ead7</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_cea3f1b2</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_d5772673</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_d6d9cfaa</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_dc6a16f1</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_dcb74eec</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_dec0bd5e</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_e9e9dfee</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_f797b3aa</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_f95a8a69</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
      <tr><td><code>dd:test_review_entry__src_faad8188</code></td><td><code>test/path</code></td><td>derived</td><td><code>derived_parent_inadmissible</code></td><td>resolves to no <code>IMASNode</code>; <code>attempt_count=0</code></td></tr>
    </tbody>
  </table>

  <p><strong>Positive control on the predicate.</strong> Applied to the 1,236 cohort rows that
  <em>do</em> resolve to an <code>IMASNode</code>, the fixture predicate flags
  <strong>0</strong>. A predicate that matched everything would be measuring nothing; this one
  separates 65 from 1,236.</p>

  <h2>Part 2 &mdash; the 26 refusals a rule decided</h2>

  <p>For each row the question is where the failing condition lives. The census instrument
  applies each gate to a named subject, and this table records that subject as the gate is
  written in code: a dd gate reads the row's own <code>IMASNode</code> (its
  <code>node_category</code>, its <code>lifecycle_status</code>, its <code>data_type</code>)
  or its IDS membership through <code>qualify_dd</code>; the derived gate calls
  <code>is_admissible_parent_name(source_id, graph)</code>, which reads the <em>parent
  identity and its children</em> and never the row.</p>

  <p>So a parent repair returns a row exactly when the gate reads something other than the row's
  own node &mdash; which is true of the 3 <code>derived</code> rows and of none of the 23
  <code>dd</code> rows: no dd gate in the list has an ancestor in its predicate, so no change to a
  parent can clear one.</p>

  <table>
    <caption>The 26 rule-driven refusals, each with the entity its failing gate reads and the
    adjudication. <code>recoverable</code> means a repair to the parent identity clears the gate
    and nothing row-side blocks it.</caption>
    <thead><tr><th><code>source_id</code></th><th>Rule</th><th>Gate reads</th><th>Verdict</th><th>Failing subject</th></tr></thead>
    <tbody>
      <tr><td><code>edge_sources/source/ggd/electrons/energy</code></td><td><code>container_node_no_signal_signature</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.data_type + row children</code></td></tr>
      <tr><td><code>mhd_linear/time_slice/toroidal_mode/plasma/psi_potential_perturbed</code></td><td><code>container_node_no_signal_signature</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.data_type + row children</code></td></tr>
      <tr><td><code>plasma_sources/source/ggd/electrons/energy</code></td><td><code>container_node_no_signal_signature</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.data_type + row children</code></td></tr>
      <tr><td><code>magnetics/b_field_pol_probe/non_linear_response/b_field_non_linear</code></td><td><code>dd_lifecycle_removed</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.lifecycle_status</code></td></tr>
      <tr><td><code>power_density</code></td><td><code>derived_parent_inadmissible</code></td><td>parent_identity</td><td><strong>recoverable</strong></td><td><code>parent_identity (source_id) + its children</code></td></tr>
      <tr><td><code>root_mean_square_of_spectral_width_of_spectrometer_channel</code></td><td><code>derived_parent_inadmissible</code></td><td>parent_identity</td><td><strong>recoverable</strong></td><td><code>parent_identity (source_id) + its children</code></td></tr>
      <tr><td><code>wavelength</code></td><td><code>derived_parent_inadmissible</code></td><td>parent_identity</td><td><strong>recoverable</strong></td><td><code>parent_identity (source_id) + its children</code></td></tr>
      <tr><td><code>core_instant_changes/change/profiles_1d/rotation_frequency_tor_sonic</code></td><td><code>duplicate_ids</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node IDS membership (qualify_dd reason_code)</code></td></tr>
      <tr><td><code>edge_transport/model/ggd/neutral/state/momentum</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>equilibrium/time_slice/constraints/b_field_tor_vacuum_r/reconstructed</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>equilibrium/time_slice/constraints/pf_current/reconstructed</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>equilibrium/time_slice/constraints/pressure/reconstructed</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>equilibrium/time_slice/constraints/pressure_rotational/reconstructed</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>equilibrium/time_slice/constraints/strike_point/position_reconstructed/r</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>equilibrium/time_slice/constraints/strike_point/position_reconstructed/z</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>equilibrium/time_slice/constraints/x_point/position_reconstructed/r</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>equilibrium/time_slice/constraints/x_point/position_reconstructed/z</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>ferritic/object/axisymmetric/annulus</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>ic_antennas/antenna/module</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>ic_antennas/antenna/module/strap/geometry/oblique</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>ntms/time_slice/mode</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>plasma_transport/model</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>plasma_transport/model/ggd</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>spectrometer_visible/channel/isotope_ratios/isotope</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>summary/pedestal_fits</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
      <tr><td><code>waves/coherent_wave</code></td><td><code>node_category_ineligible</code></td><td>row_node</td><td>not recoverable</td><td><code>row_node.node_category</code></td></tr>
    </tbody>
  </table>

  <h3>The 3 recoverable rows, and what a repair would be</h3>

  <table>
    <caption>The parent identities of the three <code>derived</code> refusals, re-read from the
    live graph 2026-09-17. <code>admit_now</code> is
    <code>is_admissible_parent_name</code>'s own verdict on the identity as it stands.</caption>
    <thead><tr><th>Parent identity</th><th>Stage</th><th>Children</th><th><code>admit_now</code></th><th>What a repair is</th></tr></thead>
    <tbody>
      <tr><td><code>power_density</code></td><td><code>superseded</code></td><td>0</td><td><code>False</code></td><td>identity has 0 children and 0 parents; a repair is a grammar decision</td></tr>
      <tr><td><code>root_mean_square_of_spectral_width_of_spectrometer_channel</code></td><td><code>superseded</code></td><td>0</td><td><code>False</code></td><td>canonical sibling <code>root_mean_square_spectral_width_of_spectrometer_channel</code> exists <strong>accepted</strong> and admits</td></tr>
      <tr><td><code>wavelength</code></td><td><code>exhausted</code></td><td>0</td><td><code>False</code></td><td>identity has 0 children and 0 parents; a repair is a grammar decision</td></tr>
    </tbody>
  </table>

  <p>One of the three has a concrete target that already exists. The refused row names
  <code>root_mean_square_of_spectral_width_of_spectrometer_channel</code>, whose ISN parse lost
  the token <code>of</code> and which is <code>superseded</code> and <code>quarantined</code>. The
  grammatical identity <code>root_mean_square_spectral_width_of_spectrometer_channel</code>
  exists as an <strong>accepted, valid</strong> name, and the same probe admits it as a parent
  (<code>reason: has qualifiers [spectral]</code>) &mdash; so re-sourcing the row onto the
  canonical parent is a repair route the graph already supports.</p>

  <p>The other two, <code>power_density</code> and <code>wavelength</code>, are bare bases with
  <strong>no children and no parent of their own</strong>; a repair means the identity acquiring a
  qualifier or a projection child, which is a grammar decision rather than a graph state. Both are
  also terminal upstream of this cohort: <code>power_density</code> is
  <code>superseded</code>, <code>valid</code>; <code>wavelength</code> is
  <code>exhausted</code>, <code>quarantined</code>. They are the catch-all bases the grammar
  review has already flagged for gating.</p>

  <h2>Controls &mdash; what the two zeros rest on</h2>

  <table>
    <caption>Every zero below is a measured zero with a live positive control beside it.</caption>
    <thead><tr><th>Statement</th><th>Value</th><th>Query / callable</th><th>Positive control</th></tr></thead>
    <tbody>
      <tr><td>Cohort rows inadmissible</td><td>91</td><td><code>q-cohort-extracted-unbound</code> + gates</td><td><code>stale</code> control population refuses 1,820 of 1,820</td></tr>
      <tr><td>Resolved dd rows flagged as fixtures</td><td><strong>0</strong> of 1,236</td><td>fixture predicate over <code>census-rows.json</code></td><td>the same predicate flags 65 rows, so it fires</td></tr>
      <tr><td>dd refusals a parent repair returns</td><td><strong>0</strong> of 23</td><td><code>GATE_SUBJECT</code> read from the gate code</td><td>the same probe returns <strong>3</strong> parent-owned rows</td></tr>
      <tr><td>Parent-repair probe agrees with the gate it mirrors</td><td>3 of 3 derived rows deny now</td><td><code>is_admissible_parent_name</code></td><td><code>electron_temperature</code> &#8594; <code>admit=True</code>; sibling canonical name &#8594; <code>admit=True</code></td></tr>
      <tr><td>Rows with any surviving DD node among them</td><td>0 of 65</td><td>census <code>node_id</code> field</td><td>1,236 cohort rows do carry a <code>node_id</code></td></tr>
    </tbody>
  </table>

  <h2>What this does not establish</h2>

  <ul>
    <li><strong>Recoverable is a conditional, not a schedule.</strong> It says a parent repair
    would clear the parent's own gate. It does not say the repair is worth making: for the two
    bare bases it is a grammar decision, and for all three a compose pass still has to succeed
    after the repair.</li>
    <li><strong>The 23 not-recoverable rows are not defects.</strong> They are a structural
    container, a removed DD node, a duplicated IDS and fit artifacts &mdash; each correctly
    excluded, and each the mechanism the gate exists for.</li>
    <li><strong>The fixture rows are a graph-hygiene finding, not a backlog.</strong> Composing a
    name for <code>test/path</code> would mint a name for a thing that is not a physical quantity.
    Cleanup is a separate act and is named in the manifest's follow-ons rather than performed
    here.</li>
  </ul>

  <h2>Artifacts</h2>

  <ul>
    <li><code>adjudicate.py</code> &mdash; the read-only instrument; classifies each row's gate
    subject and probes the parent of every <code>derived</code> refusal.</li>
    <li><code>adjudicate.log</code> &mdash; the captured run, exit 0, carrying the per-row
    verdicts and both controls.</li>
    <li><code>adjudication-rows.json</code> &mdash; the per-row verdicts this page renders,
    so every count above is re-derivable from the row set rather than transcribed.</li>
    <li><code>placeholder-code-evidence.log</code> &mdash; the code read for Part 1: the
    declared prefix, both seeding sites, both teardown fixtures and the id forms.</li>
    <li><code>refusal-adjudication.svg</code> &mdash; the decomposition, embedded above.</li>
  </ul>
</main>
</body>
</html>
