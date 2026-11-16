# Survey two sources of meaning for ITER plant signals

## Goal

Say what the MINT plotting suite installed on this cluster, and the operator
display files behind the web operator-interface site, can contribute to finding
and describing ITER archive signals, and how each would be read.

## Why

The archive describes almost every variable with the same two words, so search
by meaning has nothing to index. The lead named two places that may hold
meaning already: MINT, the plotting tool operators use against the same
archive, and the web operator-interface site, which shows plant topology with
signal names on it. The aim is a working prototype quickly, so the question is
what can be used now, not what is ideal.

## What is known

- MINT is installed as a module; its newest tree is under
  `/work/imas/opt/EasyBuild/software/MINT/` and carries the packages `mint`,
  `iplotlib`, `iplotDataAccess`, `iplotProcessing`, `iplotWidgets`, and a
  server-sent-events client.
- The web operator-interface site sits behind the same sign-in gateway as the
  other ITER sites. Signed in, its application path answers with an
  application bootstrap document, not a page: it is a server-rendered
  application, so its content is not reachable by fetching pages. The display
  definitions it renders are files, and those files are the thing to find.

## Questions to answer

For MINT, by reading the installed code and its packaged data, with no live
request:

1. How does its data-access layer find variables: does it call a search or
   browse service, and if so what does that service take and return (name
   patterns, descriptions, units, a hierarchy)? Name the functions and the
   configuration keys, and any service address only by its configuration key.
2. Does it carry or read saved workspaces, signal sets, aliases, units or
   descriptions that group signals by purpose? Give counts and the shape of
   one, with real names replaced.
3. How does it turn a pulse or a time range into a request, and does it list
   pulses or categories?
4. What processing expressions does it support for combining signals, since an
   agent composing signals should speak the same language where one exists.

For the operator displays:

5. Where do display definition files live on this cluster or in a repository
   reachable from it? Search named roots only, with a time bound on every
   search, and report the roots searched.
6. In a sample of such files, what do they hold per widget: variable names,
   labels, units, tooltips, the display's title and its links to other
   displays? State what fraction of variable names in a sample also appear in
   the private archive catalogue.

## Rules

- Read only. No request to any archive, service or site; no sign-in.
- Every search names its root and carries a time bound. Never search from the
  filesystem root or a shared-storage root.
- Real hosts, variable names, display names and titles go only into a private
  notes file under `~/.cache/imas-codex-codac-uda/meaning-sources/`. The
  tracked evidence fragment gives counts, shapes and placeholders.
- State what you could not determine and why. An absent finding is reported as
  not found in the roots searched, never as absent.

## Done when

The evidence fragment answers each numbered question with the files read and
counts, ends with a ranked list of what could feed a prototype this week and
what each would need, and names the private notes file. One table compares the
two sources by what they give per signal: name, label, unit, description,
grouping, topology.

## Scope

Write paths: the evidence fragment
`docs/evidence/fragments/codac-uda-signals/survey-plant-signal-meaning-sources.html`.
No product source changes.
