{% if shared_parts %}
## Settled shared wording — write around it, never against it

These parts of the definition are fixed text, stored once for the whole catalog
and installed verbatim by the renderer after you write. They are settled: do not
restate, paraphrase, or duplicate them, and do not write anything they
contradict. Write only what is specific to this standard name, and make your
one-line `description` and every sentence and equation you write consistent with
them. A sentence describing a settled part is written as it reads here.

{% for p in shared_parts %}- **{{ p.part }}** (`{{ p.anchor }}`): "{{ p.sentence }}"{% if p.equation %} — equation: `$${{ p.equation }}$$`{% endif %}
{% endfor %}

If a part states the population a quantity counts, your `description` states the
same population in its own words, and no sentence or equation of yours may
include anything that population excludes. Text that departs from a settled part
is refused before it is stored.
{% endif %}