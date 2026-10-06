---
name: mapping/candidate_judgment
description: Decisions questions for IDS routing and (signal-source, DD-candidate) quantity judgment through the Jev decisions model
used_by: imas_codex.ids.candidates
dynamic: true
---

{
  "ids_routing": {
    "type": "choice",
    "instructions": "Which IMAS IDS would hold the values this signal source provides?"
  },
  "same_quantity": {
    "type": "noul",
    "instructions": "Would the IMAS field at {ref} correctly hold the values that signal_source provides? True: storing them there needs at most a unit conversion, a sign or COCOS flip, or the selection of one array element. False: the field holds a different quantity, a different component or coordinate, a different object, or only a related or derived quantity.",
    "criteria": {"true": "yes", "false": "no"}
  }
}