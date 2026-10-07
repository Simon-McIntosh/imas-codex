---
name: code/triage
description: Decisions questions for code-file diagnostic relevance through the Jev decisions model
used_by: imas_codex.discovery.code.scorer
dynamic: true
---

{
{% if with_content %}
  "relevance_grade": {
    "type": "score",
    "instructions": "How much does the source file at `file.path` matter to someone mapping this facility's diagnostic signals, machine description and data access into IMAS?",
    "criteria": [
      "Unrelated: generic utilities, vendored libraries, build or test scaffolding, or physics simulation with no measured inputs.",
      "Peripheral: plotting, GUI or file-format helpers that only pass diagnostic data through without interpreting it.",
      "Supporting: facility-specific code that touches measured data or the machine description indirectly, such as wrappers, configuration or output writers.",
      "Direct: reads measured diagnostic or shot data from a facility data system, processes measured signals, or carries sensor and machine geometry.",
      "Core: the primary reader, processor or definition for a diagnostic or for the machine description, which a mapper would have to read to get the mapping right."
    ]
  },
  "data_access_depth": {
    "type": "score",
    "instructions": "How deeply does the source file at `file.path` access the facility's measured-data systems?",
    "criteria": [
      "No access to measured data.",
      "Mentions or configures a data system without reading data.",
      "Reads measured data through a higher-level wrapper.",
      "Calls the facility data-system API directly to read signals or shot records.",
      "Implements or declares the data-access interface itself."
    ]
  },
  "signal_processing_depth": {
    "type": "score",
    "instructions": "How much does the source file at `file.path` transform measured diagnostic signals?",
    "criteria": [
      "No processing of measured signals.",
      "Trivial handling such as unit scaling or copying.",
      "Filtering, calibration or correction of measured signals.",
      "Fitting, inversion or reconstruction constrained by measured signals."
    ]
  },
  "machine_description_depth": {
    "type": "score",
    "instructions": "How much of this facility's machine or diagnostic description does the source file at `file.path` carry?",
    "criteria": [
      "None.",
      "Refers to geometry or channel maps defined elsewhere.",
      "Reads sensor positions, coil or vessel geometry, or channel maps from an input file.",
      "Defines sensor positions, lines of sight, coil, limiter or vessel geometry, channel maps or calibration constants itself."
    ]
  },
  "imas_mapping_depth": {
    "type": "score",
    "instructions": "How directly does the source file at `file.path` map this facility's data to or from IMAS?",
    "criteria": [
      "No IMAS involvement, or only a generic copy of an IMAS or ITM access library.",
      "Uses IMAS types inside another code without mapping facility data.",
      "Fills or reads IMAS IDS fields for this facility's data.",
      "Defines a facility-specific mapping table between local signals and IMAS paths."
    ]
  },
{% endif %}
  "loads_diagnostic_data": {
    "type": "noul",
    "instructions": [
      "Does the source file at `file.path` read measured experimental data from the facility's data systems?",
      "`facility` lists this facility's primary data system, its data-access tools and the code patterns that call them. Calling any of these, or another shot or pulse database interface (for example MDSplus mdsopen/mdsvalue/tdi, PPF, JPF, or a facility database wrapper or MEX file), counts as reading measured data.",
      "A header, module or wrapper that declares the interface used to read measured data also counts.",
      "Measured data means signals recorded by plasma diagnostics or plant sensors during a discharge, or quantities derived from them and stored per shot.",
      "Reading model inputs, code configuration or simulation output does not count."
    ],
    "criteria": {
      "true": "the file reads measured shot data, or declares the interface that does",
      "false": "the file never reads measured shot data"
    }
  },
  "processes_diagnostic_signals": {
    "type": "noul",
    "instructions": [
      "Does the source file at `file.path` process measured diagnostic signals?",
      "Processing means calibrating, filtering, correcting, fitting, inverting or analysing measured signals, including reconstruction or profile fitting constrained by measurements.",
      "Plotting measured data without transforming it, solving model equations without measured inputs, and generic numerical libraries do not count."
    ],
    "criteria": {
      "true": "the file transforms or analyses measured signals",
      "false": "the file does not transform measured signals"
    }
  },
  "describes_machine_or_diagnostics": {
    "type": "noul",
    "instructions": [
      "Does the source file at `file.path` define or read the description of this facility's machine or diagnostics?",
      "That covers sensor and probe positions, diagnostic lines of sight, coil, vessel, limiter and divertor geometry, channel-to-signal maps, calibration constants, and the sign, unit or coordinate conventions of measured signals.",
      "Generic geometry utilities and plotting do not count unless they carry this facility's own description."
    ],
    "criteria": {
      "true": "the file carries or reads the facility's machine or diagnostic description",
      "false": "it does not"
    }
  },
  "maps_to_imas": {
    "type": "noul",
    "instructions": [
      "Does the source file at `file.path` map this facility's data into or out of IMAS?",
      "That covers filling or reading IMAS IDS fields for this facility, and facility-specific mapping tables between local signals and IMAS paths.",
      "A generic copy of the IMAS or ITM access-layer library, and IMAS types used inside a simulation code, do not count."
    ],
    "criteria": {
      "true": "the file maps this facility's data to or from IMAS",
      "false": "it does not"
    }
  },
  "reads_or_writes_reconstruction_db": {
    "type": "noul",
    "instructions": [
      "Does the source file at `file.path` read or write the result records of an equilibrium or plasma-reconstruction database?",
      "In scope: the reader, writer or interface that loads or stores reconstruction-result records — for example the EQDBMS `EQDBGET` interface, the `eqrdNN` readers, the `eqwtNN` writers and the `_BF` field-list includes that name the stored fields — and equivalent facility database access code.",
      "Out of scope: computing or solving the reconstruction itself. A solver whose inputs are model parameters rather than stored result records is a simulation and is marked by the simulation question, not this one.",
      "Reading a reconstruction result (for example a stored equilibrium or q-profile) to use it as an input to analysis or mapping is in scope and is not the same as computing it."
    ],
    "criteria": {
      "true": "the file reads or writes equilibrium or plasma-reconstruction result database records, or declares the interface that does",
      "false": "the file does not read or write reconstruction-result records"
    }
  },
  "is_simulation": {
    "type": "noul",
    "instructions": [
      "Is the source file at `file.path` part of a predictive or forward simulation code, or the internals of a physics solver, whose inputs are model parameters rather than measured signals?",
      "Examples: transport codes (TRANSP, ASTRA, JETTO), Fokker-Planck or heating codes, MHD stability codes, orbit-following codes, and equilibrium-solver internals that do not touch measurements."
    ]
  },
  "role": {
    "type": "choice",
    "instructions": "Which role best describes the source file at `file.path`?",
    "criteria": {
      "diagnostic_data_access": "reads or retrieves measured diagnostic or shot data from a facility data system",
      "signal_processing": "calibrates, corrects, fits or analyses measured signals",
      "machine_description": "sensor positions, lines of sight, coil, vessel or limiter geometry, channel maps, calibration constants or signal conventions",
      "imas_mapping": "maps this facility's data to or from IMAS IDSs",
      "simulation_or_solver": "predictive simulation or physics-solver internals",
      "visualization": "plots or displays data",
      "control_or_operations": "real-time control, plant operation or hardware configuration",
      "infrastructure_or_utility": "build files, tests, generic utilities, vendored libraries, data-system internals"
    }
  }
}