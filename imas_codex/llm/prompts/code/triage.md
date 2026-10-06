---
name: code/triage
description: Decisions questions for code-file diagnostic relevance through the Jev decisions model
used_by: imas_codex.discovery.code.scorer
dynamic: true
---

{
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