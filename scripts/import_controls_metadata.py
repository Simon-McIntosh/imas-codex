"""Import a private controls snapshot into the feed-covered archive pilot."""

import argparse
import json
from pathlib import Path

from iter_codex.discovery.signals.controls_metadata import import_controls_metadata
from iter_codex.embeddings.encoder import Encoder
from iter_codex.graph.client import GraphClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument(
        "--settings",
        type=Path,
        required=True,
        help="Private JSON containing an independently known control_name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Private import receipt; contains excluded selectors",
    )
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"complete": False}))
    with GraphClient() as graph:
        receipt = import_controls_metadata(
            args.response,
            args.catalogue,
            positive_control=settings["control_name"],
            graph=graph,
            encoder=Encoder(),
        )
    receipt["complete"] = True
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "complete",
                    "entries_read",
                    "attached",
                    "signals_added",
                    "nodes_changed",
                    "left_out",
                    "with_unit",
                    "name_only_descriptions",
                    "empty_descriptions",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
