"""Repeatable rename of the project's product surfaces.

Commands::

    python -m scripts.rename apply  [--tree DIR] [--dry-run] [--report FILE]
    python -m scripts.rename check  [--tree DIR] [--json]
    python -m scripts.rename cutover [--file FILE] [--json]
    python -m scripts.rename digest [--tree DIR]

`apply` rewrites the tracked tree; `check` performs the closing search and
holds the kept-name list to account; `cutover` proves the written sequence
accounts for every surface of the consumer inventory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.rename import check as check_cmd, cutover as cutover_cmd
from scripts.rename.apply import KEPT_LIST, apply_rename, tree_digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.rename")
    sub = parser.add_subparsers(dest="command", required=True)

    apply_parser = sub.add_parser("apply", help="rewrite the tree to the new name")
    apply_parser.add_argument("--tree", type=Path, default=Path("."))
    apply_parser.add_argument("--kept-list", type=Path, default=KEPT_LIST)
    apply_parser.add_argument("--dry-run", action="store_true")
    apply_parser.add_argument("--report", type=Path)

    check_parser = sub.add_parser("check", help="closing search and kept-name check")
    check_parser.add_argument("--tree", type=Path, default=Path("."))
    check_parser.add_argument("--kept-list", type=Path, default=KEPT_LIST)
    check_parser.add_argument("--json", action="store_true")

    cutover_parser = sub.add_parser("cutover", help="inventory accounting check")
    cutover_parser.add_argument("--file", type=Path, default=cutover_cmd.CUTOVER_FILE)
    cutover_parser.add_argument("--json", action="store_true")

    digest_parser = sub.add_parser("digest", help="digest of every file under a tree")
    digest_parser.add_argument("--tree", type=Path, default=Path("."))

    args = parser.parse_args(argv)

    if args.command == "apply":
        report = apply_rename(args.tree, kept_list=args.kept_list, dry_run=args.dry_run)
        payload = report.as_dict()
        text = json.dumps(payload, indent=2)
        if args.report:
            args.report.write_text(text + "\n")
        print(text)
        return 0

    if args.command == "check":
        return check_cmd.main(args.tree, kept_list=args.kept_list, as_json=args.json)

    if args.command == "cutover":
        return cutover_cmd.main(args.file, as_json=args.json)

    print(tree_digest(args.tree))
    return 0


if __name__ == "__main__":
    sys.exit(main())
