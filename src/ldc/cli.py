"""``ldc <command>``: one command per step of the post's experiments."""

from __future__ import annotations

import argparse
import importlib

COMMANDS = {
    "convert": ("ldc.convert_te", "build the windowed-statistics dataset from the Rieth .RData files"),
    "train-sup": ("ldc.train_sup", "train the supervised reference encoder"),
    "train-ssl": ("ldc.train_ssl", "LeJEPA training, from scratch or continued from an encoder"),
    "diagnose": ("ldc.experiments.diagnose", "probe, geometry and Fisher criterion of a run"),
    "grad": ("ldc.experiments.grad", "class-conditional SIGReg gradient on the supervised embedding"),
    "affine": ("ldc.experiments.affine", "invertible affine control"),
    "mask-audit": ("ldc.experiments.mask_audit", "fault signal shared by the two views"),
    "report": ("ldc.report", "tables and figures of the post from the runs"),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ldc", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")
    for name, (module, help_) in COMMANDS.items():
        if name == "convert":  # has its own argparse
            sub.add_parser(name, help=help_, add_help=False)
            continue
        p = sub.add_parser(name, help=help_, description=help_)
        importlib.import_module(module).add_args(p)
    args, rest = ap.parse_known_args(argv)
    module = importlib.import_module(COMMANDS[args.command][0])
    if args.command == "convert":
        module.main(rest)
    elif rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    else:
        module.main(args)


if __name__ == "__main__":
    main()
