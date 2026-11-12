import json, os, statistics, subprocess, sys, time
PY_BIN = "/home/ITER/mcintos/Code/imas-codex/.venv/bin/python"
ARMS = {
    "base": ("/home/ITER/mcintos/.cache/imas-codex-saicl/base-tree", "980fa78b5"),
    "new": ("/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/07a8cbfa-1dc5-41ed-bf84-0d88174bdbe9/saicl-lazy-command-group", None),
}
STEPS = [
    ("interpreter", "pass"),
    ("import imas_codex.cli", "import imas_codex.cli"),
    ("resolve serve through the group",
     "import click\nfrom imas_codex.cli import main\nassert main.get_command(click.Context(main), 'serve') is not None"),
    ("import imas_codex.llm.server",
     "import click\nfrom imas_codex.cli import main\nmain.get_command(click.Context(main), 'serve')\nimport imas_codex.llm.server"),
    ("construct AgentsServer(dd_only=True)",
     "import click\nfrom imas_codex.cli import main\nmain.get_command(click.Context(main), 'serve')\nimport imas_codex.llm.server\nfrom imas_codex.llm.server import AgentsServer\nAgentsServer(dd_only=True)"),
]
out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename,
       "python": PY_BIN, "runs_per_step": 3, "arms": {}}
for arm, (tree, rev) in ARMS.items():
    env = {**os.environ, "PYTHONPATH": tree, "UV_NO_SYNC": "1",
           "UV_PROJECT_ENVIRONMENT": "/home/ITER/mcintos/Code/imas-codex/.venv"}
    rev = rev or subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    steps = []
    for name, src in STEPS:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = subprocess.run([PY_BIN, "-c", src], cwd=tree, env=env, capture_output=True, text=True)
            times.append(round(time.perf_counter() - t0, 3))
            assert r.returncode == 0, (name, r.stderr[-800:])
        steps.append({"step": name, "seconds": times, "median": statistics.median(times)})
    base = steps[0]["median"]
    prev = base
    for s in steps[1:]:
        s["incremental"] = round(s["median"] - prev, 3)
        prev = s["median"]
    out["arms"][arm] = {"tree": tree, "revision": rev, "loadavg": open("/proc/loadavg").read().split()[:3], "steps": steps}
json.dump(out, open("/home/ITER/mcintos/.cache/imas-codex-saicl/parts.json", "w"), indent=1)
print(json.dumps(out, indent=1))
