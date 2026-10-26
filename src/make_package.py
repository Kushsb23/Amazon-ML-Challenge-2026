"""Build <team>_submission.zip in the layout required by the challenge README.

    python make_package.py TEAM_NAME        # run from src/; writes ../../<TEAM_NAME>_submission.zip (next to the repo)
"""
import os, sys, zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
team = sys.argv[1] if len(sys.argv) > 1 else "TEAM_NAME"
out = os.path.join(os.path.dirname(REPO), f"{team}_submission.zip")
files = [(os.path.join(REPO, "output", f), f"output/{f}") for f in ("matching_results.tsv", "candidate_pairs.tsv")]
for sub in ("src", "notebooks"):
    for dp, dn, fn in os.walk(os.path.join(REPO, sub)):
        dn[:] = [d for d in dn if d not in ("__pycache__", ".ipynb_checkpoints")]
        for f in fn:
            if f.endswith((".py", ".ipynb")):
                p = os.path.join(dp, f)
                files.append((p, "code/business_entity_resolution/" + os.path.relpath(p, REPO).replace(os.sep, "/")))
for f in ("README.md", "requirements.txt"):
    files.append((os.path.join(REPO, f), f"code/business_entity_resolution/{f}"))
files.append((os.path.join(REPO, "docs", "methodology.md"), "Documentation_template.md"))
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p, arc in files:
        z.write(p, arc)                      # zip root = output/, code/, Documentation_template.md
with zipfile.ZipFile(out) as z:
    bad = z.testzip()
    names = z.namelist()
print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB, {len(names)} files, integrity {'OK' if bad is None else 'BAD ' + bad})")
for n in names:
    print("  ", n)
