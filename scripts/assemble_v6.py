"""Assemble the v6 grounded submission: merge the 4 grounded-VQA shards and graft the
banked v1 captions (best captions we have; v4/Cosmos captions hurt). Emits a co-located
submission pair under <v6>/merged/ ready for validate_submission.py + eval-server upload.
"""
import json, glob, shutil
from pathlib import Path

V6_DIR  = "/ptmp/anujs/savana/aicity-outputs/infer_grounded_v6"
V1_CAP  = "/ptmp/anujs/savana/aicity-outputs/realtest_v1_cap/merged/submission_captions.json"
VQA_GT  = "/ptmp/anujs/savana/aicity-data/wts_real_test/WTS_VQA_PUBLIC_TEST.json"
OUT     = Path(V6_DIR) / "merged"
OUT.mkdir(parents=True, exist_ok=True)

# 1) expected question ids (ground truth enumeration)
exp = set()
for e in json.loads(Path(VQA_GT).read_text()):
    for ph in e.get("event_phase", []):
        for q in ph.get("conversations", []):
            exp.add(q["id"])

# 2) merge VQA shards (dedupe by id)
shards = sorted(glob.glob(f"{V6_DIR}/shard_*_of_*/submission_vqa.json"))
print(f"shards found : {len(shards)}")
ans = {}
for s in shards:
    rows = json.loads(Path(s).read_text())
    for r in rows:
        ans.setdefault(r["id"], r["correct"])
    print(f"  {s}: {len(rows)} rows")

miss  = exp - set(ans)
extra = set(ans) - exp
print(f"merged ids   : {len(ans)}   expected: {len(exp)}   missing: {len(miss)}   extra: {len(extra)}")

# 3) complete + sanitize: fill any missing with 'a', coerce non-abcd to 'a'
for qid in miss:
    ans[qid] = "a"
fixed = 0
for qid, v in list(ans.items()):
    if v not in ("a", "b", "c", "d"):
        ans[qid] = "a"; fixed += 1
merged = [{"id": qid, "correct": ans[qid]} for qid in exp]   # exactly the expected set, GT order
print(f"final VQA    : {len(merged)}   filled-missing: {len(miss)}   coerced-bad: {fixed}")
(OUT / "submission_vqa.json").write_text(json.dumps(merged, indent=2))

# 4) graft banked v1 captions
cap = json.loads(Path(V1_CAP).read_text())
print(f"captions(v1) : {len(cap)} keys  <- {V1_CAP}")
shutil.copy(V1_CAP, OUT / "submission_captions.json")

# 5) answer distribution sanity (a degenerate all-'a' would scream here)
from collections import Counter
print("answer dist  :", dict(Counter(m["correct"] for m in merged)))
print(f"\nREADY:\n  --captions {OUT}/submission_captions.json\n  --vqa      {OUT}/submission_vqa.json")
