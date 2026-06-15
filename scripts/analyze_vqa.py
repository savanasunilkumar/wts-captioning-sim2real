"""VQA error analysis on SynWTS val (has GT): break accuracy down by question
TYPE, and compare against the majority-class PRIOR. Reveals where we lose points
and whether failures are visual (acc >> prior) or prior-driven (acc ~ prior).
No GPU -- uses saved predictions + GT + the dataset's question text.

Usage: python analyze_vqa.py <pred_dir_glob>
  pred_dir must contain submission_vqa.json (id,correct) and vqa_gt.json (id,correct).
"""
import json, glob, sys
from collections import defaultdict, Counter
sys.path.insert(0, "/work/anujs/savana/aicity-track2/scripts")
from wts_dataset import WTSDataset

pred_glob = sys.argv[1] if len(sys.argv) > 1 else "/ptmp/anujs/savana/aicity-outputs/abtest_simple/shard_*"
pred, gt = {}, {}
for f in glob.glob(pred_glob + "/submission_vqa.json"):
    for x in json.load(open(f)): pred[x["id"]] = x["correct"]
for f in glob.glob(pred_glob + "/vqa_gt.json"):
    for x in json.load(open(f)): gt[x["id"]] = x["correct"]
print(f"loaded preds={len(pred)} gt={len(gt)} from {pred_glob}")

# reconstruct question text per id (id format: f"{sid}_{view}_{file_id}_{qidx}")
ds = WTSDataset("/ptmp/anujs/savana/aicity-data/synwts", "val")
qtext = {}
for sid in ds.scenarios():
    for q in ds.load_vqa(sid):
        qid = f"{sid}_{q.view}_{q.file_id}_{q.question_idx}"
        qtext[qid] = q.question.strip().lower()

# group by question TYPE (normalized question text)
by_type = defaultdict(lambda: {"n": 0, "correct": 0, "gt": Counter()})
matched = 0
for qid in gt:
    if qid not in pred:
        continue
    typ = qtext.get(qid)
    if typ is None:
        typ = "<<unmatched id>>"
    else:
        matched += 1
    bt = by_type[typ]
    bt["n"] += 1
    bt["correct"] += int(pred[qid] == gt[qid])
    bt["gt"][gt[qid]] += 1

rows = []
for typ, d in by_type.items():
    acc = 100 * d["correct"] / d["n"]
    prior = 100 * max(d["gt"].values()) / d["n"]   # majority-class accuracy = blind prior
    rows.append((d["n"], acc, prior, acc - prior, typ))
rows.sort(reverse=True)

print(f"\nmatched {matched}/{len(gt)} ids to question text\n")
print(f"{'N':>5} {'ourAcc':>7} {'prior':>6} {'lift':>6}  question-type (visual value = lift над prior)")
print("-" * 100)
tot_n = tot_c = tot_prior_c = 0
for n, acc, prior, lift, typ in rows:
    flag = "  <-- AT/BELOW PRIOR (no visual use)" if lift <= 1 else ""
    print(f"{n:>5} {acc:>6.1f}% {prior:>5.1f}% {lift:>+5.1f}  {typ[:60]}{flag}")
    tot_n += n; tot_c += round(acc*n/100); tot_prior_c += round(prior*n/100)
print("-" * 100)
print(f"OVERALL  ours={100*tot_c/tot_n:.1f}%   blind-prior={100*tot_prior_c/tot_n:.1f}%   (lift from video = {100*(tot_c-tot_prior_c)/tot_n:+.1f} pts)")
