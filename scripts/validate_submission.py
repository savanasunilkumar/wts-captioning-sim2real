"""Validate merged real-test submissions for COVERAGE + FORMAT before submitting.
No ground truth needed -- confirms every expected key/id is present and well-formed,
mirroring how metrics_all.py / the VQA scorer will read our files.

Usage:
  python scripts/validate_submission.py --captions <merged_captions.json> --vqa <merged_vqa.json>
"""
import argparse, json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, "/work/anujs/savana/aicity-track2")
from scripts.infer_realtest import caption_units  # reuse the exact GT-mirroring enumeration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", default="/ptmp/anujs/savana/aicity-data/wts_real_test/unpacked/WTS_DATASET_PUBLIC_TEST")
    ap.add_argument("--vqa-file", default="/ptmp/anujs/savana/aicity-data/wts_real_test/WTS_VQA_PUBLIC_TEST.json")
    ap.add_argument("--captions", required=True)
    ap.add_argument("--vqa", required=True)
    args = ap.parse_args()

    # ---------- CAPTIONS ----------
    units = caption_units(Path(args.test_root))
    expected = {u["key"]: [s[0] for s in u["segs"]] for u in units}
    wts = sum(1 for u in units if u["kind"] == "wts")
    bdd = sum(1 for u in units if u["kind"] == "bdd")
    sub = json.loads(Path(args.captions).read_text())

    print("=== CAPTIONS ===")
    print(f"expected keys: {len(expected)}  ({wts} WTS + {bdd} BDD)   submission keys: {len(sub)}")
    missing_keys = [k for k in expected if k not in sub]
    extra_keys = [k for k in sub if k not in expected]
    print(f"missing keys : {len(missing_keys)} {missing_keys[:5]}")
    print(f"extra keys   : {len(extra_keys)} {extra_keys[:5]}")

    empty_cap = 0
    missing_phase = 0
    total_seg = 0
    for k, phases in expected.items():
        got = {seg.get("labels", [None])[0]: seg for seg in sub.get(k, [])}
        for ph in phases:
            total_seg += 1
            seg = got.get(ph)
            if seg is None:
                missing_phase += 1
            elif not seg.get("caption_pedestrian", "").strip() or not seg.get("caption_vehicle", "").strip():
                empty_cap += 1
    print(f"total expected segments      : {total_seg}")
    print(f"missing (key,phase) segments : {missing_phase}")
    print(f"segments w/ empty ped or veh : {empty_cap}")

    # ---------- VQA ----------
    vdata = json.loads(Path(args.vqa_file).read_text())
    expected_ids = set()
    for e in vdata:
        for ph in e.get("event_phase", []):
            for q in ph.get("conversations", []):
                expected_ids.add(q["id"])
    vsub = json.loads(Path(args.vqa).read_text())
    sub_ids = [x["id"] for x in vsub]
    sub_id_set = set(sub_ids)

    print("\n=== VQA ===")
    print(f"expected ids: {len(expected_ids)}   answers: {len(vsub)}   unique ids: {len(sub_id_set)}")
    miss = expected_ids - sub_id_set
    dup = len(sub_ids) - len(sub_id_set)
    bad_letter = [x["id"] for x in vsub if x.get("correct") not in ("a", "b", "c", "d")]
    print(f"missing ids        : {len(miss)} {list(miss)[:5]}")
    print(f"duplicate ids      : {dup}")
    print(f"non-abcd answers   : {len(bad_letter)} {bad_letter[:5]}")
    print(f"answer distribution: {dict(Counter(x['correct'] for x in vsub))}")

    ok = (not missing_keys and not missing_phase and not empty_cap
          and not miss and not bad_letter and dup == 0)
    print("\n" + ("ALL CHECKS PASSED -- safe to submit" if ok else "ISSUES FOUND -- review above before submitting"))


if __name__ == "__main__":
    main()
