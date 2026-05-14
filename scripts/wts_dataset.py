"""WTS / SynWTS dataset access utilities."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

VIEWS_CAPTION = ("vehicle_view", "overhead_view")
VIEWS_VQA = ("vehicle_view", "overhead_view", "environment")

# Phase label normalization (both number and name forms appear in the data).
PHASE_NUM_TO_NAME = {
    "0": "pre-recognition",
    "1": "recognition",
    "2": "judgment",
    "3": "action",
    "4": "avoidance",
}
PHASE_NAME_TO_NUM = {
    "pre-recognition": "0",
    "prerecognition": "0",
    "pre_recognition": "0",
    "recognition": "1",
    "judgment": "2",
    "judgement": "2",   # UK spelling variant in SynWTS
    "action": "3",
    "avoidance": "4",
}


def normalize_phase(label: str) -> tuple[str, str]:
    """Return (number, name) for either input form."""
    if label in PHASE_NUM_TO_NAME:
        return label, PHASE_NUM_TO_NAME[label]
    if label in PHASE_NAME_TO_NUM:
        return PHASE_NAME_TO_NUM[label], label
    return label, label  # unknown — pass through


@dataclass
class CaptionSegment:
    phase_num: str
    phase_name: str
    pedestrian: str
    vehicle: str
    start_time: str
    end_time: str


@dataclass
class CaptionView:
    scenario_id: str
    view: str
    video: str
    segments: list[CaptionSegment]


@dataclass
class VQAQuestion:
    scenario_id: str
    view: str
    file_id: int
    question_idx: int
    question: str
    options: dict[str, str]
    correct: str | None = None
    phase_num: str | None = None
    phase_name: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    overhead_videos: list[str] = field(default_factory=list)
    video_file: str | None = None


class WTSDataset:
    def __init__(self, root: str | Path, split: Literal["train", "val", "test"] = "val"):
        self.root = Path(root)
        self.split = split
        self.ann_root = self.root / "data" / "annotations"
        self.vid_root = self.root / "data" / "videos" / split

    def scenarios(self) -> list[str]:
        d = self.ann_root / "caption" / self.split
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    def load_captions(self, scenario_id: str) -> list[CaptionView]:
        out = []
        for view in VIEWS_CAPTION:
            jf = self.ann_root / "caption" / self.split / scenario_id / view / f"{scenario_id}_caption.json"
            if not jf.exists():
                continue
            data = json.loads(jf.read_text())
            video_file = data.get(view) or f"{scenario_id}_{view}.mp4"
            segs = []
            for ep in data.get("event_phase", []):
                num, name = normalize_phase(ep["labels"][0])
                segs.append(CaptionSegment(
                    phase_num=num,
                    phase_name=name,
                    pedestrian=ep.get("caption_pedestrian", ""),
                    vehicle=ep.get("caption_vehicle", ""),
                    start_time=ep.get("start_time", ""),
                    end_time=ep.get("end_time", ""),
                ))
            out.append(CaptionView(scenario_id=scenario_id, view=view, video=video_file, segments=segs))
        return out

    def load_vqa(self, scenario_id: str) -> list[VQAQuestion]:
        out = []
        for view in VIEWS_VQA:
            jf = self.ann_root / "vqa" / self.split / scenario_id / view / f"{scenario_id}.json"
            if not jf.exists():
                continue
            data = json.loads(jf.read_text())
            for item in data:
                qid = item.get("id", -1)
                overhead = item.get("overhead_videos", [])
                video_file = item.get(view) if view in ("vehicle_view", "overhead_view") else None

                if view == "environment":
                    questions = item.get("environment", [])
                    for i, q in enumerate(questions):
                        out.append(VQAQuestion(
                            scenario_id=scenario_id, view=view, file_id=qid,
                            question_idx=i,
                            question=q["question"],
                            options={k: q[k] for k in ("a", "b", "c", "d") if k in q},
                            correct=q.get("correct"),
                            overhead_videos=overhead,
                        ))
                else:
                    for phase_idx, phase in enumerate(item.get("event_phase", [])):
                        pnum, pname = normalize_phase(phase["labels"][0])
                        for i, q in enumerate(phase.get("conversations", [])):
                            out.append(VQAQuestion(
                                scenario_id=scenario_id, view=view, file_id=qid,
                                question_idx=phase_idx * 100 + i,
                                question=q["question"],
                                options={k: q[k] for k in ("a", "b", "c", "d") if k in q},
                                correct=q.get("correct"),
                                phase_num=pnum, phase_name=pname,
                                start_time=phase.get("start_time"),
                                end_time=phase.get("end_time"),
                                overhead_videos=overhead,
                                video_file=video_file,
                            ))
        return out

    def get_video_paths(self, scenario_id: str) -> dict[str, list[Path]]:
        scen_dir = self.vid_root / scenario_id
        if not scen_dir.exists():
            return {}
        return {
            view_dir.name: sorted(view_dir.glob("*.mp4"))
            for view_dir in scen_dir.iterdir() if view_dir.is_dir()
        }

    def iter_all(self) -> Iterator[tuple[str, list[CaptionView], list[VQAQuestion], dict[str, list[Path]]]]:
        for sid in self.scenarios():
            yield sid, self.load_captions(sid), self.load_vqa(sid), self.get_video_paths(sid)
