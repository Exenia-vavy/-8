import json
import logging
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────
def load_json_safe(path: Path) -> dict | None:
    if not path.exists():
        logging.warning(f"📁 Fichier manquant (ignoré) : {path.name}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if data else None
    except Exception as e:
        logging.error(f"❌ Erreur lecture {path.name} : {e}")
        return None

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()

def iou(bbox1: list, bbox2: list) -> float:
    try:
        x1_1, y1_1, x2_1, y2_1 = map(float, bbox1)
        x1_2, y1_2, x2_2, y2_2 = map(float, bbox2)
    except (TypeError, ValueError):
        return 0.0
        
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
        
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
    area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

# ─────────────────────────────────────────────────────────────
# TRAITEMENT
# ─────────────────────────────────────────────────────────────
def deduplicate_text(ocr_data: dict | None, window_sec: float = 3.0, sim_thresh: float = 0.85) -> list:
    if not ocr_data or not ocr_data.get("frame_results"):
        return []
    
    cleaned = []
    last_text, last_ts = None, -10.0
    
    for frame in ocr_data["frame_results"]:
        ts = frame.get("timestamp_seconds", 0)
        for txt_obj in frame.get("texts", []):
            text = txt_obj.get("text", "").strip()
            conf = txt_obj.get("confidence", 0.5)
            if not text: continue
            
            if (last_text is None) or (ts - last_ts > window_sec) or (similarity(text, last_text) < sim_thresh):
                cleaned.append({
                    "timestamp": round(ts, 2),
                    "type": "text",
                    "content": text,
                    "source": "easyocr",
                    "confidence": conf
                })
                last_text, last_ts = text, ts
    return cleaned

def merge_objects(det_lists: list, window_sec: float = 2.0, iou_thresh: float = 0.3) -> list:
    all_raw = []
    for data in det_lists:
        if data and data.get("frame_detections"):
            src = data.get("model_used", "unknown")
            for frame in data["frame_detections"]:
                ts = frame.get("timestamp_seconds", 0)
                for obj in frame.get("objects", []):
                    all_raw.append({
                        "ts": ts,
                        "class": obj.get("class_name", "unknown"),
                        "conf": float(obj.get("confidence", 0.5)),
                        "bbox": obj.get("bbox", [0, 0, 0, 0]),
                        "source": src
                    })

    all_raw.sort(key=lambda x: (x["class"], x["ts"]))
    events = []

    for item in all_raw:
        merged = False
        for ev in events:
            # ✅ CORRECTION : clés cohérentes ("class" dans ev et item)
            if (ev["class"] == item["class"] and 
                item["ts"] - ev["end_ts"] <= window_sec and
                iou(ev["bbox"], item["bbox"]) >= iou_thresh):
                ev["end_ts"] = item["ts"]
                ev["conf"] = round((ev["conf"] + item["conf"]) / 2, 3)
                ev["sources"].append(item["source"])
                ev["detection_count"] += 1
                merged = True
                break
        if not merged:
            events.append({
                "start_ts": item["ts"],
                "end_ts": item["ts"],
                "class": item["class"],
                "conf": item["conf"],
                "bbox": item["bbox"],
                "sources": [item["source"]],
                "detection_count": 1
            })

    # Formatage final pour le rapport
    return [{
        "start_timestamp": ev["start_ts"],
        "end_timestamp": ev["end_ts"],
        "type": "object",
        "content": ev["class"],
        "source_models": list(set(ev["sources"])),
        "avg_confidence": ev["conf"],
        "representative_bbox": ev["bbox"],
        "total_detections": ev["detection_count"]
    } for ev in events]

def build_final_report(audio_data: dict | None, text_events: list, object_events: list) -> list:
    timeline = text_events + object_events
    if audio_data and audio_data.get("segments"):
        for seg in audio_data["segments"]:
            timeline.append({
                "timestamp": round(seg["start"], 2),
                "type": "audio",
                "content": seg["text"].strip(),
                "source": f"whisper_{audio_data.get('model_used', 'base')}",
                "confidence": 0.95,
                "end_timestamp": round(seg["end"], 2)
            })
    timeline.sort(key=lambda x: x.get("timestamp", x.get("start_timestamp", 0)))
    return timeline

# ─────────────────────────────────────────────────────────────
# EXPORT & MAIN
# ─────────────────────────────────────────────────────────────
def export_report(timeline: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report_final.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 JSON généré : {json_path}")
    
    df = pd.DataFrame(timeline)
    excel_path = output_dir / "report_final.xlsx"
    df.to_excel(excel_path, index=False, engine="openpyxl")
    logging.info(f"📊 Excel généré : {excel_path}")

if __name__ == "__main__":
    logging.info("🚀 Lancement du pipeline ПЗ8 (post-traitement & rapport)...")
    
    BASE = Path(r"C:\Users\Exenia\Desktop\videoss")
    
    ocr_json = load_json_safe(BASE / "ocr_scan" / "ocr_results.json")
    audio_json = load_json_safe(BASE / "audio_pz4" / "audio_transcription.json")
    yolo_json = load_json_safe(BASE / "yolo_pz5" / "yolo_detections.json")
    resnet_json = load_json_safe(BASE / "resnet_pz6" / "resnet_fasterRCNN_detections.json")
    llm_json = load_json_safe(BASE / "llm_pz7_ru" / "llm_detections_ru.json")
    
    text_events = deduplicate_text(ocr_json, window_sec=3.0)
    object_events = merge_objects([yolo_json, resnet_json, llm_json], window_sec=2.0, iou_thresh=0.3)
    timeline = build_final_report(audio_json, text_events, object_events)
    
    export_report(timeline, BASE / "final_report")
    logging.info("✅ ПЗ8 TERMINÉ. Rapport prêt dans final_report/")
