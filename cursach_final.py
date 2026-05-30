import os
import cv2
import json
import torch
import whisper
import easyocr
import subprocess
from Levenshtein import ratio

# =====================================================================
# 1. CONFIGURATION DES CHEMINS (Modifiez uniquement cette partie)
# =====================================================================
VIDEO_SOURCE = r"C:\Users\Exenia\Desktop\videoss\movie2.mp4"
DOSSIER_SORTIE = r"C:\Users\Exenia\Documents"

# Fichiers temporaires et finaux
WAV_TEMP = os.path.join(DOSSIER_SORTIE, "son_temporaire.wav")
JSON_OCR = os.path.join(DOSSIER_SORTIE, "final_ocr_propre.json")
JSON_AUDIO = os.path.join(DOSSIER_SORTIE, "final_audio_propre.json")
JSON_OBJETS = os.path.join(DOSSIER_SORTIE, "final_objets_propre.json")

# =====================================================================
# 2. FONCTIONS DE TRAITEMENT (PZ 3, 4, 6)
# =====================================================================

def executer_pz3_ocr(video_path):
    print("\n[Étape 1/5] Lancement de l'OCR (PZ3)...")
    reader = easyocr.Reader(['ru', 'en'])
    cap = cv2.VideoCapture(video_path)
    
    donnees_brutes = []
    # Scan toutes les 5 secondes sur les 100 premieres secondes
    for sec in range(0, 100, 5):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = cap.read()
        if not success: break
        
        resultats = reader.readtext(frame)
        if resultats:
            phrases = [res[1] for res in resultats]
            texte_joint = ' | '.join(phrases)
            donnees_brutes.append({"seconde": sec, "texte": texte_joint})
            
    cap.release()
    return donnees_brutes

def executer_pz4_whisper(video_path, wav_path):
    print("\n[Étape 2/5] Lancement de Whisper (PZ4)...")
    print("  -> Extraction audio via FFmpeg...")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-ar', '16000', '-ac', '1', wav_path], capture_output=True)
    
    print("  -> Transcription audio en cours...")
    model = whisper.load_model("base")
    result = model.transcribe(wav_path)
    
    donnees_audio = []
    for segment in result['segments']:
        donnees_audio.append({
            "debut": int(segment['start']),
            "fin": int(segment['end']),
            "texte": segment['text'].strip()
        })
    
    # Nettoyage du fichier audio temporaire
    if os.path.exists(wav_path):
        os.remove(wav_path)
        
    return donnees_audio

def executer_pz6_resnet(video_path):
    print("\n[Étape 3/5] Lancement de ResNet (PZ6)...")
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    COCO_CLASSES = weights.meta["categories"]
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    detections_chronologiques = []
    frame_count = 0
    SKIP_FRAMES = 60  # On accelere pour le script global
    THRESHOLD = 0.5
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % SKIP_FRAMES != 0: continue
        
        temps_secondes = round(frame_count / fps, 2)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(device).unsqueeze(0)
        
        with torch.no_grad():
            preds = model(img_tensor)[0]
            
        mask = preds["scores"] > THRESHOLD
        labels = preds["labels"][mask].cpu().numpy()
        
        for label in labels:
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"obj_{int(label)}"
            detections_chronologiques.append({
                "seconde": temps_secondes,
                "objet": class_name
            })
            
    cap.release()
    return detections_chronologiques

# =====================================================================
# 3. FONCTIONS DE NETTOYAGE (PZ 8)
# =====================================================================

def appliquer_deduplication_texte(donnees_brutes, chemin_sortie):
    if not donnees_brutes: return
    donnees_nettoyees = []
    premier = donnees_brutes[0]
    
    groupe_actuel = {
        "debut": premier.get("seconde", premier.get("debut", 0)),
        "fin": premier.get("seconde", premier.get("fin", 0)),
        "texte": premier["texte"]
    }
    
    for element in donnees_brutes[1:]:
        texte_suivant = element["texte"]
        next_fin = element.get("seconde", element.get("fin", 0))
        
        if ratio(groupe_actuel["texte"].lower(), texte_suivant.lower()) > 0.70:
            groupe_actuel["fin"] = next_fin
        else:
            donnees_nettoyees.append(groupe_actuel)
            groupe_actuel = {
                "debut": element.get("seconde", element.get("debut", 0)),
                "fin": next_fin,
                "texte": texte_suivant
            }
    donnees_nettoyees.append(groupe_actuel)
    
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        json.dump(donnees_nettoyees, f, ensure_ascii=False, indent=4)

def appliquer_fusion_objets(detections, chemin_sortie):
    if not detections: return
    detections.sort(key=lambda x: (x["objet"], x["seconde"]))
    evenements_fuses = []
    objets_actifs = {}
    TOLERANCE_TEMPS = 2.0
    
    for det in detections:
        nom_objet = det["objet"]
        temps_actuel = det["seconde"]
        
        if nom_objet in objets_actifs:
            dernier_evenement = objets_actifs[nom_objet]
            if temps_actuel - dernier_evenement["fin"] <= TOLERANCE_TEMPS:
                dernier_evenement["fin"] = temps_actuel
            else:
                evenements_fuses.append(dernier_evenement)
                objets_actifs[nom_objet] = {"objet": nom_objet, "debut": temps_actuel, "fin": temps_actuel}
        else:
            objets_actifs[nom_objet] = {"objet": nom_objet, "debut": temps_actuel, "fin": temps_actuel}
            
    for dernier_evenement in objets_actifs.values():
        evenements_fuses.append(dernier_evenement)
        
    evenements_fuses.sort(key=lambda x: x["debut"])
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        json.dump(evenements_fuses, f, ensure_ascii=False, indent=4)

# =====================================================================
# 4. POINT D'ENTRÉE PRINCIPAL (LANCEMENT DU CYCLE ENTIER)
# =====================================================================
if __name__ == "__main__":
    print("==================================================")
    print("DEBUT DU PIPELINE GLOBAL DE TRAITEMENT MULTIMEDIA")
    print("==================================================")
    
    if not os.path.exists(VIDEO_SOURCE):
        print(f"[ERREUR] La video source est introuvable : {VIDEO_SOURCE}")
        exit()

    # Exécution de l'analyse brute
    ocr_brut = executer_pz3_ocr(VIDEO_SOURCE)
    audio_brut = executer_pz4_whisper(VIDEO_SOURCE, WAV_TEMP)
    objets_brut = executer_pz6_resnet(VIDEO_SOURCE)
    
    print("\n[Étape 4/5] Nettoyage et Deduplication des textes...")
    appliquer_deduplication_texte(ocr_brut, JSON_OCR)
    appliquer_deduplication_texte(audio_brut, JSON_AUDIO)
    
    print("\n[Étape 5/5] Fusion chronologique des objets...")
    appliquer_fusion_objets(objets_brut, JSON_OBJETS)
    
    print("\n==================================================")
    print("   [SUCCÈS] TOUTES LES ÉTAPES ONT ÉTÉ ENCHAÎNÉES  ")
    print(f" Vos fichiers finaux propres sont dans : {DOSSIER_SORTIE}")
    print("==================================================")
