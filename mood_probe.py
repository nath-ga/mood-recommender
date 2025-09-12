# mood_probe.py 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mood_probe.py
Zweck: Aus Medien-Beschreibungen automatisch stimmungsähnliche Labels (Moods) erzeugen.
- Zero-Shot-Klassifikation (multilingual, DE/EN)
- CSV-Ein-/Ausgabe mit robustem Encoding-Handling
- Batch-Verarbeitung mit Fortschrittsbalken
- Sinnvolle Fehlermeldungen für Anfängerfreundlichkeit
- NEU:
  * --min_score (Labels unterhalb der Schwelle werden verworfen; Fallback "neutral")
  * Automatisches deutsches Hypothesis-Template für deutsche Texte
  * Kleines Dateicache für schnellere Wiederholungen

Beispiele:
  # Einzel-Text
  python mood_probe.py --text "Eine warme, leichte Komödie über Freundschaft und Neuanfang." --lang de --prefer_lang de --topk 3 --min_score 0.35

  # CSV -> CSV
  python mood_probe.py --in data/sample.csv --out outputs/sample_moods.csv --lang both --prefer_lang de --topk 3 --min_score 0.35 --auto_lang
"""

import argparse
import sys
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # stabile Erkennung

# ---- NEU: Cache-Helfer ----
import hashlib, json, os
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_key(text: str, cand: List[str], model: str, hyp: str, topk: int, prefer: str) -> str:
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    h.update("\n".join(cand).encode("utf-8"))
    h.update(model.encode("utf-8"))
    h.update(hyp.encode("utf-8"))
    h.update(f"{topk}|{prefer}".encode("utf-8"))
    return h.hexdigest()

def cache_get(key: str):
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)  # {"labels":[...], "scores":[...]}
        except Exception:
            return None
    return None

def cache_put(key: str, labels: List[str], scores: List[float]):
    path = os.path.join(CACHE_DIR, key + ".json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"labels": labels, "scores": scores}, f, ensure_ascii=False)
    except Exception:
        pass

# --- Default-Moods (anpassbar) ---
DEFAULT_MOODS_DE = [
    "heiter", "aufmunternd", "leicht", "romantisch",
    "abenteuerlich", "spannend", "düster", "rau",
    "nachdenklich", "melancholisch", "gemütlich", "beruhigend"
]
DEFAULT_MOODS_EN = [
    "feel-good", "uplifting", "light-hearted", "romantic",
    "adventurous", "suspenseful", "dark", "gritty",
    "thought-provoking", "melancholic", "cozy", "calm"
]
# DE<->EN Synonymliste (erweiterbar)
DE_EN_MAP = {
    "heiter": "feel-good",
    "aufmunternd": "uplifting",
    "leicht": "light-hearted",
    "romantisch": "romantic",
    "abenteuerlich": "adventurous",
    "spannend": "suspenseful",
    "düster": "dark",
    "rau": "gritty",
    "nachdenklich": "thought-provoking",
    "melancholisch": "melancholic",
    "gemütlich": "cozy",
    "beruhigend": "calm"
}
EN_DE_MAP = {v: k for k, v in DE_EN_MAP.items()}

DE_HYP = "Dieser Text vermittelt eine {} Stimmung."
# (Englisch bleibt überschreibbar per --hypothesis)
# EN_HYP = "This text evokes a {} mood."

def unify_labels(labels, scores, prefer_lang="de"):
    """
    Führt DE/EN-Duplikate zusammen und gibt einheitliche Labels zurück.
    prefer_lang: 'de' oder 'en'
    Regel: gleiche Konzepte werden gemerged, Score = max der Paar-Scores.
    """
    concept_scores = {}
    for lab, sc in zip(labels, scores):
        lab_l = lab.strip().lower()
        if lab_l in EN_DE_MAP:   # englisches Label
            key = lab_l
        elif lab_l in DE_EN_MAP: # deutsches Label
            key = DE_EN_MAP[lab_l]
        else:
            key = lab_l  # unbekannt -> eigener Key
        concept_scores[key] = max(concept_scores.get(key, 0.0), float(sc))

    # zurück in Zielsprache mappen
    out_labels, out_scores = [], []
    for key, sc in sorted(concept_scores.items(), key=lambda x: x[1], reverse=True):
        if prefer_lang == "de" and key in EN_DE_MAP:
            out_labels.append(EN_DE_MAP[key])
        else:
            out_labels.append(key)  # EN oder unbekannt
        out_scores.append(sc)
    return out_labels, out_scores

def detect_lang(text):
    try:
        code = detect(text)
        return "de" if code.startswith("de") else "en"
    except Exception:
        return "en"

def parse_args():
    p = argparse.ArgumentParser(description="Zero-Shot Mood Probe (DE/EN)")
    p.add_argument("--in", dest="inp", type=str, help="Pfad zu CSV mit Spalten: title,description")
    p.add_argument("--out", dest="out", type=str, default="moods_out.csv", help="Pfad zur Ausgabe-CSV")
    p.add_argument("--lang", choices=["de", "en", "both"], default="both", help="Welche Mood-Liste nutzen")
    p.add_argument("--text", type=str, help="Einzelner Beschreibungstext zum Schnelltest")
    p.add_argument("--topk", type=int, default=3, help="Anzahl Top-Moods")
    p.add_argument("--hypothesis", type=str,
                   default="This text evokes a {} mood.",
                   help="Hypothesis-Template (Zero-Shot), DE-Beispiel: 'Dieser Text vermittelt eine {} Stimmung.'")
    p.add_argument("--model", type=str,
                   default="joeddav/xlm-roberta-large-xnli",
                   help="Multilinguales NLI-Modell (DE+EN)")
    p.add_argument("--batch_size", type=int, default=8, help="Batch-Größe für Pipeline (Speicher beachten)")
    p.add_argument("--prefer_lang", choices=["de","en"], default="de",
               help="In welcher Sprache die endgültigen Mood-Labels ausgegeben werden sollen")
    p.add_argument("--auto_lang", action="store_true",
               help="Spracherkennung pro Zeile: wählt automatisch DE/EN-Kandidaten")
    # NEU:
    p.add_argument("--min_score", type=float, default=0.0,
               help="Mindestscore pro Label; Labels darunter werden verworfen (0.0 = deaktiviert)")
    return p.parse_args()

def get_candidate_labels(lang: str) -> List[str]:
    if lang == "de":
        return DEFAULT_MOODS_DE
    if lang == "en":
        return DEFAULT_MOODS_EN
    return DEFAULT_MOODS_DE + DEFAULT_MOODS_EN

def load_csv_safely(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except Exception:
            continue
    raise RuntimeError(
        f"CSV konnte nicht gelesen werden. Bitte prüfe Datei und Encoding (erwartet Spalten: title, description): {path}"
    )

def ensure_required_columns(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    need = ["title", "description"]
    if not all(col in cols for col in need):
        raise RuntimeError("CSV braucht Spalten: title, description (Groß-/Kleinschreibung egal).")
    df.rename(columns={cols["title"]: "title", cols["description"]: "description"}, inplace=True)

def build_pipeline(model_name: str):
    try:
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError(
            "Fehler beim Import von 'transformers'. Bitte vorher installieren:\n"
            "  pip install -r requirements.txt\n"
            f"Originalfehler: {e}"
        )
    return pipeline("zero-shot-classification", model=model_name, device=-1)  # CPU

def classify_batch(zero_shot, texts: List[str], candidate_labels: List[str],
                   hypothesis_template: str, topk: int) -> List[Tuple[List[str], List[float]]]:
    results = zero_shot(
        sequences=texts,
        candidate_labels=candidate_labels,
        hypothesis_template=hypothesis_template,
        multi_label=True
    )
    out = []
    if isinstance(results, dict):
        results = [results]
    for res in results:
        pairs = sorted(zip(res["labels"], res["scores"]), key=lambda x: x[1], reverse=True)[:topk]
        out.append(( [p[0] for p in pairs], [float(p[1]) for p in pairs] ))
    return out

def _filter_min_score(labels, scores, min_score: float):
    if min_score <= 0:
        return labels, scores
    kept = [(l, s) for l, s in zip(labels, scores) if s >= min_score]
    if kept:
        labs, scs = zip(*kept)
        return list(labs), list(scs)
    # Fallback
    return ["neutral"], [min_score]

def single_text_mode(args):
    zero_shot = build_pipeline(args.model)
    text = args.text.strip()

    # Kandidatenlabels + Hypothesis wählen
    if args.auto_lang:
        lang = detect_lang(text)
        candidate_labels = get_candidate_labels(lang)
        hyp = DE_HYP if lang == "de" else args.hypothesis
    else:
        candidate_labels = get_candidate_labels(args.lang)
        hyp = DE_HYP if args.lang == "de" else args.hypothesis

    # Cache prüfen
    key = _cache_key(text, candidate_labels, args.model, hyp, args.topk, args.prefer_lang)
    cached = cache_get(key)
    if cached:
        labels, scores = cached["labels"], cached["scores"]
    else:
        labels, scores = classify_batch(zero_shot, [text], candidate_labels, hyp, args.topk)[0]
        labels, scores = unify_labels(labels, scores, prefer_lang=args.prefer_lang)
        labels, scores = _filter_min_score(labels, scores, args.min_score)
        labels, scores = labels[:args.topk], scores[:args.topk]
        cache_put(key, labels, scores)

    # Ausgabe
    print("Input:", args.text)
    for lab, sc in zip(labels, scores):
        print(f"{lab}: {sc:.3f}")

def csv_mode(args):
    df = load_csv_safely(args.inp)
    ensure_required_columns(df)
    df["description"] = df["description"].fillna("").astype(str)

    zero_shot = build_pipeline(args.model)
    descs = df["description"].tolist()
    moods_col, scores_col = [], []

    for start in tqdm(range(0, len(descs), args.batch_size)):
        chunk = descs[start:start+args.batch_size]

        # pro Text Labelkandidaten + Hypothesis wählen und (wegen Auto-Lang) einzeln klassifizieren
        for txt in chunk:
            if args.auto_lang:
                lang = detect_lang(txt)
                cand = get_candidate_labels(lang)
                hyp = DE_HYP if lang == "de" else args.hypothesis
            else:
                cand = get_candidate_labels(args.lang)
                hyp = DE_HYP if args.lang == "de" else args.hypothesis

            key = _cache_key(txt, cand, args.model, hyp, args.topk, args.prefer_lang)
            cached = cache_get(key)
            if cached:
                labels, scores = cached["labels"], cached["scores"]
            else:
                labels, scores = classify_batch(zero_shot, [txt], cand, hyp, args.topk)[0]
                labels, scores = unify_labels(labels, scores, prefer_lang=args.prefer_lang)
                labels, scores = _filter_min_score(labels, scores, args.min_score)
                labels, scores = labels[:args.topk], scores[:args.topk]
                cache_put(key, labels, scores)

            moods_col.append("; ".join(labels))
            scores_col.append("; ".join([f"{s:.3f}" for s in scores]))

    out_df = df.copy()
    out_df["moods_topk"] = moods_col
    out_df["moods_scores"] = scores_col
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Fertig. Ausgabe gespeichert unter: {args.out}")

def main():
    args = parse_args()
    if not args.text and not args.inp:
        print("Bitte nutze entweder --text oder --in <csv>.", file=sys.stderr)
        sys.exit(1)
    try:
        if args.text:
            single_text_mode(args)
        else:
            csv_mode(args)
    except Exception as e:
        print(f"[FEHLER] {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
