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

Beispiele:
  # Einzel-Text
  python mood_probe.py --text "Eine warme, leichte Komödie über Freundschaft und Neuanfang."

  # CSV -> CSV
  python mood_probe.py --in data/sample.csv --out outputs/sample_moods.csv --lang both --topk 3
"""

import argparse
import sys
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

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

def parse_args():
    p = argparse.ArgumentParser(description="Zero-Shot Mood Probe (DE/EN)")
    p.add_argument("--in", dest="inp", type=str, help="Pfad zu CSV mit Spalten: title,description")
    p.add_argument("--out", dest="out", type=str, default="moods_out.csv", help="Pfad zur Ausgabe-CSV")
    p.add_argument("--lang", choices=["de", "en", "both"], default="both", help="Welche Mood-Liste nutzen")
    p.add_argument("--text", type=str, help="Einzelner Beschreibungstext zum Schnelltest")
    p.add_argument("--topk", type=int, default=3, help="Anzahl Top-Moods")
    p.add_argument("--hypothesis", type=str,
                   default="This text evokes a {} mood.",
                   help="Hypothesis-Template (Zero-Shot), DE-Beispiel: 'Dieser Text erzeugt eine {} Stimmung.'")
    p.add_argument("--model", type=str,
                   default="joeddav/xlm-roberta-large-xnli",
                   help="Multilinguales NLI-Modell (DE+EN)")
    p.add_argument("--batch_size", type=int, default=8, help="Batch-Größe für Pipeline (Speicher beachten)")
    return p.parse_args()

def get_candidate_labels(lang: str) -> List[str]:
    if lang == "de":
        return DEFAULT_MOODS_DE
    if lang == "en":
        return DEFAULT_MOODS_EN
    return DEFAULT_MOODS_DE + DEFAULT_MOODS_EN

def load_csv_safely(path: str) -> pd.DataFrame:
    # Versucht UTF-8, dann latin-1 – gibt klare Fehlermeldung aus
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
    # Tolerant gegenüber Groß-/Kleinschreibung
    need = ["title", "description"]
    if not all(col in cols for col in need):
        raise RuntimeError("CSV braucht Spalten: title, description (Groß-/Kleinschreibung egal).")
    # Normalisiere Spaltennamen
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
    # device=-1 => CPU (anfängerfreundlich/stabil)
    return pipeline("zero-shot-classification", model=model_name, device=-1)

def classify_batch(zero_shot, texts: List[str], candidate_labels: List[str],
                   hypothesis_template: str, topk: int) -> List[Tuple[List[str], List[float]]]:
    """
    Gibt für jeden Text ein Tupel (labels, scores) zurück, jeweils Top-k.
    """
    results = zero_shot(
        sequences=texts,
        candidate_labels=candidate_labels,
        hypothesis_template=hypothesis_template,
        multi_label=True
    )
    out = []
    # Wenn nur ein Text -> dict, sonst Liste von dicts
    if isinstance(results, dict):
        results = [results]
    for res in results:
        pairs = sorted(zip(res["labels"], res["scores"]), key=lambda x: x[1], reverse=True)[:topk]
        out.append(( [p[0] for p in pairs], [float(p[1]) for p in pairs] ))
    return out

def single_text_mode(args):
    zero_shot = build_pipeline(args.model)
    candidate_labels = get_candidate_labels(args.lang)
    texts = [args.text.strip()]
    res = classify_batch(zero_shot, texts, candidate_labels, args.hypothesis, args.topk)[0]
    labels, scores = res
    print("Input:", args.text)
    for lab, sc in zip(labels, scores):
        print(f"{lab}: {sc:.3f}")

def csv_mode(args):
    df = load_csv_safely(args.inp)
    ensure_required_columns(df)
    df["description"] = df["description"].fillna("").astype(str)

    zero_shot = build_pipeline(args.model)
    candidate_labels = get_candidate_labels(args.lang)

    batch_size = max(1, args.batch_size)
    moods_col, scores_col = [], []

    print(f"Verarbeite {len(df)} Zeilen …")
    for i in tqdm(range(0, len(df), batch_size)):
        batch_desc = df["description"].iloc[i:i+batch_size].tolist()
        batch_res = classify_batch(zero_shot, batch_desc, candidate_labels, args.hypothesis, args.topk)
        for labels, scores in batch_res:
            moods_col.append("; ".join(labels))
            scores_col.append("; ".join([f"{s:.3f}" for s in scores]))

    out_df = df.copy()
    out_df["moods_topk"] = moods_col
    out_df["moods_scores"] = scores_col

    # Schreibe immer UTF-8, damit Folgeprozesse leicht haben
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
