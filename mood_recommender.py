# mood_recommender.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mood_recommender.py
Erzeugt Empfehlungen aus einer gelabelten CSV (Output von mood_probe.py).

Zwei Quellen für den Nutzer-Mood:
  A) Slider-Modus (ohne KI): --valence und --energy
  B) Freitext-Modus (mit KI): --user_text (Zero-Shot gegen dieselben Mood-Labels)

Eingabe-CSV (utf-8) braucht mind. Spalten:
  title, description, moods_topk, moods_scores
(moods_* stammen direkt aus mood_probe.py)

Ausgabe: Top-N Empfehlungen auf STDOUT und optional als CSV.

Beispiele (PowerShell):
  # A) Slider: Valenz -1..+1 (schlecht..gut), Energie 0..1 (müde..aufgedreht)
  python .\mood_recommender.py `
    --csv outputs\moods_de.csv `
    --valence 0.6 `
    --energy 0.3 `
    --topn 7 `
    --out outputs\recs_valence.csv

  # B) Freitext (optional, braucht transformers):
  python .\mood_recommender.py `
    --csv outputs\moods_de.csv `
    --user_text "Ich bin erschöpft und hätte gern etwas Heiteres, Beruhigendes." `
    --topn 7 `
    --out outputs\recs_text.csv
"""

import argparse
import math
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# --- Kanonische Mood-Liste (muss zu mood_probe passen) ---
CANON_DE = [
    "heiter","aufmunternd","leicht","romantisch",
    "abenteuerlich","spannend","düster","rau",
    "nachdenklich","melancholisch","gemütlich","beruhigend"
]

# leichte/helle vs. dunkle Cluster (für kleine Diversitäts-Regel)
LIGHT_CLUSTER = {"heiter","aufmunternd","leicht","romantisch","gemütlich","beruhigend"}
DARK_CLUSTER  = {"düster","rau"}
NEUTRAL_MID   = {"nachdenklich","melancholisch","spannend","abenteuerlich"}

# --------- Hilfsfunktionen ---------
def parse_scores(s: str) -> List[float]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [float(x) for x in s.split(";")]

def parse_labels(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip().lower() for x in s.split(";")]

def item_vector(labels: List[str], scores: List[float], alphabet: List[str]) -> np.ndarray:
    vec = np.zeros(len(alphabet), dtype=float)
    for l, sc in zip(labels, scores):
        if l in alphabet:
            vec[alphabet.index(l)] = max(vec[alphabet.index(l)], sc)
    # optional Normierung auf 1
    if vec.sum() > 0:
        vec = vec / np.linalg.norm(vec)
    return vec

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

# --------- Nutzer-Mood aus Slider ---------
def slider_to_mood_weights(valence: float, energy: float) -> Dict[str, float]:
    """
    valence: -1 (negativ) .. +1 (positiv)
    energy:  0 (müde/ruhig) .. 1 (aufgedreht)
    Heuristik: mappe auf Label-Gewichte.
    """
    # Clamps
    v = max(-1.0, min(1.0, valence))
    e = max(0.0, min(1.0, energy))

    w = {lbl: 0.0 for lbl in CANON_DE}

    # positive Valenz -> leichte/helle Labels
    if v > 0:
        w["heiter"]        += 0.4 * v
        w["aufmunternd"]   += 0.4 * v
        w["leicht"]        += 0.3 * v
        w["romantisch"]    += 0.2 * v
        w["gemütlich"]     += 0.2 * (1-e)   # je ruhiger, desto gemütlicher
        w["beruhigend"]    += 0.2 * (1-e)
        w["abenteuerlich"] += 0.2 * e       # je energiegeladener, desto abenteuerlich
        w["spannend"]      += 0.15 * e
    # negative Valenz -> nachdenklich/melancholisch/dunkler, aber mit „Aufheller“
    if v <= 0:
        w["nachdenklich"]  += 0.35 * (-v)
        w["melancholisch"] += 0.3  * (-v)
        w["beruhigend"]    += 0.25 * (1-e)  # bei wenig Energie beruhigend
        w["gemütlich"]     += 0.2  * (1-e)
        # bei hoher Energie: eher düster/rau, aber behutsam
        w["düster"]        += 0.2  * e * (-v)
        w["rau"]           += 0.15 * e * (-v)
        # sanfter Gegenpol, um nicht nur „runterzuziehen“
        w["heiter"]        += 0.1  * (1 - (-v))  # kleine Chance auf leichte Empfehlung

    # kleine Normalisierung
    s = sum(w.values())
    if s > 0:
        for k in w: w[k] /= s
    return w

def weights_to_vector(weights: Dict[str, float], alphabet: List[str]) -> np.ndarray:
    vec = np.zeros(len(alphabet), dtype=float)
    for i, lbl in enumerate(alphabet):
        vec[i] = weights.get(lbl, 0.0)
    if vec.sum() > 0:
        vec = vec / np.linalg.norm(vec)
    return vec

# --------- Nutzer-Mood aus Freitext (optional) ---------
def zero_shot_user_vector(user_text: str, alphabet: List[str]) -> np.ndarray:
    try:
        from transformers import pipeline
    except Exception as e:
        raise RuntimeError("Für --user_text benötigst du 'transformers'. Installiere es mit:\n  pip install transformers torch")

    hyp_de = "Dieser Text vermittelt eine {} Stimmung."
    z = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=-1)
    res = z(user_text, candidate_labels=alphabet, hypothesis_template=hyp_de, multi_label=True)
    pairs = sorted(zip(res["labels"], res["scores"]), key=lambda x: x[1], reverse=True)

    weights = {lbl: sc for lbl, sc in pairs}
    # min. Soft-Norm
    total = sum(weights.values())
    if total > 0:
        for k in list(weights.keys()):
            weights[k] /= total
    return weights_to_vector(weights, alphabet)

# --------- Diversitäts-Postprozessor ---------
def diversify(cands: List[dict], topn: int = 7) -> List[dict]:
    """Behalte Relevanz, limitiere aber sehr dunkle Häufungen und erlaube 1 'helle' Serendipity."""
    picked = []
    dark_dominated = 0
    light_present = False

    for c in cands:
        if len(picked) >= topn: break
        top3 = c.get("top_labels", [])
        is_dark = sum(1 for l in top3 if l in DARK_CLUSTER) >= 2
        is_light = any(l in LIGHT_CLUSTER for l in top3)
        # max 3 stark dunkle
        if is_dark and dark_dominated >= 3:
            continue
        picked.append(c)
        if is_dark: dark_dominated += 1
        if is_light: light_present = True

    if not light_present and len(picked) < topn:
        for c in cands:
            top3 = c.get("top_labels", [])
            if any(l in LIGHT_CLUSTER for l in top3) and c not in picked:
                picked.append(c); break

    return picked[:topn]

# --------- Hauptlogik ---------
def build_item_matrix(df: pd.DataFrame, alphabet: List[str]) -> Tuple[np.ndarray, List[List[str]]]:
    vecs = []
    top_labels = []
    for _, row in df.iterrows():
        labs = parse_labels(row.get("moods_topk", ""))
        scs  = parse_scores(row.get("moods_scores", ""))
        vecs.append(item_vector(labs, scs, alphabet))
        top_labels.append(labs[:3])
    return np.vstack(vecs), top_labels

def recommend(csv_path: str, user_vec: np.ndarray, topn: int = 7, out_csv: str = None):
    df = pd.read_csv(csv_path, encoding="utf-8")
    cols = {c.lower(): c for c in df.columns}
    required = ["title","description","moods_topk","moods_scores"]
    for must in required:
        if must not in cols:
            raise RuntimeError("CSV braucht Spalten: title, description, moods_topk, moods_scores (aus mood_probe.py).")

    # tolerant umbenennen
    rename_map = {
        cols["title"]: "title",
        cols["description"]: "description",
        cols["moods_topk"]: "moods_topk",
        cols["moods_scores"]: "moods_scores"
    }
    # optional: type (film / audiobook)
    if "type" in cols:
        rename_map[cols["type"]] = "type"

    df = df.rename(columns=rename_map)

    # falls 'type' fehlt, setze 'unknown'
    if "type" not in df.columns:
        df["type"] = "unknown"


    M, top_labels = build_item_matrix(df, CANON_DE)
    scores = [cosine(user_vec, M[i]) for i in range(M.shape[0])]

    cands = []
    for i, sc in enumerate(scores):
        cands.append({
            "idx": i,
            "title": df.loc[i, "title"],
            "type": df.loc[i, "type"],   # NEU
            "score": round(float(sc), 4),
            "top_labels": top_labels[i]
        })

    cands.sort(key=lambda x: x["score"], reverse=True)
    picked = diversify(cands, topn=topn)

    # Ausgabe hübsch
    print("\nEmpfehlungen:")
    for rank, c in enumerate(picked, 1):
        tl = ", ".join(c["top_labels"])
        print(f"{rank:>2}. {c['title']} ({c['type']}) | Score {c['score']:.3f} | Moods: {tl}")

    # Optional speichern
    if out_csv:
        out_rows = []
        for c in picked:
            out_rows.append({
                "rank": len(out_rows)+1,
                "title": c["title"],
                "type": c["type"],  # NEU
                "score": c["score"],
                "moods": ", ".join(c["top_labels"]),
                "why": explain_reason(user_vec, M[c["idx"]], c["top_labels"])
            })
        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\nGespeichert unter: {out_csv}")

def explain_reason(user_vec: np.ndarray, item_vec: np.ndarray, item_top_labels: List[str]) -> str:
    # einfache Erklärung: Schnittmenge bevorzugter Labels ~ Projektion
    # (hier minimalistisch: gib die Item-Toplabels zurück)
    return "Passend wegen: " + ", ".join(item_top_labels[:3])

def main():
    ap = argparse.ArgumentParser(description="Mood-basierter Recommender")
    ap.add_argument("--csv", required=True, help="Pfad zur gelabelten CSV (Output von mood_probe.py)")
    ap.add_argument("--topn", type=int, default=7, help="Anzahl Empfehlungen")
    ap.add_argument("--out", type=str, default=None, help="Optional: CSV mit Empfehlungen speichern")

    # Modus A: Slider
    ap.add_argument("--valence", type=float, help="Valenz -1 (negativ) .. +1 (positiv)")
    ap.add_argument("--energy",  type=float, help="Energie 0 (ruhig) .. 1 (aufgedreht)")

    # Modus B: Freitext
    ap.add_argument("--user_text", type=str, help="Freitext zur aktuellen Stimmung (optional, benötigt transformers)")

    args = ap.parse_args()

    if args.user_text:
        user_vec = zero_shot_user_vector(args.user_text.strip(), CANON_DE)
    else:
        # Fallback Slider, mit sinnvollen Defaults
        v = args.valence if args.valence is not None else 0.3
        e = args.energy  if args.energy  is not None else 0.3
        weights = slider_to_mood_weights(v, e)
        user_vec = weights_to_vector(weights, CANON_DE)

    recommend(args.csv, user_vec, topn=args.topn, out_csv=args.out)

if __name__ == "__main__":
    main()

