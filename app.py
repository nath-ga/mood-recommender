# app.py
# Film oder Hörbuch vorschlagen passend zur Gefühlslage des Nutzers

import os
import io
import numpy as np
import pandas as pd
import streamlit as st

# Kanonische Labels (wie in mood_recommender.py)
CANON_DE = [
    "heiter","aufmunternd","leicht","romantisch",
    "abenteuerlich","spannend","düster","rau",
    "nachdenklich","melancholisch","gemütlich","beruhigend"
]
LIGHT_CLUSTER = {"heiter","aufmunternd","leicht","romantisch","gemütlich","beruhigend"}
DARK_CLUSTER  = {"düster","rau"}

def parse_scores(s: str):
    if not isinstance(s, str) or not s.strip():
        return []
    return [float(x) for x in s.split(";")]

def parse_labels(s: str):
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip().lower() for x in s.split(";")]

def item_vector(labels, scores, alphabet):
    vec = np.zeros(len(alphabet), dtype=float)
    for l, sc in zip(labels, scores):
        if l in alphabet:
            idx = alphabet.index(l)
            vec[idx] = max(vec[idx], sc)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def slider_to_mood_weights(valence: float, energy: float):
    v = max(-1.0, min(1.0, valence))
    e = max(0.0, min(1.0, energy))
    w = {lbl: 0.0 for lbl in CANON_DE}
    if v > 0:
        w["heiter"]        += 0.4 * v
        w["aufmunternd"]   += 0.4 * v
        w["leicht"]        += 0.3 * v
        w["romantisch"]    += 0.2 * v
        w["gemütlich"]     += 0.2 * (1-e)
        w["beruhigend"]    += 0.2 * (1-e)
        w["abenteuerlich"] += 0.2 * e
        w["spannend"]      += 0.15 * e
    if v <= 0:
        w["nachdenklich"]  += 0.35 * (-v)
        w["melancholisch"] += 0.3  * (-v)
        w["beruhigend"]    += 0.25 * (1-e)
        w["gemütlich"]     += 0.2  * (1-e)
        w["düster"]        += 0.2  * e * (-v)
        w["rau"]           += 0.15 * e * (-v)
        w["heiter"]        += 0.1  * (1 - (-v))
    s = sum(w.values())
    if s > 0:
        for k in w: w[k] /= s
    return w

def weights_to_vector(weights, alphabet):
    vec = np.zeros(len(alphabet), dtype=float)
    for i, lbl in enumerate(alphabet):
        vec[i] = weights.get(lbl, 0.0)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def uplift_weights(weights):
    w = dict(weights)
    dark_keys = {"düster","rau","melancholisch","nachdenklich","spannend"}
    light_keys = {"heiter","aufmunternd","leicht","romantisch","gemütlich","beruhigend"}
    budget = 0.0
    for k in dark_keys:
        take = 0.5 * w.get(k, 0.0)
        w[k] = max(0.0, w.get(k,0.0) - take)
        budget += take
    if budget > 0 and light_keys:
        share = budget / len(light_keys)
        for k in light_keys:
            w[k] = w.get(k,0.0) + share
    s = sum(w.values())
    if s > 0:
        for k in w: w[k] /= s
    return w

def zero_shot_user_vector(user_text: str, alphabet):
    try:
        from transformers import pipeline
    except Exception:
        st.error("Für den Freitext-Modus wird transformers benötigt. Bitte `pip install transformers torch` ausführen.")
        return np.zeros(len(alphabet), dtype=float)
    hyp_de = "Dieser Text vermittelt eine {} Stimmung."
    z = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=-1)
    res = z(user_text, candidate_labels=alphabet, hypothesis_template=hyp_de, multi_label=True)
    pairs = sorted(zip(res["labels"], res["scores"]), key=lambda x: x[1], reverse=True)
    weights = {lbl: sc for lbl, sc in pairs}
    total = sum(weights.values())
    if total > 0:
        for k in list(weights.keys()):
            weights[k] /= total
    return weights_to_vector(weights, alphabet)

def diversify(sorted_cands, topn=7):
    picked, dark_dominated, light_present = [], 0, False
    for c in sorted_cands:
        if len(picked) >= topn: break
        top3 = c.get("top_labels", [])
        is_dark = sum(1 for l in top3 if l in DARK_CLUSTER) >= 2
        is_light = any(l in LIGHT_CLUSTER for l in top3)
        if is_dark and dark_dominated >= 3:
            continue
        picked.append(c)
        if is_dark: dark_dominated += 1
        if is_light: light_present = True
    if not light_present and len(picked) < topn:
        for c in sorted_cands:
            top3 = c.get("top_labels", [])
            if any(l in LIGHT_CLUSTER for l in top3) and c not in picked:
                picked.append(c); break
    return picked[:topn]

@st.cache_data
def load_items_from_csv(path_or_buffer):
    df = pd.read_csv(path_or_buffer, encoding="utf-8")
    cols = {c.lower(): c for c in df.columns}
    need = ["title","description","moods_topk","moods_scores"]
    for n in need:
        if n not in cols:
            raise ValueError("CSV braucht Spalten: title, description, moods_topk, moods_scores.")
    df = df.rename(columns={
        cols["title"]:"title",
        cols["description"]:"description",
        cols["moods_topk"]:"moods_topk",
        cols["moods_scores"]:"moods_scores",
        **({cols["type"]:"type"} if "type" in cols else {})
    })
    if "type" not in df.columns:
        df["type"] = "unknown"
    # Vektoren vorbereiten
    vecs, top_labels = [], []
    for _, row in df.iterrows():
        labs = parse_labels(row.get("moods_topk",""))
        scs  = parse_scores(row.get("moods_scores",""))
        vecs.append(item_vector(labs, scs, CANON_DE))
        top_labels.append(labs[:3])
    M = np.vstack(vecs) if len(vecs)>0 else np.zeros((0, len(CANON_DE)))
    return df, M, top_labels

def recommend(df, M, top_labels, user_vec, topn=7, use_diversity=True):
    scores = [cosine(user_vec, M[i]) for i in range(M.shape[0])]
    cands = [{
        "idx": i,
        "title": df.loc[i,"title"],
        "type": df.loc[i,"type"],
        "score": round(float(scores[i]), 4),
        "top_labels": top_labels[i]
    } for i in range(len(scores))]
    cands.sort(key=lambda x: x["score"], reverse=True)
    return diversify(cands, topn=topn) if use_diversity else cands[:topn]

# UI
st.set_page_config(page_title="Stimmungs-Empfehlungen", page_icon="🎬", layout="centered")

st.title("Stimmungs-Empfehlungen")
st.write("Wähle deine aktuelle Stimmung und erhalte passende Film- oder Hörbuchvorschläge.")

# Daten
st.subheader("Daten")
default_path = os.path.join("outputs", "moods_de.csv")

use_upload = st.checkbox("Eigene Liste hochladen", value=False)
st.caption("CSV-Datei im UTF-8-Format. Benötigte Spalten: title, description, moods_topk, moods_scores (optional: type).")

if use_upload:
    up = st.file_uploader("Datei auswählen", type=["csv"])
    if up is None:
        st.stop()
    df, M, top_labels = load_items_from_csv(up)
else:
    if not os.path.exists(default_path):
        st.warning("Keine Beispieldatei gefunden. Bitte eigene Liste hochladen.")
        st.stop()
    df, M, top_labels = load_items_from_csv(default_path)

# Mood-Eingabe
st.subheader("Deine Stimmung")
col1, col2 = st.columns(2)
with col1:
    valence = st.slider("Stimmungslage (schlecht ↔ gut)", min_value=-1.0, max_value=1.0, value=0.3, step=0.1)
with col2:
    energy  = st.slider("Energielevel (ruhig ↔ aktiv)",  min_value=0.0,  max_value=1.0, value=0.3, step=0.1)

MODE_LABELS = {"match": "passend", "uplift": "aufhellen", "mix": "Mischung"}
MODE_VALUES = {v: k for k, v in MODE_LABELS.items()}

strategy_label = st.selectbox("Empfehlungsmodus", [MODE_LABELS["match"], MODE_LABELS["uplift"], MODE_LABELS["mix"]], index=0)
strategy = MODE_VALUES[strategy_label]
if strategy == "mix":
    alpha = st.slider("Anteil Aufhellen in der Mischung", 0.0, 1.0, 0.5, 0.05)
else:
    alpha = 0.5  # bleibt ungenutzt

use_text = st.checkbox("Kurz beschreiben (optional)", value=False)
if use_text:
    user_text = st.text_input("Wie war dein Tag?")
else:
    user_text = ""

# Nutzervektor
if use_text and user_text.strip():
    user_vec = zero_shot_user_vector(user_text.strip(), CANON_DE)
else:
    w_match = slider_to_mood_weights(valence, energy)
    if strategy == "match":
        w_final = w_match
    elif strategy == "uplift":
        w_final = uplift_weights(w_match) if valence <= 0 else w_match
    else:
        w_uplift = uplift_weights(w_match) if valence <= 0 else w_match
        w_final = {k: (1-alpha)*w_match.get(k,0.0) + alpha*w_uplift.get(k,0.0) for k in CANON_DE}
        s = sum(w_final.values())
        if s > 0:
            for k in w_final: w_final[k] /= s
    user_vec = weights_to_vector(w_final, CANON_DE)

# Empfehlungen
st.subheader("Vorschläge")
topn = st.slider("Anzahl Vorschläge", 3, 15, 7, 1)
use_diversity = st.checkbox("Für Abwechslung sorgen", value=True)

if st.button("Vorschläge anzeigen"):
    recs = recommend(df, M, top_labels, user_vec, topn=topn, use_diversity=use_diversity)
    if not recs:
        st.info("Keine Empfehlungen gefunden.")
    else:
        out_rows = []
        for r_idx, r in enumerate(recs, 1):
            st.write(f"{r_idx}. {r['title']} ({r['type']})  |  Score {r['score']:.3f}  |  Moods: {', '.join(r['top_labels'])}")
            out_rows.append({
                "rank": r_idx,
                "title": r["title"],
                "type": r["type"],
                "score": r["score"],
                "moods": ", ".join(r["top_labels"])
            })
        out_df = pd.DataFrame(out_rows)
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Ergebnisse als CSV herunterladen", data=csv_bytes, file_name="recommendations.csv", mime="text/csv")
