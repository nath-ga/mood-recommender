# README.txt
# Mood Recommender (Proof of Concept)

## 🎯 Projektidee
Viele Menschen wählen Filme oder Hörbücher nicht nur nach Genre, sondern nach Stimmung.  
Dieses Projekt untersucht, ob sich **Stimmungs-Labels (Moods)** automatisch aus Inhaltsbeschreibungen ableiten lassen 
und wie man daraus einfache **Mood-Empfehlungen** generieren kann.

## 📂 Projektstruktur
mood-recommender/
├── README.md
├── requirements.txt
├── mood_probe.py # Zero-Shot Mood Labeling
├── data/ # Beispieldaten
│ └── sample.csv
└── outputs/ # Ergebnisse (generierte Mood-Labels)
└── sample_moods.csv

## ⚙️ Installation
1. Repository klonen oder Ordner lokal anlegen  
2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
🚀 Nutzung
Einzeltext testen
python mood_probe.py --text "Eine warme, leichte Komödie über Freundschaft und Neuanfang."
→ Ausgabe: Top-3 Moods mit Scores

CSV verarbeiten
python mood_probe.py --in data/sample.csv --out outputs/sample_moods.csv --lang both --topk 3
→ Ausgabe: Neue CSV mit zusätzlichen Spalten moods_topk und moods_scores.

🧪 Aktueller Stand
Erste Tests erfolgreich: Automatisches Labeling funktioniert. Beispiele mit 3 Filmbeschreibungen zeigen plausible Mood-Zuordnungen
Wenn ja, wird daraus ein kleiner Mood-Recommender (z. B. per Streamlit).

📌 Hinweis
Alle Labels entstehen maschinell über Zero-Shot-Textklassifikation.

Ziel ist ein Proof of Concept für das Portfolio, nicht ein perfektes Empfehlungssystem.