# README.txt
# Mood Recommender (Proof of Concept)
**Repo:** [mood-recommender](https://github.com/nath-ga/mood-recommender)  

## 🎯 Projektidee
Viele Menschen wählen Filme oder Hörbücher nicht nur nach Genre, sondern nach Stimmung.  
Dieses Projekt untersucht, ob sich **Stimmungs-Labels (Moods)** automatisch aus Inhaltsbeschreibungen ableiten lassen 
und wie man daraus einfache **Mood-Empfehlungen** generieren kann.

## 📂 Projektstruktur
```
mood-recommender/
├── README.md
├── requirements.txt
├── mood_probe.py # Zero-Shot Mood Labeling
├── mood_recommender.py # Mood-basiertes Empfehlungssystem (Cosine Similarity)
├── data/ # Beispieldaten
│ └── sample.csv
├── outputs/ # Ergebnisse (generierte Mood-Labels)
│ └── sample_moods.csv
└── cache/ # Lokaler Cache für schnellere Wiederholungen (ignored)
```

⚙️ Installation
1. Repository klonen oder Ordner lokal anlegen  
2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt

🚀 Nutzung
1. Einzeltext testen

python mood_probe.py --text "Eine warme, leichte Komödie über Freundschaft und Neuanfang."
→ Ausgabe: Top-3 Moods mit Scores

2. CSV verarbeiten
python mood_probe.py --in data/sample.csv --out outputs/sample_moods.csv --lang both --topk 3
→ Ausgabe: Neue CSV mit zusätzlichen Spalten moods_topk und moods_scores.

3. Empfehlungen erzeugen
Slider-Modus (Valenz/Energie):

python mood_recommender.py --csv outputs/sample_moods.csv --valence 0.7 --energy 0.2 --topn 5
Freitext-Modus (optional, Hugging Face nötig):

python mood_recommender.py --csv outputs/sample_moods.csv --user_text "Bin erschöpft, hätte gern etwas Heiteres." --topn 5
→ Ausgabe: Top-N Empfehlungen mit Score und Mood-Begründung.

🧪 Aktueller Stand
Automatisches Labeling funktioniert (Zero-Shot Hugging Face).
Erste Empfehlungen mit Cosine Similarity sind möglich.
Serendipity-Mechanismus: Falls Liste zu einseitig ist, wird ein „hellerer“ Titel ergänzt.

📌 Hinweise
Alle Labels entstehen maschinell über Zero-Shot-Textklassifikation.
Eingeschlossene Daten (sample.csv, sample_moods.csv) sind synthetische Beispiele.
Eigene Tests mit größeren CSV-Dateien sind möglich, aber nicht Teil des Repos.
Ziel ist ein Proof of Concept für das Portfolio, nicht ein perfektes Empfehlungssystem.

🧭 So funktioniert es 
Du wählst, wie du dich gerade fühlst – von „eher schlecht“ bis „richtig gut“ und ob du eher ruhig oder aktiv bist.  
Die App sucht dann Titel, deren Stimmung dazu passt (oder – je nach Modus – dich eher **aufhellt**).  
Dafür vergleicht sie Stimmungsprofile der Titel mit deiner aktuellen Stimmung.

Eingaben
- **Stimmungslage:** schlecht ↔ gut  
- **Energielevel:** ruhig ↔ aktiv  
- **Empfehlungsmodus:**  
  - **passend** – Treffer, die deiner Stimmung ähneln  
  - **aufhellen** – Vorschläge mit helleren, beruhigenden Moods  
  - **Mischung** – ein Mix aus beidem  
- **Optional:** Kurzer Freitext wie „Langer Tag, etwas Leichtes wäre gut.“

Hinweise zu Daten
- Du kannst eine eigene CSV laden (UTF-8, Spalten: `title`, `description`, `moods_topk`, `moods_scores`, optional `type`).  
- Im Repo liegt eine kleine **Beispiel-CSV** nur zum Ausprobieren.

Datenschutz
- Deine Eingaben bleiben lokal in der App.  
- Bei Nutzung des Freitext-Modus wird ein lokales Sprachmodell verwendet (kein Upload zu Dritten).

### Präsentation

![Slide 1](presentation/1.png)
![Slide 2](presentation/2.png)
![Slide 3](presentation/3.png)
![Slide 4](presentation/4.png)
![Slide 5](presentation/5.png)
![Slide 6](presentation/6.png)
![Slide 7](presentation/7.png)

📜 Lizenz
Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).

➡️ **Hinweis:** Sie dürfen den Code gerne für eigene Experimente, Lernprojekte oder Weiterentwicklungen verwenden.  
Ich freue mich jedoch, wenn Sie mir kurz Bescheid geben, falls Sie ihn veröffentlichen oder in größerem Rahmen nutzen.

