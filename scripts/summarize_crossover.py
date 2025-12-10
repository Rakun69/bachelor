import json
import pandas as pd
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# KONFIGURATION: Pfad zur JSON-Datei (einfach hier anpassen)
# ---------------------------------------------------------
json_path = r"/home/ramon/bachelor/data/visualizations/md_warm_nr7/crossover_results.json"
# ---------------------------------------------------------

# Ordner der JSON-Datei automatisch bestimmen
output_dir = os.path.dirname(json_path)

# JSON roh einlesen
with open(json_path, "r") as f:
    raw = f.read()

# 1) Normales JSON?
try:
    data = json.loads(raw)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
except json.JSONDecodeError:
    data = None

# 2) NDJSON fallback
if data is None:
    data = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except:
                    pass

rows = []

for entry in data:
    if not isinstance(entry, dict):
        continue

    iot = entry.get("iot_readings")

    std = entry.get("standard", {})
    nova = entry.get("nova", {})

    rows.append({
        "IoT Readings": iot,
        "Std Prove (s)": std.get("prove_time_s"),
        "Std Verify (s)": std.get("verify_time_s"),
        "Nova Prove (s)": nova.get("prove_time_s"),
        "Nova Verify (s)": nova.get("verify_time_s"),
    })

df = pd.DataFrame(rows)
df = df.sort_values("IoT Readings")

# -----------------------------
# CSV speichern (im gleichen Ordner)
# -----------------------------
csv_path = os.path.join(output_dir, "crossover_overview_summary.csv")
df.to_csv(csv_path, index=False)

print(f"CSV gespeichert in:\n{csv_path}")
