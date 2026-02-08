import requests
import re
import time

# =============================
# CONFIG
# =============================
API_KEY = "usr-xjsbUTmRnRMb-qfo1kfinVDDon-Bvq27v8O737ZdvYY"
BASE_URL = "https://trefle.io/api/v1/plants"
MAX_PLANTS = 2500
OUTPUT_FILE = "plants.js"

# =============================
# HELPERS
# =============================
def slugify(name):
    return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_"))

def light_text(light):
    if light is None:
        return "Moderate light."
    if light < 5:
        return "Low to moderate light."
    if light < 8:
        return "Bright, indirect light."
    return "Full sun."

def humidity_text(h):
    if h is None:
        return "Moderate humidity."
    if h < 4:
        return "Low humidity tolerant."
    if h < 7:
        return "Moderate humidity."
    return "High humidity preferred."

def soil_text(ph_min, ph_max):
    if ph_min is None or ph_max is None:
        return "Well-draining soil."
    if ph_max < 6.5:
        return "Slightly acidic, well-draining soil."
    if ph_min > 7:
        return "Alkaline, well-draining soil."
    return "Neutral, well-draining soil."

# =============================
# FETCH PLANTS SAFELY
# =============================
plants = []
page = 1

while len(plants) < MAX_PLANTS:
    print(f"📥 Fetching page {page}...")

    resp = requests.get(
        BASE_URL,
        params={"token": API_KEY, "page": page},
        timeout=20
    )

    data = resp.json()

    if "data" not in data:
        print(f"❌ API error on page {page}: {data}")
        break

    for plant in data["data"]:
        plants.append(plant)
        if len(plants) >= MAX_PLANTS:
            break

    if not data.get("links", {}).get("next"):
        break

    page += 1
    time.sleep(0.8)

print(f"✅ Retrieved {len(plants)} plants")

# =============================
# BUILD JS FILE
# =============================
entries = []

for p in plants:
    sci_name = p.get("scientific_name")
    if not sci_name:
        continue

    key = slugify(sci_name)
    growth = p.get("growth") or {}

    # ---- FAMILY FIX (IMPORTANT) ----
    family_raw = p.get("family")
    if isinstance(family_raw, dict):
        family = family_raw.get("name", "Unknown")
    elif isinstance(family_raw, str):
        family = family_raw
    else:
        family = "Unknown"

    native = p.get("native_distribution") or []
    native_text = ", ".join(native[:2]) if native else "various regions"

    habit = growth.get("habit") or "distinct growth habit"

    entry = f'''
  {key}: {{
    commonName: "{p.get("common_name") or sci_name}",
    scientificName: "{sci_name}",
    family: "{family}",
    description:
      "A plant species native to {native_text}, known for its {habit}.",
    careGuide: {{
      water: "{humidity_text(growth.get("atmospheric_humidity"))}",
      light: "{light_text(growth.get("light"))}",
      temperature: "Thrives in mild to warm climates.",
      humidity: "{humidity_text(growth.get("atmospheric_humidity"))}",
      soil: "{soil_text(growth.get("ph_min"), growth.get("ph_max"))}"
    }},
    characteristics: [
      "{habit}",
      "{'Edible plant' if p.get('edible') else 'Non-edible plant'}",
      "{'Toxic if ingested' if p.get('toxicity') else 'Non-toxic'}"
    ],
    growthInfo: {{
      size: "{(p.get('maximum_height') or {}).get('cm', 'Varies')} cm",
      growthRate: "Unknown",
      lifecycle: "{', '.join(p.get('duration', [])) or 'Perennial'}"
    }}
  }}
'''
    entries.append(entry)

# =============================
# WRITE OUTPUT
# =============================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("export const PLANT_INFO = {\n")
    f.write(",\n".join(entries))
    f.write("\n};")

print(f"🎉 plants.js created successfully with {len(entries)} entries")