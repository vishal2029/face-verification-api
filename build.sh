python3 - <<'PYCODE'
from deepface import DeepFace
print("🧠 Preloading DeepFace ArcFace weights…")
DeepFace.build_model("ArcFace")
print("✅ ArcFace weights ready.")
PYCODE
