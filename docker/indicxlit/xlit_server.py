"""Reads {"lang": "hi", "words": [...]} JSON on stdin, writes {"word": [top candidates]} JSON on stdout."""
import json, sys, numpy as np
if not hasattr(np, "float"):
    np.float = float
from ai4bharat.transliteration import XlitEngine
if sys.argv[1:2] == ["--warmup"]:
    XlitEngine(sys.argv[2], beam_width=4, src_script_type="roman")
    sys.exit()
req = json.load(sys.stdin)
engine = XlitEngine(req["lang"], beam_width=10, src_script_type="roman")
out = {}
for w in req["words"]:
    r = engine.translit_word(w, topk=5)
    out[w] = r.get(req["lang"], []) if isinstance(r, dict) else r
json.dump(out, sys.stdout, ensure_ascii=False)
