"""NVIDIA Nemotron ASR as a worker process: one of the ASR experts of the MoE score (task 3) for English.
It needs transformers >= 5.13, which the main .venv (4.57, for pyannote and the other models) cannot take, so it runs
in its own venv (see README) and utility_functions talks to it over stdin/stdout:
    in:  one line per request, the path of an .npz holding 16 kHz float32 pieces piece_0, piece_1, ...
    out: one JSON line per request, {"texts": [...]} (one text per piece) or {"error": "..."}"""
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoProcessor


def main(model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForRNNT.from_pretrained(model_id).to(device).eval()
    sr = processor.feature_extractor.sampling_rate
    print(json.dumps({"ready": model_id, "device": device}), flush=True)
    for line in sys.stdin:
        try:
            data = np.load(line.strip())
            texts = []
            for k in sorted(data.files, key=lambda k: int(k.split("_")[1])):
                inputs = processor(data[k], sampling_rate=sr, return_tensors="pt").to(device, dtype=model.dtype)
                with torch.inference_mode():
                    out = model.generate(**inputs, return_dict_in_generate=True)
                texts.append(" ".join(processor.decode(out.sequences, skip_special_tokens=True)[0].split()))
            print(json.dumps({"texts": texts}, ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
