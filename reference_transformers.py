"""Dense `transformers` reference for the Phonon-2 container.

read container -> expand every record to fp32 (five-value: exact {0,+-lo,+-hi}; intN: q*scale;
fp16: as stored) -> stock ParakeetForTDT(config).load_state_dict(strict=True) -> greedy TDT
generate through ParakeetProcessor, the same call the leaderboard scoring uses.

Usage: python reference_transformers.py model.fermion --audio recording.wav
       python reference_transformers.py model.fermion --utts rows.json --out receipt.json
Outputs per utterance: transcript (+ token ids) and, for --dump-encoder N utts, the encoder
output in fp32 so the MLX graph can be compared against the torch graph.  Pure reference:
no speed claim is made or measured here.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fermion_container import read_container  # noqa: E402


def container_state_dict(container: str):
    """HF-named fp32 torch state dict; num_batches_tracked restored to int64."""
    import torch
    tensors, index = read_container(container, with_raw=False)
    sd = {}
    for k, v in tensors.items():
        if k.endswith("num_batches_tracked"):
            # the writer stores this int64 BatchNorm counter as fp16 -> inf for counts >= 65,520;
            # BatchNorm.eval() never reads it, so a non-finite value becomes 0
            v32 = float(np.asarray(v, dtype=np.float32).reshape(-1)[0])
            sd[k] = torch.tensor(int(v32) if np.isfinite(v32) else 0, dtype=torch.int64)
        else:
            arr = np.ascontiguousarray(v.astype(np.float32))
            # the writer squeezed the kernel-1 pointwise convs to [O, I]; stock ParakeetForTDT wants [O, I, 1]
            if arr.ndim == 2 and re.search(r"\.conv\.pointwise_conv[12]\.weight$", k):
                arr = arr[:, :, None]
            sd[k] = torch.from_numpy(arr)
    return sd, index


def load_model(container: str, base_dir: str, dtype=None):
    import torch
    from transformers import AutoProcessor, ParakeetForTDT, ParakeetTDTConfig
    cfg = ParakeetTDTConfig.from_pretrained(base_dir)
    model = ParakeetForTDT(cfg)
    sd, index = container_state_dict(container)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    receipt = {"missing": list(missing), "unexpected": list(unexpected),
               "container_records": len(index), "state_dict_keys": len(sd),
               "params": sum(p.numel() for p in model.parameters())}
    if missing or unexpected:
        raise RuntimeError(f"G-LOAD (transformers): missing {list(missing)[:5]} unexpected {list(unexpected)[:5]}")
    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    # a model built from its config carries no generation_config; the base repo's is authoritative
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig.from_pretrained(base_dir)
    receipt["generation_config"] = {k: getattr(model.generation_config, k, None)
                                    for k in ("decoder_start_token_id", "bos_token_id", "pad_token_id", "max_symbols_per_step")}
    processor = AutoProcessor.from_pretrained(base_dir)
    return model, processor, receipt


def main():
    import argparse
    import torch
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("container")
    ap.add_argument("--base-dir", default="nvidia/parakeet-tdt-0.6b-v3",
                    help="the base repo (config, processor, generation config); a Hub id or a local directory")
    ap.add_argument("--audio", nargs="+", metavar="FILE",
                    help="audio file(s) to transcribe (any sample rate; resampled to 16 kHz, mixed to mono)")
    ap.add_argument("--utts", default=None, help="JSON list of {id, path, ref} rows (the evaluation rows); ignored with --audio")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--dump-encoder", type=int, default=None,
                    help="dump the fp32 encoder output of the first N utterances (default 4 with --utts, 0 with --audio)")
    ap.add_argument("--out", default=None, help="write transcripts (+ token ids) as JSON here; default: print only")
    a = ap.parse_args()
    if not a.audio and not a.utts:
        ap.error("give --audio FILE [FILE ...] or --utts rows.json")
    if a.dump_encoder is None:
        a.dump_encoder = 0 if a.audio else 4
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    dtype = getattr(torch, a.dtype)
    t0 = time.perf_counter()
    model, processor, rc = load_model(a.container, a.base_dir, dtype=dtype)
    rc["load_s"] = round(time.perf_counter() - t0, 1)
    rc["torch"] = torch.__version__
    import transformers; rc["transformers"] = transformers.__version__
    rc["dtype"] = a.dtype
    if a.audio:
        utts = [{"id": Path(f).stem, "path": f, "split": "file", "ref": None} for f in a.audio]
    else:
        utts = json.load(open(a.utts))[: a.n]
    rows = []
    enc_dump = {}
    t1 = time.perf_counter()
    for i, u in enumerate(utts):
        w, sr = sf.read(u["path"], dtype="float32", always_2d=True)
        w = w.mean(axis=1) if w.shape[1] > 1 else w[:, 0]
        if sr != 16000:
            try:
                import librosa
                w = librosa.resample(w, orig_sr=sr, target_sr=16000).astype(np.float32)
            except ImportError:
                import torchaudio.functional as taf
                w = taf.resample(torch.from_numpy(w), sr, 16000).numpy().astype(np.float32)
            sr = 16000
        inp = processor([w], sampling_rate=16000, return_tensors="pt", padding=True)
        feats = inp["input_features"].to(dtype)
        with torch.no_grad():
            if i < a.dump_encoder:
                enc = model.encoder(input_features=feats, attention_mask=inp.get("attention_mask"))
                enc_dump[u["id"]] = enc.last_hidden_state[0].float().numpy()
            out = model.generate(input_features=feats, attention_mask=inp.get("attention_mask"))
        seq = getattr(out, "sequences", out)
        text = processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
        ids = [int(x) for x in seq[0].tolist()]
        rows.append({"id": u["id"], "split": u["split"], "text": text, "token_ids": ids, "ref": u["ref"]})
        print(f"{u['id']:>18} {text[:90]}")
    rc["decode_wall_s"] = round(time.perf_counter() - t1, 1)
    if a.out:
        json.dump({"load": rc, "rows": rows}, open(a.out, "w"), indent=1)
        if enc_dump:
            np.savez(a.out.replace(".json", "_encoder.npz"), **enc_dump)
        print(json.dumps(rc, indent=1))


if __name__ == "__main__":
    main()
