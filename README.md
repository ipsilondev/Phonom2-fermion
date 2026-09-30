---
license: cc-by-4.0
tags:
- speech-recognition
- asr
- quantized
- ternary
- fermion
- fastconformer
- tdt
---

# Phonon-2 Inference Model Pack (`ipsilondev/Phonom2-fermion`)

Stand-alone inference package and Git submodule containing weights, manifests, and offline tokenizer for **Phonon-2** (2.1-bit ternary FastConformer 0.6B + TDT ASR).

## Files Included

| File | Size | Purpose |
| :--- | :--- | :--- |
| **`model.fermion`** | 169.2 MB | 2.1-bit quint5 packed model container (24-layer FastConformer + 2-layer LSTM decoder + joint) |
| **`config.json`** | 277 KB | Architecture hyperparameters, mel filterbank settings, and model metadata |
| **`packed_manifest.json`** | 821 B | Manifest indexing packed ternary tensor offsets, trit planes, and scale vectors |
| **`bps_manifest.json`** | 799 B | Fermion package manifest |
| **`tokenizer.json`** | 1.1 MB | Offline vocabulary and SentencePiece BPE tokenizer (8,193 tokens) |
| **`tokenizer_config.json`** | 352 B | Tokenizer configuration |
| **`processor_config.json`** | 408 B | Audio preprocessor configuration (128 mel bins, 16kHz) |
| **`reference_transformers.py`** | 6.8 KB | Stand-alone PyTorch / Transformers model loader |
| **`fermion_container.py`** | 3.9 KB | Stand-alone container decompressor and trit dequantizer |

---

## Submodule Usage

Add this repository as a Git submodule in your downstream projects:

### Via Hugging Face (Recommended for large model weights)
```bash
git submodule add https://huggingface.co/ipsilondev/Phonom2-fermion models/phonon2
git submodule update --init --recursive
```

### Via GitHub
```bash
git submodule add https://github.com/ipsilondev/Phonom2-fermion.git models/phonon2
git submodule update --init --recursive
git lfs pull
```

---

## Python Inference Quickstart

```python
import torch
import reference_transformers as rt

# Load model using local container and tokenizer (completely offline)
model, processor, _ = rt.load_model(
    container_path="models/phonon2/model.fermion",
    base_dir="models/phonon2",
    dtype=torch.float32
)
model.eval()

# Process audio
import soundfile as sf
audio, sr = sf.read("test.wav")
inputs = processor([audio], sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    out = model.generate(input_features=inputs["input_features"])

text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
print("Transcript:", text)
```

## License

Model weights: CC-BY-4.0 (derived from NVIDIA Parakeet TDT 0.6B).
Code & tools: Apache-2.0.
