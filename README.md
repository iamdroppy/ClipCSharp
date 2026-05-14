# ClipCSharp

A technical C# port of OpenAI CLIP (Contrastive Language–Image Pre-Training), implemented with **TorchSharp**.

This repository focuses on architecture parity with the CLIP Python design (dual encoder + contrastive training + zero-shot inference), while using idiomatic .NET code and explicit tensor-shape handling.

---

## Why this project exists

`ClipCSharp` is a practical/educational port of CLIP for .NET developers who want:

- A readable CLIP implementation in C#.
- End-to-end control over text+vision encoding, contrastive loss, and inference.
- A baseline for extending into production scenarios (real tokenizer assets, model checkpoints, larger-scale training, ONNX/export pipelines, etc.).

---

## Current implementation status

### ✅ Implemented

- CLIP dual-encoder architecture:
  - Vision Transformer image encoder (`VisionEncoder`)
  - GPT-style causal transformer text encoder (`TextEncoder`)
- Shared embedding projection + L2 normalization for both modalities.
- Learnable temperature (`LogitScale`) with stability clamp.
- Symmetric InfoNCE contrastive loss (`ContrastiveLoss`).
- Minimal training loop (`ClipTrainer`) with AdamW and optional warmup+cosine LR schedule.
- Image preprocessing pipeline matching CLIP conventions:
  - resize shorter side (bicubic)
  - center crop
  - CLIP mean/std normalization
- Zero-shot classifier utility with class embedding caching.
- Byte-level BPE tokenizer implementation + minimal built-in demo vocab.

### ⚠️ Important limitations (current state)

- The demo uses `BpeTokenizer.CreateMinimal()` (character-level fallback style demo tokenizer), **not** the full OpenAI CLIP tokenizer assets by default.
- No preloaded OpenAI checkpoint conversion/loading is wired in yet.
- `ClipCSharp.csproj` currently targets **`net10.0`** and sets `RuntimeIdentifier` to **`osx-x64`**.

---

## Repository layout

```text
ClipCSharp/
  ClipCSharp.csproj
  Program.cs
  src/
    Inference/
      ImagePreprocessor.cs
      ZeroShotClassifier.cs
    Models/
      ClipConfig.cs
      ClipModel.cs
      MultiHeadAttention.cs
      TextEncoder.cs
      TransformerBlock.cs
      VisionEncoder.cs
    Tokenizer/
      BpeTokenizer.cs
    Training/
      ClipTrainer.cs
      ContrastiveLoss.cs
```

---

## Architecture details

## 1) Top-level CLIP model

`ClipModel` composes:

- `VisionEncoder` → image embeddings `[B, D]`
- `TextEncoder` → text embeddings `[B, D]`
- `LogitScale` (learnable scalar, exponentiated and clamped)

Training forward:

- Input images: `[B, 3, H, W]`
- Input tokens: `[B, T]`
- Output: `(imageFeatures [B, D], textFeatures [B, D], scale scalar)`

Logits:

- `logits = scale * (imageFeatures @ textFeatures^T)` → `[B, B]`

Both feature sets are L2-normalized before similarity.

## 2) Vision encoder (`VisionEncoder`)

Pipeline:

1. Patch embedding via `Conv2d(kernel=stride=PatchSize)`
2. Flatten patches into token sequence
3. Prepend learned `[CLS]`
4. Add learned positional embeddings
5. Pre-LN transformer stack (`TransformerBlock`, full attention)
6. Post-LN
7. Take `[CLS]` representation
8. Linear projection to shared embedding dimension
9. L2 normalization

Typical shapes (ViT-B/32, input 224x224):

- Patches: `G = 224/32 = 7`, `N = 49`
- Sequence length: `N + 1 = 50`
- Transformer tensor: `[B, 50, VisionWidth]`
- Output: `[B, EmbedDim]`

## 3) Text encoder (`TextEncoder`)

Pipeline:

1. Token embedding + position embedding
2. Causal masked transformer stack
3. Final LayerNorm
4. Select representation at EOT position
5. Linear projection to shared embedding dimension
6. L2 normalization

Notes:

- Causal mask is additive with `-∞` in upper triangle.
- EOT position is selected as `argmax(token_id)` following CLIP convention when token IDs preserve EOT ordering.

## 4) Transformer internals

`TransformerBlock` uses pre-norm residual design:

- `x = x + MHA(LN1(x))`
- `x = x + FC2(GELU(FC1(LN2(x))))`

`MultiHeadAttention`:

- Q/K/V linear projections (no bias)
- scaled dot-product attention
- optional additive mask support (`null` for vision, causal mask for text)
- output projection back to model width

---

## CLIP configuration presets

`ClipConfig` provides architecture presets:

- `ViTB32()` (default)
- `ViTB16()`
- `ViTL14()`

Core fields include:

- text: vocab/context/width/layers/heads
- vision: image size/patch size/width/layers/heads
- `EmbedDim`
- `InitLogitScale`

Derived helpers:

- `GridSize = ImageSize / PatchSize`
- `NumPatches = GridSize * GridSize`
- `SeqLenVision = NumPatches + 1`

---

## Inference pipeline

## Image preprocessing (`ImagePreprocessor`)

`LoadFromFile(path)`:

- decodes with SkiaSharp
- converts to RGBA8888
- strips alpha
- returns `[3, H, W]` float tensor in `[0,255]`

`Preprocess(image, resolution=224)`:

1. convert to float and scale to `[0,1]`
2. resize shorter side to `resolution` (bicubic)
3. center crop to `[resolution, resolution]`
4. normalize using CLIP constants:
   - mean: `[0.48145466, 0.4578275, 0.40821073]`
   - std:  `[0.26862954, 0.26130258, 0.27577711]`

Accepted input ranks:

- single image: `[C,H,W]`
- batch: `[B,C,H,W]`

## Zero-shot classification (`ZeroShotClassifier`)

Flow:

1. `SetClasses(classNames, template)` pre-encodes and caches class text embeddings.
2. `Probabilities(images)` computes:
   - `imgFeat = EncodeImage(images)`
   - `logits = scale * (imgFeat @ classEmbeddings^T)`
   - `softmax(logits)`
3. `Classify(images)` returns argmax indices.
4. `TopK(image, k)` returns `(className, probability)` pairs.

The label encoder processes classes in mini-batches (size 64) to reduce memory pressure.

---

## Tokenization details

`BpeTokenizer` implements GPT-2/CLIP-style byte-level BPE mechanics:

- Regex token splitting compatible with CLIP-style rules.
- Byte-to-Unicode reversible mapping (GPT-2 method).
- Merge-rank based pair merging.
- Special tokens:
  - `<|startoftext|>`
  - `<|endoftext|>`
- `TokenizeAndPad(text, contextLength=77)` adds SOT/EOT and truncates/pads.

### Production note

For production parity with OpenAI CLIP checkpoints, load the **real** tokenizer assets (`vocab.json` + `merges.txt`) and construct `BpeTokenizer` from those assets instead of `CreateMinimal()`.

---

## Training details

`ContrastiveLoss.Compute(imageFeatures, textFeatures, logitScale)`:

- computes similarity matrix `[B,B]`
- labels are diagonal matches (`0..B-1`)
- symmetric objective:
  - CE(image→text) + CE(text→image)
  - averaged by 0.5

`ClipTrainer`:

- uses AdamW (`β1=0.9, β2=0.98, ε=1e-6`)
- default LR `5e-4`, default weight decay `0.2`
- supports cosine schedule with linear warmup (`SetLearningRate`)

Expected batch contract:

- images `[B,3,H,W]` (already preprocessed and device-correct)
- tokens `[B,T]` (`int64`, device-correct)

---

## Build and run

## Requirements

- .NET SDK supporting the target framework configured in `ClipCSharp.csproj` (`net10.0` currently).
- Native dependencies pulled via TorchSharp packages.

## Packages (from project file)

- `TorchSharp` 0.102.6
- `TorchSharp-cpu` 0.102.6
- `SkiaSharp` 2.88.6

## Run demo

From repository root:

```bash
dotnet run --project ClipCSharp/ClipCSharp.csproj
```

The demo in `Program.cs`:

- selects CUDA if available, else CPU
- constructs ViT-B/32 config/model
- loads `input.jpg`
- preprocesses image to `[1,3,224,224]`
- tokenizes sample prompts
- prints cosine similarities between image and prompt embeddings

---

## Practical parity notes vs CLIP Python

This port preserves the key CLIP mechanics:

- dual encoder design
- L2-normalized shared embedding space
- temperature-scaled cosine similarity
- symmetric contrastive objective
- CLIP-style image normalization constants

To reach stronger checkpoint-level parity with OpenAI Python release behavior, typical next steps are:

1. Full official tokenizer asset loading path.
2. Weight import/conversion from known CLIP checkpoints.
3. Numerical parity tests on fixed prompts/images.
4. Optional mixed precision / larger-batch training ergonomics.

---

## Known caveats

- `RuntimeIdentifier` is pinned to `osx-x64` in the current project file; adjust if targeting other platforms.
- Tokenizer default in demo is intentionally minimal and not benchmark-grade.
- `net10.0` is configured in project file; ensure local SDK/toolchain alignment.

---

## Suggested next engineering tasks

- Add `Tokenizer.LoadFromFiles(vocabPath, mergesPath)` helper.
- Add checkpoint serialization + loading (`state_dict` equivalent handling in TorchSharp).
- Add unit tests for:
  - shape contracts
  - tokenizer round-trip/edge cases
  - causal mask correctness
  - contrastive loss symmetry
- Add benchmark harness (throughput/latency per batch size/device).
- Add ONNX export + ONNX Runtime inference path (optional for deployment).

---

## License

No license file is included in this repository snapshot. Add a `LICENSE` file to make usage rights explicit.
