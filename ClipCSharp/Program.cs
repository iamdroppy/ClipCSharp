using ClipCSharp.Inference;
using ClipCSharp.Models;
using ClipCSharp.Tokenizer;
using ClipCSharp.Training;
using TorchSharp;
using static TorchSharp.torch;

// ============================================================
//  CLIP (Contrastive Language–Image Pre-Training) in C#
//  using TorchSharp (LibTorch C# bindings)
// ============================================================

Console.WriteLine("=== CLIP C# Demo ===\n");

// ------------------------------------------------------------------
// 1. Choose device (GPU if available, else CPU)
// ------------------------------------------------------------------
var device = cuda.is_available() ? CUDA : CPU;
Console.WriteLine($"Device: {device.type}");

// ------------------------------------------------------------------
// 2. Build model (ViT-B/32 default)
// ------------------------------------------------------------------
var cfg   = ClipConfig.ViTB32();
var model = new ClipModel(cfg);
model.to(device);
model.eval();

long paramCount = model.parameters().Sum(p => p.numel());
Console.WriteLine($"Parameters: {paramCount:N0}");
Console.WriteLine($"  Vision encoder:  ViT-B/{cfg.PatchSize}  " +
                  $"({cfg.VisionWidth}d x {cfg.VisionLayers}L x {cfg.VisionHeads}H)");
Console.WriteLine($"  Text encoder:    GPT-2 style  " +
                  $"({cfg.TextWidth}d x {cfg.TextLayers}L x {cfg.TextHeads}H)");
Console.WriteLine($"  Shared emb dim:  {cfg.EmbedDim}");

// ------------------------------------------------------------------
// 3. Tokeniser (minimal demo vocab; replace with the real CLIP vocab
//    loaded from vocab.json + merges.txt for production use)
// ------------------------------------------------------------------
var tokenizer = BpeTokenizer.CreateMinimal();
Console.WriteLine($"\nTokenizer vocab size: {tokenizer.VocabSize}");

// Load `input.jpg` from the current working directory and run it through
// the full CLIP preprocessing pipeline (resize → centre-crop → normalise).
const string inputImagePath = "input.jpg";
Console.WriteLine($"Loading image: {Path.GetFullPath(inputImagePath)}");

using var rawImage = ImagePreprocessor.LoadFromFile(inputImagePath, device);
                                                            // [3, H, W] float in [0, 255]
using var image    = ImagePreprocessor.Preprocess(rawImage, cfg.ImageSize)
                                      .unsqueeze(0);        // [1, 3, 224, 224]

Console.WriteLine("\n--- Embedding Similarity Demo ---");

string[] descriptions =
[
    "a photo of a car",
    "a photo of a red car",
    "a photo of a blue car",
    "a cat",
];

// Tokenise all descriptions -> [4, 77]
var tokenBatch = stack(
    descriptions.Select(d =>
        tensor(tokenizer.TokenizeAndPad(d, cfg.ContextLength),
               dtype: ScalarType.Int64, device: device))
    .ToArray());

Tensor imgFeat = model.EncodeImage(image);       // [1, D]
Tensor txtFeat = model.EncodeText(tokenBatch);   // [4, D]

// Cosine similarities: [1, 4]
Tensor sim = matmul(imgFeat, txtFeat.t());

Console.WriteLine("Cosine similarity (input.jpg vs text prompts):");
for (int i = 0; i < descriptions.Length; i++)
    Console.WriteLine($"  \"{descriptions[i]}\"  ->  {sim[0, i].item<float>():F4}");