using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ClipCSharp.Models;

/// <summary>
/// Vision Transformer (ViT) used as CLIP's image encoder.
///
/// Architecture:
///   Conv2d patch embedding (stride = patch size)  →  flatten patches
///   prepend [CLS] token  +  learnable position embeddings
///   → LayerNorm (pre-norm)
///   → N × TransformerBlock (full attention, no causal mask)
///   → LayerNorm
///   → take [CLS] token embedding
///   → linear projection to shared embedding space
/// </summary>
public sealed class VisionEncoder : Module<Tensor, Tensor>
{
    private readonly Conv2d    _patchEmb;
    private readonly Parameter _clsToken;
    private readonly Parameter _posEmb;
    private readonly LayerNorm _preNorm;
    private readonly ModuleList<TransformerBlock> _blocks;
    private readonly LayerNorm _postNorm;
    private readonly Linear    _proj;

    public VisionEncoder(ClipConfig cfg) : base(nameof(VisionEncoder))
    {
        int numPatches = cfg.NumPatches;

        // Patch embedding: treat each (P×P) patch as a token
        _patchEmb = Conv2d(
            in_channels:  3,
            out_channels: cfg.VisionWidth,
            kernelSize:   cfg.PatchSize,
            stride:       cfg.PatchSize,
            bias:         false);

        _clsToken = Parameter(zeros(1, 1, cfg.VisionWidth));
        _posEmb   = Parameter(zeros(1, numPatches + 1, cfg.VisionWidth));

        _preNorm = LayerNorm(cfg.VisionWidth);

        _blocks = new ModuleList<TransformerBlock>(
            Enumerable.Range(0, cfg.VisionLayers)
                      .Select(i => new TransformerBlock($"vision_block_{i}", cfg.VisionWidth, cfg.VisionHeads))
                      .ToArray());

        _postNorm = LayerNorm(cfg.VisionWidth);
        _proj     = Linear(cfg.VisionWidth, cfg.EmbedDim, hasBias: false);

        // Initialise with small random values (matches ViT paper)
        init.normal_(_posEmb, mean: 0.0, std: 0.02);
        init.normal_(_clsToken, mean: 0.0, std: 0.02);

        RegisterComponents();
    }

    /// <param name="images">Float image tensor [B, 3, H, W] normalised to ≈ [0,1].</param>
    /// <returns>L2-normalised image embeddings [B, EmbedDim].</returns>
    public override Tensor forward(Tensor images)
    {
        long B = images.shape[0];

        // 1. Patch embedding: [B, 3, H, W] → [B, VisionWidth, G, G] → [B, G*G, VisionWidth]
        Tensor patches = _patchEmb.forward(images);   // [B, W, G, G]
        patches = patches.flatten(start_dim: 2)        // [B, W, G*G]
                         .transpose(1, 2);             // [B, G*G, W]

        // 2. Prepend [CLS] token
        Tensor cls = _clsToken.expand(B, 1, -1);      // [B, 1, W]
        Tensor x   = cat(new[] { cls, patches }, dim: 1); // [B, G*G+1, W]

        // 3. Add positional embeddings (broadcast over B)
        x = x + _posEmb;

        // 4. Pre-norm
        x = _preNorm.forward(x);

        // 5. Transformer blocks (no causal mask — full attention)
        foreach (var block in _blocks)
            x = block.forward(x, mask: null);

        // 6. Post-norm, take [CLS] at position 0
        x = _postNorm.forward(x);
        Tensor clsOut = x.select(dim: 1, index: 0);  // [B, VisionWidth]

        // 7. Project to shared space and L2-normalise
        Tensor projected = _proj.forward(clsOut);     // [B, EmbedDim]
        return L2Normalize(projected);
    }

    // Manual L2 normalisation (functional.normalize not in TorchSharp 0.103)
    private static Tensor L2Normalize(Tensor x, int dim = -1)
    {
        Tensor nrm = x.norm(dim, keepdim: true, p: 2f);
        return x / nrm.clamp(min: 1e-12f);
    }
}
