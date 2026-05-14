using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ClipCSharp.Models;

/// <summary>
/// GPT-2–style autoregressive transformer used as CLIP's text encoder.
///
/// Architecture (per the paper):
///   token embedding  +  positional embedding
///   → N × TransformerBlock (causal mask)
///   → LayerNorm
///   → take the [EOT] position as the sequence representation
///   → linear projection to shared embedding space
/// </summary>
public sealed class TextEncoder : Module<Tensor, Tensor>
{
    private readonly Embedding   _tokenEmb;
    private readonly Embedding   _posEmb;
    private readonly ModuleList<TransformerBlock> _blocks;
    private readonly LayerNorm   _ln;
    private readonly Linear      _proj;

    public TextEncoder(ClipConfig cfg) : base(nameof(TextEncoder))
    {
        _tokenEmb = Embedding(cfg.VocabSize,     cfg.TextWidth);
        _posEmb   = Embedding(cfg.ContextLength, cfg.TextWidth);

        _blocks = new ModuleList<TransformerBlock>(
            Enumerable.Range(0, cfg.TextLayers)
                      .Select(i => new TransformerBlock($"text_block_{i}", cfg.TextWidth, cfg.TextHeads))
                      .ToArray());

        _ln   = LayerNorm(cfg.TextWidth);
        _proj = Linear(cfg.TextWidth, cfg.EmbedDim, hasBias: false);

        RegisterComponents();
    }

    /// <param name="tokens">Token id tensor [B, T] (int64).</param>
    /// <returns>L2-normalised text embeddings [B, EmbedDim].</returns>
    public override Tensor forward(Tensor tokens)
    {
        long B = tokens.shape[0];
        long T = tokens.shape[1];

        // Build causal mask: upper-triangular −∞, diagonal 0
        Tensor mask = BuildCausalMask(T, tokens.device);

        // Position ids [0 … T-1]
        Tensor pos = arange(T, device: tokens.device).unsqueeze(0).expand(B, T);

        Tensor x = _tokenEmb.forward(tokens) + _posEmb.forward(pos); // [B, T, C]

        foreach (var block in _blocks)
            x = block.forward(x, mask);

        x = _ln.forward(x);  // [B, T, C]

        // CLIP takes the embedding at the [EOT] position (highest token id)
        // as the sequence representation.
        Tensor eotPositions = tokens.argmax(dim: -1);     // [B]
        Tensor rep = ExtractEot(x, eotPositions);          // [B, C]

        // Project to shared space and L2-normalise
        Tensor projected = _proj.forward(rep);             // [B, EmbedDim]
        return L2Normalize(projected);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static Tensor BuildCausalMask(long seqLen, Device device)
    {
        // Upper-triangular mask filled with −∞ so position i cannot attend to j > i.
        Tensor mask = full(new long[] { seqLen, seqLen }, float.NegativeInfinity,
                           dtype: ScalarType.Float32, device: device);
        return triu(mask, diagonal: 1);
    }

    private static Tensor ExtractEot(Tensor x, Tensor eotPositions)
    {
        // x: [B, T, C]   eotPositions: [B]
        long B = x.shape[0];
        long C = x.shape[2];
        Tensor idx = eotPositions.view(B, 1, 1).expand(B, 1, C);  // [B, 1, C]
        return x.gather(dim: 1, index: idx).squeeze(1);            // [B, C]
    }

    // Manual L2 normalisation (functional.normalize not in TorchSharp 0.103)
    private static Tensor L2Normalize(Tensor x, int dim = -1)
    {
        Tensor nrm = x.norm(dim, keepdim: true, p: 2f);
        return x / nrm.clamp(min: 1e-12f);
    }
}
