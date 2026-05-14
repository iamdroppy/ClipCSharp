using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ClipCSharp.Models;

/// <summary>
/// Scaled dot-product multi-head attention.
/// Supports both causal (autoregressive) masking for the text encoder
/// and full (bidirectional) attention for the vision encoder.
/// </summary>
public sealed class MultiHeadAttention : Module<Tensor, Tensor?, Tensor>
{
    private readonly int   _numHeads;
    private readonly int   _headDim;
    private readonly float _scale;

    private readonly Linear _queryProj;
    private readonly Linear _keyProj;
    private readonly Linear _valueProj;
    private readonly Linear _outProj;

    public MultiHeadAttention(string name, int embedDim, int numHeads) : base(name)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException($"embedDim ({embedDim}) must be divisible by numHeads ({numHeads}).");

        _numHeads = numHeads;
        _headDim  = embedDim / numHeads;
        _scale    = 1f / MathF.Sqrt(_headDim);

        _queryProj = Linear(embedDim, embedDim, hasBias: false);
        _keyProj   = Linear(embedDim, embedDim, hasBias: false);
        _valueProj = Linear(embedDim, embedDim, hasBias: false);
        _outProj   = Linear(embedDim, embedDim);

        RegisterComponents();
    }

    /// <param name="x">Input tensor  [B, T, C]</param>
    /// <param name="mask">
    ///   Optional additive attention mask [T, T] (−∞ for masked positions).
    ///   Pass <c>null</c> for full attention (vision encoder).
    ///   Pass a causal mask for the text encoder.
    /// </param>
    /// <returns>[B, T, C]</returns>
    public override Tensor forward(Tensor x, Tensor? mask)
    {
        long B = x.shape[0];
        long T = x.shape[1];
        long C = x.shape[2];

        // Helper: linear → reshape → [B, H, T, D]
        Tensor Reshape(Linear proj)
            => proj.forward(x)
                   .view(B, T, _numHeads, _headDim)
                   .transpose(1, 2);

        Tensor q = Reshape(_queryProj);   // [B, H, T, D]
        Tensor k = Reshape(_keyProj);
        Tensor v = Reshape(_valueProj);

        // Scaled dot-product  →  [B, H, T, T]
        Tensor attn = matmul(q, k.transpose(-2, -1)) * _scale;

        if (mask is not null)
            attn = attn + mask.unsqueeze(0).unsqueeze(0);  // broadcast B, H

        attn = softmax(attn, dim: -1);

        // Weighted sum of values  →  [B, T, C]
        Tensor output = matmul(attn, v)   // [B, H, T, D]
                            .transpose(1, 2)              // [B, T, H, D]
                            .contiguous()
                            .view(B, T, C);

        return _outProj.forward(output);
    }
}
