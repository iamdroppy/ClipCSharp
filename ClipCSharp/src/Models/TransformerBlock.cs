using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ClipCSharp.Models;

/// <summary>
/// A single pre-LayerNorm transformer block used by both the text and vision
/// encoders.  Architecture: LN → MHA → residual → LN → FFN(4×) → residual.
/// </summary>
public sealed class TransformerBlock : Module<Tensor, Tensor?, Tensor>
{
    private readonly LayerNorm         _ln1;
    private readonly MultiHeadAttention _attn;
    private readonly LayerNorm         _ln2;
    private readonly Linear            _fc1;
    private readonly Linear            _fc2;

    public TransformerBlock(string name, int embedDim, int numHeads) : base(name)
    {
        _ln1  = LayerNorm(embedDim);
        _attn = new MultiHeadAttention($"{name}.attn", embedDim, numHeads);
        _ln2  = LayerNorm(embedDim);

        // FFN: expand 4× then contract
        int ffnDim = embedDim * 4;
        _fc1 = Linear(embedDim, ffnDim);
        _fc2 = Linear(ffnDim,   embedDim);

        RegisterComponents();
    }

    /// <param name="x">    [B, T, C]</param>
    /// <param name="mask"> optional causal mask [T, T]; null = full attention</param>
    /// <returns>[B, T, C]</returns>
    public override Tensor forward(Tensor x, Tensor? mask)
    {
        // Self-attention with pre-LN and residual
        x = x + _attn.forward(_ln1.forward(x), mask);

        // FFN with pre-LN and residual
        Tensor h = _ln2.forward(x);
        h = _fc1.forward(h);
        h = functional.gelu(h);          // GELU matches the original paper
        h = _fc2.forward(h);
        x = x + h;

        return x;
    }
}
