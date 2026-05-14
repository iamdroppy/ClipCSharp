using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ClipCSharp.Models;

/// <summary>
/// CLIP model: two encoders sharing a learnable temperature parameter.
///
/// During training:
///   logits = image_features · text_features.T  ×  exp(logit_scale)
///   loss   = symmetric cross-entropy over (image, text) pairs in the batch
///
/// At inference:
///   - Encode all candidate text prompts once.
///   - For each image compute cosine similarities and softmax → class probs.
/// </summary>
public sealed class ClipModel : Module<Tensor, Tensor, (Tensor imageFeatures, Tensor textFeatures, Tensor logitScale)>
{
    public readonly VisionEncoder Visual;
    public readonly TextEncoder   Textual;

    // Learnable temperature (log scale keeps it positive)
    public readonly Parameter LogitScale;

    public ClipModel(ClipConfig cfg) : base(nameof(ClipModel))
    {
        Visual     = new VisionEncoder(cfg);
        Textual    = new TextEncoder(cfg);
        LogitScale = Parameter(tensor(cfg.InitLogitScale));

        RegisterComponents();
    }

    /// <summary>Forward pass used during training.</summary>
    /// <param name="images"> [B, 3, H, W] float images (normalised).</param>
    /// <param name="tokens"> [B, T] int64 token ids.</param>
    /// <returns>
    ///   imageFeatures [B, EmbedDim] — L2 normalised.<br/>
    ///   textFeatures  [B, EmbedDim] — L2 normalised.<br/>
    ///   logitScale    scalar        — exp(log_scale), clamped to [1, 100].
    /// </returns>
    public override (Tensor, Tensor, Tensor) forward(Tensor images, Tensor tokens)
    {
        Tensor imgFeats = Visual.forward(images);
        Tensor txtFeats = Textual.forward(tokens);

        // Clamp temperature to a stable range (prevents divergence)
        Tensor scale = LogitScale.clamp(0f, (float)Math.Log(100.0)).exp();

        return (imgFeats, txtFeats, scale);
    }

    /// <summary>
    /// Compute image-to-text and text-to-image similarity logits.
    /// logits[i, j] = scale × img_feats[i] · txt_feats[j]
    /// </summary>
    public (Tensor imageLogits, Tensor textLogits) ComputeLogits(
        Tensor imageFeatures, Tensor textFeatures, Tensor logitScale)
    {
        Tensor img2txt = logitScale * matmul(imageFeatures, textFeatures.t()); // [B, B]
        Tensor txt2img = img2txt.t();
        return (img2txt, txt2img);
    }

    // -----------------------------------------------------------------------
    // Convenience wrappers for inference (no gradient tracking)
    // -----------------------------------------------------------------------

    /// <summary>Encode images without tracking gradients.</summary>
    public Tensor EncodeImage(Tensor images)
    {
        using var _ = no_grad();
        return Visual.forward(images);
    }

    /// <summary>Encode text tokens without tracking gradients.</summary>
    public Tensor EncodeText(Tensor tokens)
    {
        using var _ = no_grad();
        return Textual.forward(tokens);
    }
}
