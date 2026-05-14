using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ClipCSharp.Training;

/// <summary>
/// Symmetric contrastive loss (InfoNCE) used to train CLIP.
///
/// For a batch of B (image, text) pairs the ground-truth target is the
/// identity: image i should be closest to text i and vice-versa.
///
/// Loss = 0.5 × (CE(img→txt logits, labels) + CE(txt→img logits, labels))
/// </summary>
public static class ContrastiveLoss
{
    /// <summary>
    /// Compute the symmetric InfoNCE loss.
    /// </summary>
    /// <param name="imageFeatures">L2-normalised [B, D].</param>
    /// <param name="textFeatures"> L2-normalised [B, D].</param>
    /// <param name="logitScale">   Temperature scalar (exp(log_t)).</param>
    /// <returns>Scalar loss tensor.</returns>
    public static Tensor Compute(
        Tensor imageFeatures,
        Tensor textFeatures,
        Tensor logitScale)
    {
        long B = imageFeatures.shape[0];

        // Similarity matrix: logits[i,j] = scale × img_i · txt_j
        Tensor logits = logitScale * matmul(imageFeatures, textFeatures.t()); // [B, B]

        // Ground-truth: diagonal (image i pairs with text i)
        Tensor labels = arange(B, dtype: ScalarType.Int64, device: imageFeatures.device);

        // Cross-entropy in both directions and average
        Tensor lossI2T = functional.cross_entropy(logits,    labels);  // image→text
        Tensor lossT2I = functional.cross_entropy(logits.t(), labels); // text→image

        return (lossI2T + lossT2I) * 0.5f;
    }

    /// <summary>
    /// Per-pair cosine similarity scores for evaluation [B, B].
    /// </summary>
    public static Tensor CosineSimilarity(Tensor imageFeatures, Tensor textFeatures)
        => matmul(imageFeatures, textFeatures.t());

    /// <summary>
    /// Retrieval accuracy (Recall@1) in the image→text direction.
    /// </summary>
    public static float RecallAtOne(Tensor imageFeatures, Tensor textFeatures)
    {
        using var _ = no_grad();

        Tensor sim     = CosineSimilarity(imageFeatures, textFeatures);
        Tensor preds   = sim.argmax(dim: -1);                    // [B]
        Tensor targets = arange(imageFeatures.shape[0],
                                dtype: ScalarType.Int64,
                                device: imageFeatures.device);

        return preds.eq(targets).to(ScalarType.Float32).mean().item<float>();
    }
}
