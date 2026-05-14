using ClipCSharp.Models;
using ClipCSharp.Tokenizer;
using static TorchSharp.torch;

namespace ClipCSharp.Inference;

/// <summary>
/// Zero-shot image classifier using CLIP.
///
/// Given a set of candidate class names and an optional prompt template,
/// this classifier encodes each class label as a text embedding and measures
/// cosine similarity against image embeddings.
///
/// Example (ImageNet zero-shot):
/// <code>
///   var clf = new ZeroShotClassifier(model, tokenizer);
///   clf.SetClasses(imageNetClasses, template: "a photo of a {0}.");
///   int[] predictions = clf.Classify(imageBatch);
/// </code>
/// </summary>
public sealed class ZeroShotClassifier
{
    private readonly ClipModel    _model;
    private readonly BpeTokenizer _tokenizer;
    private readonly int          _contextLength;

    // Pre-computed and cached text embeddings for the current class set
    private Tensor?  _classEmbeddings;   // [NumClasses, EmbedDim]
    private string[] _classNames = [];

    public ZeroShotClassifier(ClipModel model, BpeTokenizer tokenizer, int contextLength = 77)
    {
        _model         = model;
        _tokenizer     = tokenizer;
        _contextLength = contextLength;
    }

    // -----------------------------------------------------------------------
    // Setup
    // -----------------------------------------------------------------------

    /// <summary>
    /// Pre-compute and cache text embeddings for all class labels.
    /// Call once before repeated calls to <see cref="Classify"/> or
    /// <see cref="Probabilities"/>.
    /// </summary>
    /// <param name="classNames">List of human-readable class names.</param>
    /// <param name="template">
    ///   Format string with one placeholder; defaults to "a photo of a {0}.".
    /// </param>
    public void SetClasses(IReadOnlyList<string> classNames, string template = "a photo of a {0}.")
    {
        _classNames      = classNames.ToArray();
        _classEmbeddings = EncodeLabels(classNames, template);
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /// <summary>
    /// Classify a batch of images.
    /// </summary>
    /// <param name="images">[B, 3, H, W] float tensor (preprocessed).</param>
    /// <returns>Predicted class index for each image in the batch.</returns>
    public int[] Classify(Tensor images)
    {
        Tensor probs = Probabilities(images);         // [B, NumClasses]
        Tensor preds = probs.argmax(dim: -1);         // [B]

        return Enumerable.Range(0, (int)preds.shape[0])
                         .Select(i => (int)preds[i].item<long>())
                         .ToArray();
    }

    /// <summary>
    /// Compute softmax probability over classes for each image.
    /// </summary>
    /// <param name="images">[B, 3, H, W]</param>
    /// <returns>[B, NumClasses] probability tensor.</returns>
    public Tensor Probabilities(Tensor images)
    {
        EnsureClassesSet();

        using var _ = no_grad();

        Tensor imgFeat = _model.EncodeImage(images);  // [B, D]
        Tensor scale   = _model.LogitScale.clamp(0f, (float)Math.Log(100.0)).exp();

        // Cosine similarity → [B, NumClasses]
        Tensor logits = scale * matmul(imgFeat, _classEmbeddings!.t());

        return nn.functional.softmax(logits, dim: -1);
    }

    /// <summary>
    /// Returns the top-k class names and their probabilities for a single image.
    /// </summary>
    public IReadOnlyList<(string className, float probability)> TopK(Tensor image, int k = 5)
    {
        if (image.ndim == 3)
            image = image.unsqueeze(0);               // [1, 3, H, W]

        Tensor probs = Probabilities(image).squeeze(0); // [NumClasses]

        var (topValues, topIndices) = probs.topk(k);   // Item1=values, Item2=indices

        return Enumerable.Range(0, k)
            .Select(i => (_classNames[(int)topIndices[i].item<long>()],
                          topValues[i].item<float>()))
            .ToList();
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    private Tensor EncodeLabels(IReadOnlyList<string> classNames, string template)
    {
        var allEmbeddings = new List<Tensor>(classNames.Count);

        using var _ = no_grad();

        // Process in small batches to avoid OOM on large class sets
        const int batchSize = 64;
        for (int start = 0; start < classNames.Count; start += batchSize)
        {
            int end   = Math.Min(start + batchSize, classNames.Count);
            var batch = classNames.Skip(start).Take(end - start).ToList();

            var tokenArrays = batch
                .Select(name => _tokenizer.TokenizeAndPad(
                    string.Format(template, name), _contextLength))
                .ToArray();

            // Stack → [BatchSize, ContextLength]
            Tensor tokens = stack(tokenArrays.Select(a =>
                tensor(a, dtype: ScalarType.Int64)).ToArray());

            Tensor emb = _model.EncodeText(tokens);   // [BatchSize, D]
            allEmbeddings.Add(emb);
        }

        // Concatenate → [NumClasses, D]
        Tensor classEmb = cat(allEmbeddings.ToArray(), dim: 0);
        return L2Normalize(classEmb);
    }

    // Manual L2 normalisation
    private static Tensor L2Normalize(Tensor x, int dim = -1)
    {
        Tensor nrm = x.norm(dim, keepdim: true, p: 2f);
        return x / nrm.clamp(min: 1e-12f);
    }

    private void EnsureClassesSet()
    {
        if (_classEmbeddings is null)
            throw new InvalidOperationException(
                "Call SetClasses() before running inference.");
    }
}
