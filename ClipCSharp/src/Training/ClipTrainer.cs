using ClipCSharp.Models;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ClipCSharp.Training;

/// <summary>
/// Minimal training loop for CLIP.
///
/// Usage:
/// <code>
///   var cfg     = ClipConfig.ViTB32();
///   var model   = new ClipModel(cfg);
///   var trainer = new ClipTrainer(model, learningRate: 5e-4f);
///   trainer.TrainEpoch(dataLoader, epochIndex: 0);
/// </code>
/// </summary>
public sealed class ClipTrainer : IDisposable
{
    private readonly ClipModel          _model;
    private readonly optim.Optimizer    _optimizer;
    private bool                        _disposed;

    public ClipTrainer(ClipModel model, float learningRate = 5e-4f, float weightDecay = 0.2f)
    {
        _model     = model;
        _model.train();

        // AdamW matches the original paper (β1=0.9, β2=0.98, ε=1e-6, WD=0.2)
        _optimizer = optim.AdamW(
            _model.parameters(),
            lr:           learningRate,
            beta1:        0.9,
            beta2:        0.98,
            eps:          1e-6,
            weight_decay: weightDecay);
    }

    // -----------------------------------------------------------------------
    // Training loop
    // -----------------------------------------------------------------------

    /// <summary>
    /// Run one epoch over the provided batches.
    /// </summary>
    /// <param name="batches">
    ///   Sequence of (images [B,3,H,W], tokens [B,T]) pairs on the correct device.
    /// </param>
    /// <param name="epochIndex">Used for console output only.</param>
    /// <returns>Average loss for the epoch.</returns>
    public float TrainEpoch(IEnumerable<(Tensor images, Tensor tokens)> batches, int epochIndex)
    {
        _model.train();

        float totalLoss  = 0f;
        int   batchCount = 0;

        foreach (var (images, tokens) in batches)
        {
            _optimizer.zero_grad();

            // Forward
            var (imgFeat, txtFeat, scale) = _model.forward(images, tokens);

            // Contrastive loss
            Tensor loss = ContrastiveLoss.Compute(imgFeat, txtFeat, scale);

            // Backward + update
            loss.backward();
            _optimizer.step();

            float lossVal = loss.item<float>();
            totalLoss  += lossVal;
            batchCount++;

            if (batchCount % 50 == 0)
                Console.WriteLine($"  [Epoch {epochIndex}] step {batchCount}  loss={lossVal:F4}  " +
                                  $"logit_scale={_model.LogitScale.item<float>():F3}");
        }

        float avgLoss = batchCount > 0 ? totalLoss / batchCount : 0f;
        Console.WriteLine($"Epoch {epochIndex} done — avg loss={avgLoss:F4}");
        return avgLoss;
    }

    // -----------------------------------------------------------------------
    // Cosine LR schedule (optional)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Set the learning rate according to a cosine decay schedule with linear
    /// warm-up, matching the original CLIP training recipe.
    /// </summary>
    public void SetLearningRate(float baseLr, int step, int warmupSteps, int totalSteps)
    {
        float lr;
        if (step < warmupSteps)
        {
            lr = baseLr * step / warmupSteps;
        }
        else
        {
            float progress = (float)(step - warmupSteps) / (totalSteps - warmupSteps);
            lr = baseLr * 0.5f * (1f + MathF.Cos(MathF.PI * progress));
        }

        foreach (var g in _optimizer.ParamGroups)
            g.LearningRate = lr;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _optimizer.Dispose();
            _disposed = true;
        }
    }
}
