namespace ClipCSharp.Models;

/// <summary>
/// Hyper-parameters for the CLIP model.
/// Defaults match the ViT-B/32 variant from the OpenAI paper.
/// </summary>
public sealed record ClipConfig
{
    // ---- Text encoder -------------------------------------------------------
    /// <summary>Vocabulary size (50_257 in the OpenAI CLIP release).</summary>
    public int VocabSize      { get; init; } = 49_408;
    /// <summary>Maximum sequence length (always 77 in CLIP).</summary>
    public int ContextLength  { get; init; } = 77;
    /// <summary>Width of the text transformer (embedding dim).</summary>
    public int TextWidth      { get; init; } = 512;
    /// <summary>Number of transformer layers in the text encoder.</summary>
    public int TextLayers     { get; init; } = 12;
    /// <summary>Number of attention heads in the text transformer.</summary>
    public int TextHeads      { get; init; } = 8;

    // ---- Vision encoder (ViT) -----------------------------------------------
    /// <summary>Input image resolution (pixels, assumed square).</summary>
    public int ImageSize      { get; init; } = 224;
    /// <summary>Patch size in pixels (32 for ViT-B/32, 16 for ViT-B/16).</summary>
    public int PatchSize      { get; init; } = 32;
    /// <summary>Width of the vision transformer.</summary>
    public int VisionWidth    { get; init; } = 768;
    /// <summary>Number of transformer layers in the vision encoder.</summary>
    public int VisionLayers   { get; init; } = 12;
    /// <summary>Number of attention heads in the vision transformer.</summary>
    public int VisionHeads    { get; init; } = 12;

    // ---- Joint projection ---------------------------------------------------
    /// <summary>Dimensionality of the shared embedding space.</summary>
    public int EmbedDim       { get; init; } = 512;

    // ---- Training -----------------------------------------------------------
    /// <summary>
    /// Log of the initial temperature τ (learnable).
    /// The paper initialises τ = 0.07, so log(τ) ≈ −2.659.
    /// </summary>
    public float InitLogitScale { get; init; } = (float)Math.Log(1.0 / 0.07);

    // ---- Convenience factories ----------------------------------------------

    /// <summary>ViT-B/32 – the standard CLIP model.</summary>
    public static ClipConfig ViTB32() => new();

    /// <summary>ViT-B/16 – higher resolution patches.</summary>
    public static ClipConfig ViTB16() => new()
    {
        PatchSize    = 16,
        VisionWidth  = 768,
        VisionLayers = 12,
        VisionHeads  = 12,
    };

    /// <summary>ViT-L/14 – large model.</summary>
    public static ClipConfig ViTL14() => new()
    {
        PatchSize    = 14,
        VisionWidth  = 1024,
        VisionLayers = 24,
        VisionHeads  = 16,
        TextWidth    = 768,
        TextLayers   = 12,
        TextHeads    = 12,
        EmbedDim     = 768,
    };

    // Derived property
    public int GridSize        => ImageSize / PatchSize;
    public int NumPatches      => GridSize * GridSize;
    public int SeqLenVision    => NumPatches + 1; // +1 for [CLS]
}
