using SkiaSharp;
using static TorchSharp.torch;

namespace ClipCSharp.Inference;

/// <summary>
/// Replicates the CLIP image preprocessing pipeline:
///
///   1. Resize the shorter side to <see cref="Resolution"/> (bicubic).
///   2. Centre-crop to <see cref="Resolution"/> × <see cref="Resolution"/>.
///   3. Convert to float and normalise with the ImageNet mean/std used in
///      the original CLIP release.
///
/// Input:  float tensor [C, H, W] or [B, C, H, W] with pixel values in [0, 255].
/// Output: float tensor [C, H, W] or [B, C, H, W] normalised to ≈ [−2.5, 2.5].
/// </summary>
public static class ImagePreprocessor
{
    // ImageNet-style normalisation constants (identical to the CLIP release)
    private static readonly float[] Mean = [0.48145466f, 0.4578275f,  0.40821073f];
    private static readonly float[] Std  = [0.26862954f, 0.26130258f, 0.27577711f];

    /// <summary>
    /// Preprocess a single image [C, H, W] or a batch [B, C, H, W].
    /// Pixel values must be in [0, 255].
    /// </summary>
    public static Tensor Preprocess(Tensor image, int resolution = 224)
    {
        bool batched = image.ndim == 4;
        if (!batched) image = image.unsqueeze(0);   // → [1, C, H, W]

        // 1. Scale to [0, 1]
        Tensor x = image.to(ScalarType.Float32) / 255f;

        // 2. Resize (shorter side → resolution, bicubic)
        x = ResizeShorterSide(x, resolution);

        // 3. Centre crop
        x = CentreCrop(x, resolution);

        // 4. Channel-wise normalise
        x = Normalise(x);

        return batched ? x : x.squeeze(0);
    }

    /// <summary>
    /// Load an image from disk (JPEG, PNG, …) into a float tensor [C, H, W]
    /// with pixel values in [0, 255]. Pass the result into <see cref="Preprocess"/>
    /// to get a model-ready tensor.
    /// </summary>
    public static Tensor LoadFromFile(string path, Device? device = null)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Image file not found: {Path.GetFullPath(path)}", path);

        using var decoded = SKBitmap.Decode(path)
            ?? throw new InvalidOperationException(
                $"SkiaSharp failed to decode image: {path}");

        // Normalise to RGBA8888 so the byte layout is predictable.
        using var rgba = decoded.Copy(SKColorType.Rgba8888)
            ?? throw new InvalidOperationException(
                $"Failed to convert image to RGBA8888: {path}");

        int H = rgba.Height;
        int W = rgba.Width;
        byte[] rgbaBytes = rgba.Bytes;          // length = H * W * 4 (R,G,B,A …)

        // Strip the alpha channel → [H * W * 3]
        var rgbBytes = new byte[H * W * 3];
        for (int i = 0, j = 0; i < rgbaBytes.Length; i += 4, j += 3)
        {
            rgbBytes[j + 0] = rgbaBytes[i + 0];   // R
            rgbBytes[j + 1] = rgbaBytes[i + 1];   // G
            rgbBytes[j + 2] = rgbaBytes[i + 2];   // B
        }

        // [H, W, 3] uint8  →  [3, H, W] float in [0, 255]
        Tensor t = tensor(rgbBytes)
            .reshape(H, W, 3)
            .permute(2, 0, 1)
            .to(ScalarType.Float32);

        if (device is not null) t = t.to(device);
        return t;
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static Tensor ResizeShorterSide(Tensor x, int target)
    {
        long H = x.shape[2];
        long W = x.shape[3];

        if (H == target && W == target) return x;

        long newH, newW;
        if (H < W)
        {
            newH = target;
            newW = (long)Math.Round((double)W * target / H);
        }
        else
        {
            newW = target;
            newH = (long)Math.Round((double)H * target / W);
        }

        // InterpolationMode is in TorchSharp.torch (not torch.nn)
        return nn.functional.interpolate(
            x,
            size:          new long[] { newH, newW },
            mode:          InterpolationMode.Bicubic,
            align_corners: false);
    }

    private static Tensor CentreCrop(Tensor x, int size)
    {
        long H  = x.shape[2];
        long W  = x.shape[3];
        long y0 = (H - size) / 2;
        long x0 = (W - size) / 2;

        return x.narrow(2, y0, size).narrow(3, x0, size).contiguous();
    }

    private static Tensor Normalise(Tensor x)
    {
        Tensor mean = tensor(Mean, dtype: ScalarType.Float32, device: x.device).view(1, 3, 1, 1);
        Tensor std  = tensor(Std,  dtype: ScalarType.Float32, device: x.device).view(1, 3, 1, 1);
        return (x - mean) / std;
    }
}
