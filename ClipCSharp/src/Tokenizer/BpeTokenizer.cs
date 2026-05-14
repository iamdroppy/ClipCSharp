using System.Text;
using System.Text.RegularExpressions;

namespace ClipCSharp.Tokenizer;

/// <summary>
/// Byte Pair Encoding tokenizer matching OpenAI's CLIP tokenizer.
/// Implements the GPT-2 / CLIP BPE scheme with byte-level fallback.
/// </summary>
public sealed class BpeTokenizer
{
    // Special tokens
    public const string SotToken = "<|startoftext|>";
    public const string EotToken = "<|endoftext|>";

    private readonly Dictionary<string, int>        _encoder;       // token string → id
    private readonly Dictionary<int, string>        _decoder;       // id → token string
    private readonly Dictionary<(string, string), int> _bpeMerges;  // merge rank table
    private readonly Dictionary<string, string>     _cache;

    // Regex that splits text into candidate tokens exactly as CLIP/GPT-2 does
    private static readonly Regex _pattern = new(
        @"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    // Byte → unicode character mapping (GPT-2 style, avoids control chars)
    private static readonly Dictionary<int, char>  _byteToUnicode;
    private static readonly Dictionary<char, int>  _unicodeToByte;

    static BpeTokenizer()
    {
        // Build the same byte→unicode table as the Python reference
        _byteToUnicode = BuildByteToUnicode();
        _unicodeToByte = _byteToUnicode.ToDictionary(kv => kv.Value, kv => kv.Key);
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    public BpeTokenizer(Dictionary<string, int> encoder, IEnumerable<(string, string)> merges)
    {
        _encoder   = encoder;
        _decoder   = encoder.ToDictionary(kv => kv.Value, kv => kv.Key);
        _bpeMerges = merges.Select((m, i) => (m, i))
                           .ToDictionary(x => x.m, x => x.i);
        _cache     = new Dictionary<string, string>();

        // Add special tokens if not present
        foreach (var tok in new[] { SotToken, EotToken })
            if (!_encoder.ContainsKey(tok))
            {
                int id = _encoder.Count;
                _encoder[tok] = id;
                _decoder[id]  = tok;
            }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    public int SotId => _encoder[SotToken];
    public int EotId => _encoder[EotToken];
    public int VocabSize => _encoder.Count;

    /// <summary>Encode a text string into a list of token ids.</summary>
    public List<int> Encode(string text)
    {
        var ids = new List<int> { SotId };

        foreach (Match match in _pattern.Matches(text.ToLowerInvariant()))
        {
            // Map each UTF-8 byte to the corresponding Unicode character
            var word = string.Concat(
                Encoding.UTF8.GetBytes(match.Value).Select(b => _byteToUnicode[b]));

            ids.AddRange(BpeSegment(word).Split(' ').Select(t => _encoder[t]));
        }

        ids.Add(EotId);
        return ids;
    }

    /// <summary>Decode a list of token ids back to a UTF-8 string.</summary>
    public string Decode(IEnumerable<int> ids)
    {
        var text  = string.Concat(ids.Select(id => _decoder[id]));
        var bytes = text.Select(c => (byte)_unicodeToByte[c]).ToArray();
        return Encoding.UTF8.GetString(bytes);
    }

    /// <summary>
    /// Tokenize and pad/truncate to <paramref name="contextLength"/> tokens,
    /// returning an int array ready to feed into the text encoder.
    /// </summary>
    public int[] TokenizeAndPad(string text, int contextLength = 77)
    {
        var tokens = Encode(text);
        var result = new int[contextLength];

        // Truncate, but always preserve SOT and EOT
        if (tokens.Count > contextLength)
        {
            tokens = tokens.Take(contextLength - 1).ToList();
            tokens.Add(EotId);
        }

        for (int i = 0; i < tokens.Count && i < contextLength; i++)
            result[i] = tokens[i];

        return result;
    }

    // -----------------------------------------------------------------------
    // BPE core
    // -----------------------------------------------------------------------

    private string BpeSegment(string word)
    {
        if (_cache.TryGetValue(word, out var cached))
            return cached;

        var symbols = word.Select(c => c.ToString()).ToList();
        ApplyMerges(symbols);

        var result = string.Join(" ", symbols);
        _cache[word] = result;
        return result;
    }

    private void ApplyMerges(List<string> symbols)
    {
        while (symbols.Count > 1)
        {
            // Find the highest-priority (lowest rank) adjacent pair
            int   bestRank = int.MaxValue;
            int   bestIdx  = -1;

            for (int i = 0; i < symbols.Count - 1; i++)
            {
                var pair = (symbols[i], symbols[i + 1]);
                if (_bpeMerges.TryGetValue(pair, out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIdx  = i;
                }
            }

            if (bestIdx == -1) break; // no more merges available

            string merged = symbols[bestIdx] + symbols[bestIdx + 1];
            symbols[bestIdx] = merged;
            symbols.RemoveAt(bestIdx + 1);
        }
    }

    // -----------------------------------------------------------------------
    // Byte ↔ Unicode table (identical to the Python reference)
    // -----------------------------------------------------------------------

    private static Dictionary<int, char> BuildByteToUnicode()
    {
        // Printable ASCII + latin supplement, then remap control / space bytes
        var bs = Enumerable.Range('!', '~' - '!' + 1)
                           .Concat(Enumerable.Range('¡', '¬' - '¡' + 1))
                           .Concat(Enumerable.Range('®', 'ÿ' - '®' + 1))
                           .ToList();

        var cs = new List<int>(bs);
        int n  = 0;
        for (int b = 0; b < 256; b++)
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n++);
            }

        return bs.Zip(cs, (b, c) => (b, (char)c))
                 .ToDictionary(x => x.b, x => x.Item2);
    }

    // -----------------------------------------------------------------------
    // Factory: build a minimal vocab + merges for demonstration purposes.
    // In production, load vocab.json + merges.txt from the OpenAI release.
    // -----------------------------------------------------------------------

    /// <summary>
    /// Creates a tokenizer pre-loaded with a built-in character-level
    /// vocabulary (every printable ASCII byte maps to itself).  Suitable for
    /// unit-tests and demos; replace with the real CLIP vocab for production.
    /// </summary>
    /// <remarks>
    /// Vocab id ordering follows OpenAI's CLIP convention:
    ///   ids 0..N-1   →   one per byte (regular tokens)
    ///   id  N        →   SOT (start of text)
    ///   id  N+1      →   EOT (end of text)
    /// Putting the special tokens at the TOP of the id range is what makes
    /// the `tokens.argmax(dim:-1)` trick in <see cref="ClipCSharp.Models.TextEncoder"/>
    /// recover the EOT position. If EOT had a low id (e.g. 1) argmax would
    /// instead point at the highest-id *regular* character — typically the
    /// space byte, since it remaps to a high Unicode codepoint — and every
    /// short prompt that shares the same first few characters would produce
    /// an identical text embedding.
    /// </remarks>
    public static BpeTokenizer CreateMinimal()
    {
        var encoder = new Dictionary<string, int>();

        // One token per byte. Ids start at 0; the BpeTokenizer constructor
        // appends SOT and EOT at the end of the vocab so EOT has the highest
        // id (required by TextEncoder's argmax-based EOT location).
        var byteMap = BuildByteToUnicode();
        int idx = 0;
        foreach (var ch in byteMap.Values)
            encoder[ch.ToString()] = idx++;

        // No merges in the minimal tokenizer (pure character-level)
        return new BpeTokenizer(encoder, Enumerable.Empty<(string, string)>());
    }
}
