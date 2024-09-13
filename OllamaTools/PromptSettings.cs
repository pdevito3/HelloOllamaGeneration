namespace OllamaTools;

using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

public enum ResponseFormat { Json, Text }
public record PromptSettings
{
    public string? ModelId { get; set; }
    public ResponseFormat ResponseFormat { get; set; } = ResponseFormat.Text;
    public double Temperature { get; set; } = 0.5;
    public double TopP { get; set; } = 1.0;
    public bool Stream { get; set; }
    public bool Raw { get; set; } = true;
    public int? MaxTokens { get; set; }

    public bool IsJson() => ResponseFormat == ResponseFormat.Json;

    public Func<ChatHistory, Kernel?, bool, string>? FormatRawPrompt { get; set; }
    public IEnumerable<string> StopSequences { get; set; } = new List<string>();
}

public static class OllamaJsonSettings
{
    public static readonly JsonSerializerOptions OllamaJsonSerializerOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };
}
