namespace HelloOllamaGeneration;

public enum ResponseFormat { Json, Text }
public record PromptSettings
{
    public string? ModelId { get; set; }
    public ResponseFormat ResponseFormat { get; set; }
    public double Temperature { get; set; } = 0.5;
    public int? NumPredict { get; set; }
    public double TopP { get; set; } = 1.0;
    public bool Stream { get; set; }
    public bool Raw { get; set; } = true;
    public int? MaxTokens { get; set; }
        
    public bool IsJson() => ResponseFormat == ResponseFormat.Json;
}