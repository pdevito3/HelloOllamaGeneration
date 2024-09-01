namespace HelloOllamaGeneration;

using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

public interface ISimpleOllamaChatService
{
    Task<ChatMessageContent> GetChatMessageContentAsync(ChatHistory chatHistory, SimpleOllamaChatService.PromptSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default);
}

public class SimpleOllamaChatService(HttpClient httpClient, string modelName) : ISimpleOllamaChatService
{
    private static readonly JsonSerializerOptions JsonSerializerOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public async Task<ChatMessageContent> GetChatMessageContentAsync(ChatHistory chatHistory, PromptSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, "/api/generate");
        request.Content = PrepareChatRequestContent(chatHistory, executionSettings, false, out var json);
        var response = await httpClient.SendAsync(request, cancellationToken);
        if (response.StatusCode == HttpStatusCode.NotFound)
        {
            var responseText = "ERROR: The configured model isn't available. Perhaps it's still downloading.";
            return new ChatMessageContent(AuthorRole.Assistant, json ? JsonSerializer.Serialize(responseText) : responseText);
        }

        var responseContent = await response.Content.ReadFromJsonAsync<OllamaResponseStreamEntry>(JsonSerializerOptions, cancellationToken);
        return new ChatMessageContent(AuthorRole.Assistant, responseContent!.Response!);
    }
    
    private JsonContent PrepareChatRequestContent(ChatHistory messages, PromptSettings? options, bool streaming, out bool json)
    {
        json = options is { ResponseFormat: "json_object" };
        return JsonContent.Create(new
        {
            Model = modelName,
            Prompt = FormatPrompt(messages),
            Format = json ? "json" : null,
            Options = new
            {
                Temperature = options?.Temperature ?? 0.5,
                NumPredict = options?.MaxTokens,
                TopP = options?.TopP ?? 1.0
            },
            Raw = true,
            Stream = streaming,
        }, options: JsonSerializerOptions);
    }

    private static string FormatPrompt(ChatHistory messages)
    {
        var sb = new StringBuilder();

        foreach (var message in messages)
        {
            if (message.Role == AuthorRole.User || message.Role == AuthorRole.System)
            {
                sb.Append("[INST] ").Append(message.Content).Append(" [/INST]");
            }
            else if (message.Role == AuthorRole.Tool)
            {
                sb.Append("[TOOL_CALLS] ").Append(message.Content).Append(" [/TOOL_CALLS]\n\n");
            }
            else
            {
                sb.Append("</s> "); // That's right, there's no matching <s>. See https://discuss.huggingface.co/t/skew-between-mistral-prompt-in-docs-vs-chat-template/66674/2
            }
        }

        return sb.ToString();
    }

    private class OllamaResponseStreamEntry
    {
        public bool Done { get; set; }
        public string? Response { get; set; }
    }
    
    public record PromptSettings
    {
        public string? ModelId { get; set; }
        public string? ServiceId { get; set; }
        public string? ResponseFormat { get; set; }
        public double? Temperature { get; set; }
        public int? NumPredict { get; set; }
        public double? TopP { get; set; }
        public bool? Stream { get; set; }
        public int? MaxTokens { get; set; }
    }
}
