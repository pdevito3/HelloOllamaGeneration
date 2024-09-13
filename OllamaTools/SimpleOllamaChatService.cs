namespace OllamaTools;

using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

public interface ISimpleOllamaChatService
{
    Task<ChatMessageContent> GetChatMessageContentAsync(ChatHistory chatHistory, PromptSettings settings,
        Kernel? kernel = null, CancellationToken cancellationToken = default);
}

public class SimpleOllamaChatService(HttpClient httpClient, string modelName) : ISimpleOllamaChatService
{
    public async Task<ChatMessageContent> GetChatMessageContentAsync(ChatHistory chatHistory,
        PromptSettings settings,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, "/api/generate");
        request.Content = PrepareChatRequestContent(chatHistory, settings);
        // httpClient.GenerateCurlInConsole(request,
        //     config =>
        //     {
        //         config.TurnOn = true;
        //         config.NeedAddDefaultHeaders = true;
        //         config.EnableCodeBeautification = false;
        //     });
        var response = await httpClient.SendAsync(request, cancellationToken);
        if (response.StatusCode == HttpStatusCode.NotFound)
        {
            var responseText = "ERROR: The configured model isn't available. Perhaps it's still downloading.";
            return new ChatMessageContent(AuthorRole.Assistant, settings.IsJson() 
                ? JsonSerializer.Serialize(responseText) 
                : responseText);
        }
            
        var responseContent = await response.Content
            .ReadFromJsonAsync<OllamaResponseStreamEntry>(OllamaJsonSettings.OllamaJsonSerializerOptions, cancellationToken);
        return new ChatMessageContent(AuthorRole.Assistant, responseContent!.Response!);
    }

    private JsonContent PrepareChatRequestContent(ChatHistory messages, PromptSettings settings)
    {
        return JsonContent.Create(new
        {
            Model = modelName,
            Prompt = settings.FormatRawPrompt != null
                ? settings.FormatRawPrompt(messages, null, false)
                : throw new InvalidOperationException("FormatRawPrompt must be set"),
            Format = settings.IsJson() ? "json" : null,
            Options = new
            {
                Temperature = Math.Round(settings?.Temperature ?? 0.5, 4),
                NumPredict = settings?.MaxTokens,
                TopP = settings?.TopP ?? 1.0
            },
            Raw = settings?.Raw,
            Stream = settings?.Stream,
            Stop = (settings?.StopSequences ?? new List<string>()).ToArray()
        }, options: OllamaJsonSettings.OllamaJsonSerializerOptions);
    }

    private class OllamaResponseStreamEntry
    {
        public bool Done { get; set; }
        public string? Response { get; set; }
    }
}
