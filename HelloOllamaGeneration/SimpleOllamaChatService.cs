namespace HelloOllamaGeneration;

using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

public interface ISimpleOllamaChatService
{
    Task<ChatMessageContent> GetChatMessageContentAsync(ChatHistory chatHistory, PromptSettings settings, 
        Kernel? kernel = null, CancellationToken cancellationToken = default);
}

public class SimpleOllamaChatService(HttpClient httpClient, string modelName) : ISimpleOllamaChatService
{
    private static readonly JsonSerializerOptions JsonSerializerOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public async Task<ChatMessageContent> GetChatMessageContentAsync(ChatHistory chatHistory, 
        PromptSettings settings, 
        Kernel? kernel = null, 
        CancellationToken cancellationToken = default)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, "/api/generate");
        request.Content = PrepareChatRequestContent(chatHistory, settings);
        var response = await httpClient.SendAsync(request, cancellationToken);
        if (response.StatusCode == HttpStatusCode.NotFound)
        {
            var responseText = "ERROR: The configured model isn't available. Perhaps it's still downloading.";
            return new ChatMessageContent(AuthorRole.Assistant, settings.IsJson() ? JsonSerializer.Serialize(responseText) : responseText);
        }

        var responseContent = await response.Content.ReadFromJsonAsync<OllamaResponseStreamEntry>(JsonSerializerOptions, cancellationToken);
        return new ChatMessageContent(AuthorRole.Assistant, responseContent!.Response!);
    }
    
    private JsonContent PrepareChatRequestContent(ChatHistory messages, PromptSettings settings)
    {
        return JsonContent.Create(new
        {
            Model = modelName,
            Prompt = FormatRawPrompt(messages),
            Format = settings.IsJson() ? "json" : null,
            Options = new
            {
                Temperature = settings?.Temperature ?? 0.5,
                NumPredict = settings?.MaxTokens,
                TopP = settings?.TopP ?? 1.0
            },
            Raw = settings?.Raw,
            Stream = settings?.Stream,
        }, options: JsonSerializerOptions);
    }

    private static string FormatRawPrompt(ChatHistory messages)
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
    
    private static string FormatRawPrompt(ChatHistory messages, Kernel? kernel, bool autoInvokeFunctions)
    {
        // TODO: First fetch the prompt template for the model via /api/show, and then use
        // that to format the messages. Currently this is hardcoded to the Mistral prompt,
        // i.e.: [INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]
        var sb = new StringBuilder();
        var indexOfLastUserOrSystemMessage = IndexOfLast(messages, m => m.Role == AuthorRole.User || m.Role == AuthorRole.System);

        // IMPORTANT: The whitespace in the prompt is significant. Do not add or remove extra spaces/linebreaks,
        // as this affects tokenization. Mistral's function calling is useless unless you get this exactly right.

        for (var index = 0; index < messages.Count; index++)
        {
            var message = messages[index];

            // Emit tools descriptor immediately before the final [INST]
            if (index == indexOfLastUserOrSystemMessage && autoInvokeFunctions && kernel is not null)
            {
                var tools = kernel.Plugins.SelectMany(p => p.GetFunctionsMetadata()).ToArray() ?? [];
                if (tools is { Length: > 0 })
                {
                    sb.Append("[AVAILABLE_TOOLS] ");
                    sb.Append(JsonSerializer.Serialize(tools.Select(OllamaChatFunction.Create), JsonSerializerOptions));
                    sb.Append("[/AVAILABLE_TOOLS]");
                }
            }

            if (message.Role == AuthorRole.User || message.Role == AuthorRole.System)
            {
                sb.Append("[INST] ");
                sb.Append(message.Content);
                sb.Append(" [/INST]");
            }
            else if (message.Role == AuthorRole.Tool)
            {
                sb.Append("[TOOL_CALLS] ");
                sb.Append(message.Content);
                sb.Append(" [/TOOL_CALLS]\n\n");
            }
            else
            {
                if (!string.IsNullOrWhiteSpace(message.Content))
                {
                    sb.Append(message.Content);
                    sb.Append("</s> "); // That's right, there's no matching <s>. See https://discuss.huggingface.co/t/skew-between-mistral-prompt-in-docs-vs-chat-template/66674/2
                }
            }
        }

        return sb.ToString();
    }

    private static int IndexOfLast<T>(IReadOnlyList<T> messages, Func<T, bool> value)
    {
        for (var i = messages.Count - 1; i >= 0; i--)
        {
            if (value(messages[i]))
            {
                return i;
            }
        }

        return -1;
    }

    private class OllamaResponseStreamEntry
    {
        public bool Done { get; set; }
        public string? Response { get; set; }
    }
}
