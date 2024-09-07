namespace HelloOllamaGeneration.Generators;

using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;

public abstract class SimpleGeneratorBase<T>(IServiceProvider services)
{
    protected abstract string DirectoryName { get; }

    protected abstract object GetId(T item);

    private static string OutputDirRoot => Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "output");

    protected string OutputDirPath => Path.Combine(OutputDirRoot, DirectoryName);

    private readonly JsonSerializerOptions SerializerOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true
    };

    public async Task<IReadOnlyList<T>> GenerateAsync()
    {
        if (!Directory.Exists(OutputDirPath))
        {
            Directory.CreateDirectory(OutputDirPath);
        }

        var sw = Stopwatch.StartNew();
        await foreach (var item in GenerateCoreAsync())
        {
            sw.Stop();
            Console.WriteLine($"Writing {item!.GetType().Name} {GetId(item)} [generated in {sw.Elapsed.TotalSeconds}s]");
            var path = GetItemOutputPath(GetId(item).ToString()!);
            await WriteAsync(path, item);
            sw.Restart();
        }

        var existingFiles = Directory.GetFiles(OutputDirPath);
        return existingFiles.Select(Read).ToList();
    }

    protected string GetItemOutputPath(string id)
        => Path.Combine(OutputDirPath, $"{id}{FilenameExtension}");

    protected abstract IAsyncEnumerable<T> GenerateCoreAsync();

    protected ISimpleOllamaChatService ChatCompletionService { get; } = services.GetRequiredService<ISimpleOllamaChatService>();

    protected async Task<TResponse> GetAndParseJsonChatCompletion<TResponse>(string prompt, int? maxTokens = null, object? tools = null)
    {
        var executionSettings = new PromptSettings
        {
            MaxTokens = maxTokens,
            Temperature = 0.9f,
            ResponseFormat = ResponseFormat.Json,
            FormatRawPrompt = (messages, k, autoInvoke) 
                => FormatMistralPromptWithFunctions(messages, k, autoInvoke)
        };

        var kernel = (Kernel?)null;
        if (tools is not null)
        {
            kernel = new();
            kernel.Plugins.AddFromObject(tools);
        }

        var chatHistory = new ChatHistory();
        chatHistory.AddUserMessage(prompt);

        return await RunWithRetries(async () =>
        {
            var response = await RunWithRetries(() =>
                ChatCompletionService.GetChatMessageContentAsync(chatHistory, executionSettings, kernel));
            var responseString = response.ToString();

            // using a retry because sometimes it doesn't return complete json
            var parsed = ReadAndDeserializeChatResponse<TResponse>(responseString, SerializerOptions)!;

            return parsed;
        }, 1);
    }

    private static async Task<TResult> RunWithRetries<TResult>(Func<Task<TResult>> operation, int backoffIncrement = 15)
    {
        var delay = TimeSpan.FromSeconds(5);
        var maxAttempts = 5;
        for (var attemptIndex = 1; ; attemptIndex++)
        {
            try
            {
                return await operation();
            }
            catch (Exception e) when (attemptIndex < maxAttempts)
            {
                Console.WriteLine($"Exception on attempt {attemptIndex}: {e.Message}. Will retry after {delay}");
                await Task.Delay(delay);
                delay += TimeSpan.FromSeconds(backoffIncrement);
            }
        }
    }

    private static TResponse? ReadAndDeserializeChatResponse<TResponse>(string json, JsonSerializerOptions options)
    {
        try
        {
            var reader = new Utf8JsonReader(Encoding.UTF8.GetBytes(json).AsSpan());
            return JsonSerializer.Deserialize<TResponse>(ref reader, options);
        }
        catch (Exception ex)
        {
            // Console.WriteLine($"Error deserializing JSON {json}: {ex.Message}");
            Console.WriteLine("Error deserializing JSON");
            throw ex;
        }
    }

    protected virtual string FilenameExtension => ".json";

    protected virtual Task WriteAsync(string path, T item)
    {
        var itemJson = JsonSerializer.Serialize(item, SerializerOptions);
        return File.WriteAllTextAsync(path, itemJson);
    }

    protected virtual T Read(string path)
    {
        try
        {
            using var existingJson = File.OpenRead(path);
            return JsonSerializer.Deserialize<T>(existingJson, SerializerOptions)!;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error reading {path}: {ex.Message}");
            throw;
        }
    }

    protected IAsyncEnumerable<V> MapParallel<U, V>(IEnumerable<U> source, Func<U, Task<V>> map)
    {
        var outputs = Channel.CreateUnbounded<V>();
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = 5 };
        var mapTask = Parallel.ForEachAsync(source, parallelOptions, async (sourceItem, ct) =>
        {
            try
            {
                var mappedItem = await map(sourceItem);
                await outputs.Writer.WriteAsync(mappedItem, ct);
            }
            catch (Exception ex)
            {
                outputs.Writer.TryComplete(ex);
            }
        });

        mapTask.ContinueWith(_ => outputs.Writer.TryComplete());

        return outputs.Reader.ReadAllAsync();
    }

    private static string FormatSimpleMistralPrompt(ChatHistory messages, Kernel? kernel, bool autoInvokeFunctions)
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

    private static string FormatMistralPromptWithFunctions(ChatHistory messages, Kernel? kernel, bool autoInvokeFunctions)
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
                    sb.Append(JsonSerializer.Serialize(tools.Select(OllamaChatFunction.Create), OllamaJsonSettings.OllamaJsonSerializerOptions));
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
                if (string.IsNullOrWhiteSpace(message.Content)) continue;
                
                sb.Append(message.Content);
                sb.Append("</s> "); // That's right, there's no matching <s>. See https://discuss.huggingface.co/t/skew-between-mistral-prompt-in-docs-vs-chat-template/66674/2
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
}