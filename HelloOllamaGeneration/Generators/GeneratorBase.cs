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

public abstract class GeneratorBase<T>(IServiceProvider services)
{
    protected abstract string DirectoryName { get; }

    protected abstract object GetId(T item);

    private static string OutputDirRoot => Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "output");

    protected string OutputDirPath => Path.Combine(OutputDirRoot, DirectoryName);

    private readonly JsonSerializerOptions SerializerOptions = new JsonSerializerOptions(JsonSerializerDefaults.Web)
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

    protected IChatCompletionService ChatCompletionService { get; } = services.GetRequiredService<IChatCompletionService>();

    protected async Task<string> GetChatCompletion(string prompt)
    {
        // Instructing it to end the content with END_OF_CONTENT is beneficial because it often tries to add a suffix like
        // "I have done the task, hope this helps!". We can avoid that by making it stop before that.
        var executionSettings = new OpenAIPromptExecutionSettings { Temperature = 0.9f, StopSequences = ["END_OF_CONTENT"] };
        var chatHistory = new ChatHistory();
        chatHistory.AddUserMessage(prompt);
        var response = await ChatCompletionService.GetChatMessageContentAsync(chatHistory, executionSettings);
        return response.ToString();
    }

    // steve's original
    // [Experimental("SKEXP0010")]
    // protected async Task<TResponse> GetAndParseJsonChatCompletion<TResponse>(string prompt, int? maxTokens = null, object? tools = null)
    // {
    //     var executionSettings = new OpenAIPromptExecutionSettings
    //     {
    //         MaxTokens = maxTokens,
    //         Temperature = 0.9f,
    //         ResponseFormat = "json_object",
    //         ToolCallBehavior = tools is null ? default : ToolCallBehavior.AutoInvokeKernelFunctions
    //     };
    //     
    //     var kernel = (Kernel?)null;
    //     if (tools is not null)
    //     {
    //         kernel = new();
    //         kernel.Plugins.AddFromObject(tools);
    //     }
    //
    //     var chatHistory = new ChatHistory();
    //     chatHistory.AddUserMessage(prompt);
    //     
    //     var response = await RunWithRetries(() => ChatCompletionService.GetChatMessageContentAsync(chatHistory, executionSettings, kernel));
    //     var responseString = response.ToString();
    //     
    //     // Due to what seems like a server-side bug, when asking for a json_object response and with tools enabled,
    //     // it often replies with two or more JSON objects concatenated together (duplicates or slight variations).
    //     // As a workaround, just read the first complete JSON object from the response.
    //     var parsed = ReadAndDeserializeSingleValue<TResponse>(responseString, SerializerOptions)!;
    //     
    //     return parsed;
    // }
    
    [Experimental("SKEXP0010")]
    protected async Task<TResponse> GetAndParseJsonChatCompletion<TResponse>(string prompt, int? maxTokens = null, object? tools = null)
    {
        var executionSettings = new OpenAIPromptExecutionSettings
        {
            MaxTokens = maxTokens,
            Temperature = 0.9f,
            ResponseFormat = "json_object",
            ToolCallBehavior = tools is null ? default : ToolCallBehavior.AutoInvokeKernelFunctions
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

            // Due to what seems like a server-side bug, when asking for a json_object response and with tools enabled,
            // it often replies with two or more JSON objects concatenated together (duplicates or slight variations).
            // As a workaround, just read the first complete JSON object from the response.
            // confirmed ollama issue https://www.youtube.com/watch?v=1DRS5KaLe3k
            
            // the single value part doesn't seem to be in the actual OG code, so just wrapping it all in a try catch to get it closer to what it expects
            var parsed = ReadAndDeserializeSingleValue<TResponse>(responseString, SerializerOptions)!;

            return parsed;
        }, 1);
    }

    
    // ai generated version
    // [Experimental("SKEXP0010")]
    // protected async Task<TResponse> GetAndParseJsonChatCompletion<TResponse>(string prompt, int? maxTokens = null, object? tools = null)
    // {
    //     var executionSettings = new OpenAIPromptExecutionSettings
    //     {
    //         MaxTokens = maxTokens,
    //         Temperature = 0.9f,
    //         ResponseFormat = "json_object",
    //         ToolCallBehavior = tools is null ? default : ToolCallBehavior.AutoInvokeKernelFunctions
    //     };
    //
    //     var kernel = (Kernel?)null;
    //     if (tools is not null)
    //     {
    //         kernel = new();
    //         kernel.Plugins.AddFromObject(tools);
    //     }
    //
    //     var chatHistory = new ChatHistory();
    //     chatHistory.AddUserMessage(prompt);
    //
    //     var responseString = string.Empty;
    //     var maxAttempts = 10; // Max attempts to retrieve the complete response
    //     var attempt = 0;
    //
    //     while (attempt < maxAttempts)
    //     {
    //         var response = await RunWithRetries(() => ChatCompletionService.GetChatMessageContentAsync(chatHistory, executionSettings, kernel));
    //         responseString += response.ToString();
    //     
    //         if (IsJsonComplete(responseString)) // Implement this method to check if the JSON is complete
    //         {
    //             break;
    //         }
    //     
    //         attempt++;
    //     }
    //
    //     if (!IsJsonComplete(responseString))
    //     {
    //         throw new InvalidOperationException("Failed to retrieve complete JSON response after multiple attempts.");
    //     }
    //
    //     var parsed = ReadAndDeserializeSingleValue<TResponse>(responseString, SerializerOptions)!;
    //     return parsed;
    // }
    
    private bool IsJsonComplete(string jsonString)
    {
        // A very basic check, could be more sophisticated depending on your needs
        jsonString = jsonString.Trim();
        return jsonString.EndsWith("}") || jsonString.EndsWith("]");
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

    
    private static TResponse? ReadAndDeserializeSingleValue<TResponse>(string json, JsonSerializerOptions options)
    {
        try
        {
            var reader = new Utf8JsonReader(Encoding.UTF8.GetBytes(json).AsSpan());
            return JsonSerializer.Deserialize<TResponse>(ref reader, options);
        }
        catch(Exception ex)
        {
            // Console.WriteLine($"Error deserializing JSON {json}: {ex.Message}");
            Console.WriteLine("Error deserializing JSON");
            throw ex;
        }
    }

    private static async Task<List<T>> CollectAsyncEnumerable(IAsyncEnumerable<T> source)
    {
        var result = new List<T>();
        await foreach (var item in source)
        {
            result.Add(item);
        }
        return result;
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
}