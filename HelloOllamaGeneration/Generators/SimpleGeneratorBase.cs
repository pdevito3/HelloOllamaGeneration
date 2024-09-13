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
using Microsoft.Extensions.Configuration;
using OllamaTools;

public abstract class SimpleGeneratorBase<T>(IServiceProvider services)
{
    protected abstract string DirectoryName { get; }

    protected abstract object GetId(T item);

    public static string OutputDirRoot => Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "output");

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
    
    protected ModelInfoOptions ModelInfoOptions()
    {
        var modelInfo = new ModelInfoOptions();
        services.GetRequiredService<IConfiguration>().GetSection("ModelInfo").Bind(modelInfo);
        return modelInfo;
    }

    protected async Task<TResponse> GetAndParseJsonChatCompletion<TResponse>(string prompt, int? maxTokens = null, object? tools = null)
    {
        // TODO tools are assumed to be on the kernel but are not actually passed in
        // mistral example https://ollama.com/library/mistral
        // llama example in bruno
        var executionSettings = new PromptSettings
        {
            MaxTokens = maxTokens,
            Temperature = 0.9f,
            ResponseFormat = ResponseFormat.Json,
            FormatRawPrompt = ModelInfoOptions().ModelName == "mistral:7b" 
                    ? (messages, k, autoInvoke) => FormatMistralPromptWithFunctions(messages, k, autoInvoke)
                    : (messages, k, autoInvoke) => FormatLlamaThreeOnePromptWithFunctions(messages, k, autoInvoke),
            StopSequences = ModelInfoOptions().ModelName == "mistral:7b"  
                ? new []{ "[/TOOL_CALLS]" } 
                : new []{ "<|eom_id|>", "<|eot_id|>" },
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
    
    protected async Task<string> GetChatCompletion(string prompt)
    {
        var executionSettings = new PromptSettings
        {
            Temperature = 0.9f,
            FormatRawPrompt = ModelInfoOptions().ModelName == "mistral:7b" 
                ? (messages, k, autoInvoke) => FormatSimpleMistralPrompt(messages, k, autoInvoke)
                : (messages, k, autoInvoke) => FormatSimpleLlamaThreeOnePrompt(messages, k, autoInvoke),
            ResponseFormat = ResponseFormat.Text,
            StopSequences = ModelInfoOptions().ModelName == "mistral:7b"  
                ? new[] { "END_OF_CONTENT" } 
                : new[] { "<|eom_id|>", "<|eot_id|>" }
        };

        var chatHistory = new ChatHistory();
        chatHistory.AddUserMessage(prompt);

        var kernel = (Kernel?)null;
        var response = await RunWithRetries(() =>
            ChatCompletionService.GetChatMessageContentAsync(chatHistory, executionSettings, kernel));
        return response?.Content;
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

    private static string FormatSimpleLlamaThreeOnePrompt(ChatHistory messages, Kernel? kernel, bool autoInvokeFunctions)
    {
        var sb = new StringBuilder();

        foreach (var message in messages)
        {
            if (message.Role == AuthorRole.System)
            {
                sb.Append("<|begin_of_text|><|start_header_id|>system<|end_header_id|> ").Append(message.Content).Append(" <|eot_id|>"); // i think i could use `eom_id` here but eot_id is safer and works for tool too
            }
            if (message.Role == AuthorRole.User)
            {
                sb.Append("<|begin_of_text|><|start_header_id|>user<|end_header_id|> ").Append(message.Content).Append(" <|eot_id|>"); // i think i could use `eom_id` here but eot_id is safer and works for tool too
            }
            if (message.Role == AuthorRole.Assistant)
            {
                sb.Append("<|begin_of_text|><|start_header_id|>assistant<|end_header_id|> ").Append(message.Content).Append(" <|eot_id|>"); // i think i could use `eom_id` here but eot_id is safer and works for tool too
            }
            else if (message.Role == AuthorRole.Tool)
            {
                // ipython: A new role introduced in Llama 3.1. Semantically, this role means "tool". This role is used to mark messages
                // with the output of a tool call when sent back to the model from the executor.
                sb.Append("<|begin_of_text|><|start_header_id|>ipython<|end_header_id|> ").Append(message.Content).Append(" <|eot_id|>");
            }
        }

        return sb.ToString();
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

    private static string FormatLlamaThreeOnePromptWithFunctions(ChatHistory messages, Kernel? kernel,
        bool autoInvokeFunctions)
    {
        var sb = new StringBuilder();

        foreach (var message in messages)
        {
            if (message.Role == AuthorRole.System)
            {
                sb.Append("<|begin_of_text|><|start_header_id|>system<|end_header_id|> ").Append(message.Content).Append(" <|eot_id|>"); // i think i could use `eom_id` here but eot_id is safer and works for tool too
            }
            if (message.Role == AuthorRole.User)
            {
                sb.Append("<|begin_of_text|><|start_header_id|>user<|end_header_id|> ").Append(message.Content).Append(" <|eot_id|>"); // i think i could use `eom_id` here but eot_id is safer and works for tool too
            }
            if (message.Role == AuthorRole.Assistant)
            {
                sb.Append("<|begin_of_text|><|start_header_id|>assistant<|end_header_id|> ").Append(message.Content).Append(" <|eot_id|>"); // i think i could use `eom_id` here but eot_id is safer and works for tool too
            }
            else if (message.Role == AuthorRole.Tool)
            {
                // ipython: A new role introduced in Llama 3.1. Semantically, this role means "tool". This role is used to mark messages
                // with the output of a tool call when sent back to the model from the executor.
                sb.Append("<|begin_of_text|><|start_header_id|>ipython<|end_header_id|> ").Append(message.Content).Append(" <|eot_id|>");
            }
        }
        
        // TODO but i think the serialize is close to a lift and shift
        // here are some docs https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling
        /*
         and this is a curl example
         
         curl --request POST \
             --url http://localhost:11434/api/generate \
             --header 'Content-Type: application/json; charset=utf-8' \
             --data '{
             "model": "llama3.1",
             "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original question.<|eot_id|><|start_header_id|>user<|end_header_id|>Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\nRespond in the format {\"name\": \"function name\", \"parameters\": {\"dictionary of argument name and its value\"}}. Do not use variables.\n\n{\n    \"type\": \"function\",\n    \"function\": {\n        \"name\": \"get_current_conditions\",\n        \"description\": \"Get the current weather conditions for a specific location\",\n        \"parameters\": {\n            \"type\": \"object\",\n            \"properties\": {\n                \"location\": {\n                    \"type\": \"string\",\n                    \"description\": \"The city and state, e.g., San Francisco, CA\"\n                },\n                \"unit\": {\n                    \"type\": \"string\",\n                    \"enum\": [\"Celsius\", \"Fahrenheit\"],\n                    \"description\": \"The temperature unit to use. Infer this from the user'\''s location.\"\n                }\n            },\n            \"required\": [\"location\", \"unit\"]\n        }\n    }\n}\n\nQuestion: what is the weather like in San Francisco?<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
             "options": {
               "temperature": 0.9,
               "numPredict": 1750,
               "topP": 1
             },
             "raw": true,
             "stream": false,
             "stop": [
               "<|eom_id|>",
               "<|eot_id|>"
             ]
           }'
         
         */
        // var indexOfLastUserOrSystemMessage = IndexOfLast(messages, m => m.Role == AuthorRole.User || m.Role == AuthorRole.System);
        //
        // // IMPORTANT: The whitespace in the prompt is significant. Do not add or remove extra spaces/linebreaks,
        // // as this affects tokenization. Mistral's function calling is useless unless you get this exactly right.
        //
        // for (var index = 0; index < messages.Count; index++)
        // {
        //     var message = messages[index];
        //
        //     // Emit tools descriptor immediately before the final [INST]
        //     if (index == indexOfLastUserOrSystemMessage && autoInvokeFunctions && kernel is not null)
        //     {
        //         var tools = kernel.Plugins.SelectMany(p => p.GetFunctionsMetadata()).ToArray() ?? [];
        //         if (tools is { Length: > 0 })
        //         {
        //             sb.Append("[AVAILABLE_TOOLS] ");
        //             sb.Append(JsonSerializer.Serialize(tools.Select(OllamaChatFunction.Create), OllamaJsonSettings.OllamaJsonSerializerOptions));
        //             sb.Append("[/AVAILABLE_TOOLS]");
        //         }
        //     }
        //
        //     if (message.Role == AuthorRole.User || message.Role == AuthorRole.System)
        //     {
        //         sb.Append("[INST] ");
        //         sb.Append(message.Content);
        //         sb.Append(" [/INST]");
        //     }
        //     else if (message.Role == AuthorRole.Tool)
        //     {
        //         sb.Append("[TOOL_CALLS] ");
        //         sb.Append(message.Content);
        //         sb.Append(" [/TOOL_CALLS]\n\n");
        //     }
        //     else
        //     {
        //         if (string.IsNullOrWhiteSpace(message.Content)) continue;
        //         
        //         sb.Append(message.Content);
        //         sb.Append("</s> "); // That's right, there's no matching <s>. See https://discuss.huggingface.co/t/skew-between-mistral-prompt-in-docs-vs-chat-template/66674/2
        //     }
        // }

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