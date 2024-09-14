namespace MinimalAi;

public interface IOLlamaInteractionService
{
    Task<TResponse> GetAndParseJsonChatCompletion<TResponse>(string prompt, int? maxTokens = null, object? tools = null);
    Task<string> GetChatCompletion(string prompt);
}