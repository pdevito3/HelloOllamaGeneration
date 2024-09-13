namespace OllamaTools;

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.SemanticKernel.ChatCompletion;

public static class ChatCompletionServiceExtensions
{
    // public static void AddOpenAIChatCompletion(this HostApplicationBuilder builder, string connectionStringName)
    // {
    //     var connectionStringBuilder = new DbConnectionStringBuilder();
    //     var connectionString = builder.Configuration.GetConnectionString(connectionStringName);
    //     if (string.IsNullOrEmpty(connectionString))
    //     {
    //         throw new InvalidOperationException($"Missing connection string {connectionStringName}");
    //     }
    //
    //     connectionStringBuilder.ConnectionString = connectionString;
    //
    //     var deployment = connectionStringBuilder.TryGetValue("Deployment", out var deploymentValue) ? (string)deploymentValue : throw new InvalidOperationException($"Connection string {connectionStringName} is missing 'Deployment'");
    //     var endpoint = connectionStringBuilder.TryGetValue("Endpoint", out var endpointValue) ? (string)endpointValue : throw new InvalidOperationException($"Connection string {connectionStringName} is missing 'Endpoint'");
    //     var key = connectionStringBuilder.TryGetValue("Key", out var keyValue) ? (string)keyValue : throw new InvalidOperationException($"Connection string {connectionStringName} is missing 'Key'");
    //
    //     builder.Services.AddSingleton<IChatCompletionService>(services =>
    //         new AzureOpenAIChatCompletionService(deployment, endpoint, key));
    // }
    
    public static void AddOllamaChatCompletionService(this IHostApplicationBuilder builder, string name, string? model = null)
    {
        var modelName = model ?? builder.Configuration[$"{name}:LlmModelName"];
        if (string.IsNullOrEmpty(modelName))
        {
            throw new InvalidOperationException($"Expected to find the default LLM model name in an environment variable called '{name}:LlmModelName'");
        }

        builder.Services.AddScoped<IChatCompletionService>(services =>
        {
            var httpClient = services.GetRequiredService<HttpClient>();
            httpClient.BaseAddress = new Uri($"http://{name}");
            return new OllamaChatCompletionService(httpClient, modelName);
        });
    }
}

