namespace MinimalAi;

using Microsoft.Extensions.Options;
using OllamaTools;

public static class Extensions
{
    public static void AddCustomMistralService(this IServiceCollection services)
    {
        services.AddKeyedScoped<ISimpleOllamaChatService, SimpleOllamaChatService>(Consts.KeyedServices.Mistral, (serviceProvider, _) =>
        {
            var httpClientFactory = serviceProvider.GetRequiredService<IHttpClientFactory>();
            var httpClient = httpClientFactory.CreateClient(Consts.KeyedServices.Mistral);
            var modelInfo = serviceProvider.GetRequiredService<IOptionsSnapshot<ModelInfoOptions>>().Value;
    
            httpClient.BaseAddress = new Uri($"http://{modelInfo.Url}");
            httpClient.Timeout = TimeSpan.FromMinutes(5);
    
            return new SimpleOllamaChatService(httpClient, "mistral:7b");
        });
        services.AddKeyedScoped<IOLlamaInteractionService, MistralService>(Consts.KeyedServices.Mistral);
    }
    
    public static void AddCustomLlamaThreeOneService(this IServiceCollection services)
    {
        services.AddKeyedScoped<ISimpleOllamaChatService, SimpleOllamaChatService>(Consts.KeyedServices.Llama31, (serviceProvider, _) =>
        {
            var httpClientFactory = serviceProvider.GetRequiredService<IHttpClientFactory>();
            var httpClient = httpClientFactory.CreateClient(Consts.KeyedServices.Llama31);
            var modelInfo = serviceProvider.GetRequiredService<IOptionsSnapshot<ModelInfoOptions>>().Value;
    
            httpClient.BaseAddress = new Uri($"http://{modelInfo.Url}");
            httpClient.Timeout = TimeSpan.FromMinutes(5);
    
            return new SimpleOllamaChatService(httpClient, "llama3.1");
        });
        services.AddKeyedScoped<IOLlamaInteractionService, LlamaThreeOneService>(Consts.KeyedServices.Llama31);
    }
}