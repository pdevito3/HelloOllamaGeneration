
using HelloOllamaGeneration;
using HelloOllamaGeneration.Generators;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel;

var builder = Host.CreateApplicationBuilder(args);
// builder.Configuration.AddJsonFile("appsettings.json");
// builder.Configuration.AddJsonFile("appsettings.Local.json", optional: true);
// builder.AddOpenAIChatCompletion("chatcompletion");

const string modelName = "mistral:7b";
// const string modelName = "phi3";
const string name = "localhost:11434";

builder.Services.AddSingleton<HttpClient>();
builder.Services.AddScoped<IChatCompletionService, OllamaChatCompletionService>();
builder.AddOllamaChatCompletionService(name, modelName);

builder.Services.AddScoped<ISimpleOllamaChatService, SimpleOllamaChatService>();
builder.Services.AddScoped<ISimpleOllamaChatService>(services =>
{
    var httpClient = services.GetRequiredService<HttpClient>();
    httpClient.BaseAddress = new Uri($"http://{name}");
    return new SimpleOllamaChatService(httpClient, modelName);
});

var services = builder.Build().Services;

var categories = await new CategoryGenerator(services).GenerateAsync();
Console.WriteLine($"Got {categories.Count} categories");

var products = await new ProductGenerator(categories, services).GenerateAsync();
Console.WriteLine($"Got {products.Count} products");