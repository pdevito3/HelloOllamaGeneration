
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

builder.Services.AddSingleton<HttpClient>();
builder.Services.AddScoped<IChatCompletionService, OllamaChatCompletionService>();
// builder.AddOllamaChatCompletionService("localhost:11434", "phi3");
builder.AddOllamaChatCompletionService("localhost:11434", "mistral:7b");

var services = builder.Build().Services;

var categories = await new CategoryGenerator(services).GenerateAsync();
Console.WriteLine($"Got {categories.Count} categories");

var products = await new ProductGenerator(categories, services).GenerateAsync();
Console.WriteLine($"Got {products.Count} products");