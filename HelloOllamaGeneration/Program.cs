
using HelloOllamaGeneration;
using HelloOllamaGeneration.Generators;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel;

var builder = Host.CreateApplicationBuilder(args);
builder.Configuration.AddJsonFile("appsettings.json");
// builder.Configuration.AddJsonFile("appsettings.Local.json", optional: true);
// builder.AddOpenAIChatCompletion("chatcompletion");

// const string modelName = "mistral:7b";
// // const string modelName = "llama3.1";
// // const string modelName = "phi3";
// const string name = "localhost:11434";
builder.Services
    .Configure<ModelInfoOptions>(builder.Configuration.GetSection("ModelInfo"));

var modelInfo = new ModelInfoOptions();
builder.Configuration.GetSection("ModelInfo").Bind(modelInfo);
Console.WriteLine($"ModelInfo: {modelInfo.Url}, {modelInfo.ModelName}");

builder.Services.AddSingleton<HttpClient>();
builder.Services.AddScoped<IChatCompletionService, OllamaChatCompletionService>();
builder.AddOllamaChatCompletionService(modelInfo.Url, modelInfo.ModelName);

builder.Services.AddScoped<ISimpleOllamaChatService, SimpleOllamaChatService>();
builder.Services.AddScoped<ISimpleOllamaChatService>(services =>
{
    var httpClient = services.GetRequiredService<HttpClient>();
    httpClient.BaseAddress = new Uri($"http://{modelInfo.Url}");
    httpClient.Timeout = TimeSpan.FromMinutes(5);
    return new SimpleOllamaChatService(httpClient, modelInfo.ModelName);
});

var services = builder.Build().Services;

var categories = await new CategoryGenerator(services).GenerateAsync();
Console.WriteLine($"Got {categories.Count} categories");

var products = await new ProductGenerator(categories, services).GenerateAsync();
Console.WriteLine($"Got {products.Count} products");

var manualTocs = await new ManualTocGenerator(categories, products, services).GenerateAsync();
Console.WriteLine($"Got {manualTocs.Count} manual TOCs");

var manuals = await new ManualGenerator(categories, products, manualTocs, services).GenerateAsync();
Console.WriteLine($"Got {manuals.Count} manuals");

var manualPdfs = await new ManualPdfConverter(products, manuals).ConvertAsync();
Console.WriteLine($"Got {manualPdfs.Count} PDFs");

var tickets = await new TicketGenerator(products, categories, manuals, services).GenerateAsync();
Console.WriteLine($"Got {tickets.Count} tickets");

var ticketThreads = await new TicketThreadGenerator(tickets, products, manuals, services).GenerateAsync();
Console.WriteLine($"Got {ticketThreads.Count} threads");

var summarizedThreads = await new TicketSummaryGenerator(products, ticketThreads, services).GenerateAsync();
Console.WriteLine($"Got {summarizedThreads.Count} thread summaries");

var evalQuestions = await new EvalQuestionGenerator(products, categories, manuals, services).GenerateAsync();
Console.WriteLine($"Got {evalQuestions.Count} evaluation questions");