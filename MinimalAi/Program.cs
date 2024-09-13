using System.Text.RegularExpressions;
using HelloOllamaGeneration;
using Microsoft.Extensions.Options;
using Microsoft.SemanticKernel.ChatCompletion;
using MinimalAi;
using MinimalAi.Features;
using MinimalAi.Models;
using OllamaTools;
using Serilog;

var builder = WebApplication.CreateBuilder(args);

Log.Logger = new LoggerConfiguration()
    .WriteTo.Console()
    .CreateLogger();
builder.Services.AddSerilog();

builder.Services.Configure<ModelInfoOptions>(builder.Configuration.GetSection("ModelInfo"));

const string ollamaClientName = "ollama";
var configuration = builder.Configuration;
builder.Services.AddHttpClient(ollamaClientName, client =>
{
    client.BaseAddress = new Uri(configuration["ModelInfo:ModelName"]);
    client.Timeout = TimeSpan.FromMinutes(5);
});
builder.Services.AddScoped<ISimpleOllamaChatService>(services =>
{
    var httpClientFactory = services.GetRequiredService<IHttpClientFactory>();
    var httpClient = httpClientFactory.CreateClient(ollamaClientName);
     var modelInfo = services.GetRequiredService<IOptionsSnapshot<ModelInfoOptions>>().Value;

    httpClient.BaseAddress = new Uri($"http://{modelInfo.Url}");
    httpClient.Timeout = TimeSpan.FromMinutes(5);

    return new SimpleOllamaChatService(httpClient, modelInfo.ModelName);
});

builder.Services.AddScoped<IOLlamaInteractionService, OLlamaInteractionService>();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();
app.UseSerilogRequestLogging();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();


app.MapGenerateCategoriesEndpoint();

app.MapGet("/weatherforecast", (IOptionsSnapshot<ModelInfoOptions> modelInfo) =>
    {
        return modelInfo;
    })
    .WithName("GetWeatherForecast")
    .WithOpenApi();

try
{
    Log.Information("Starting application");
    await app.RunAsync();
}
catch (Exception e)
{
    Log.Error(e, "The application failed to start correctly");
    throw;
}
finally
{
    Log.Information("Shutting down application");
    Log.CloseAndFlush();
}
