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
    client.BaseAddress = new Uri(configuration["ModelInfo:Url"]);
    client.Timeout = TimeSpan.FromMinutes(5);
});

builder.Services.AddCustomMistralService();
builder.Services.AddCustomLlamaThreeOneService();

builder.Services.AddScoped<IChatCompletionService>(services =>
{
    var httpClient = services.GetRequiredService<HttpClient>();
    httpClient.BaseAddress = new Uri($"http://{configuration["ModelInfo:Url"]}");
    return new OllamaChatCompletionService(httpClient, "mistral:7b");
});

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

app.MapGet("/weatherforecast/hardcode/{city}", async (string city, 
        [FromKeyedServices("llama3.1")] IOLlamaInteractionService llamaThreeOne) =>
    {
        // TODO this full prompt might be better raw as the service is trying to add sections to it. need to pull out a chat history and create it? 
         var prompt = $$$"""
                      <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original question.<|eot_id|><|start_header_id|>user<|end_header_id|>
                      Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.
                      
                      Respond in the format {"name": "function name", "parameters": {"dictionary of argument name and its value"}}. Do not use variables.
                      
                      
                      
                      {
                          "type": "function",
                          "function": {
                              "name": "get_current_conditions",
                              "description": "Get the current weather conditions for a specific location",
                              "parameters": {
                                  "type": "object",
                                  "properties": {
                                      "location": {
                                          "type": "string",
                                          "description": "The city and state, e.g., San Francisco, CA"
                                      },
                                      "unit": {
                                          "type": "string",
                                          "enum": ["Celsius", "Fahrenheit"],
                                          "description": "The temperature unit to use. Infer this from the user's location."
                                      }
                                  },
                                  "required": ["location", "unit"]
                              }
                          }
                      }
                      
                      
                      
                      Question: what is the weather like in {{{city}}}?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                      """;
        
        var response = await llamaThreeOne.GetChatCompletion(prompt);

        return response;
    })
    .WithName("GetWeatherForecastHardcode")
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
