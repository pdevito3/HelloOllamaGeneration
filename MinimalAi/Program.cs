using System.Text;
using System.Text.RegularExpressions;
using HelloOllamaGeneration;
using Microsoft.Extensions.Options;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
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

app.MapGet("/weatherforecast/hardcode/llama31/{city}", async (string city, 
        [FromKeyedServices("llama3.1")] IOLlamaInteractionService llamaThreeOne,
        [FromKeyedServices("llama3.1")] ISimpleOllamaChatService llamaThreeOneChatCompletionService) =>
    {
        // TODO this full prompt might be better raw as the service is trying to add sections to it. need to pull out a chat history and create it? 
//          var prompt = $$$"""
//                       <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original question.<|eot_id|><|start_header_id|>user<|end_header_id|>
//                       Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.
//                       
//                       Respond in the format {"name": "function name", "parameters": {"dictionary of argument name and its value"}}. Do not use variables.
//                       
//                       
//                       
//                       {
//                           "type": "function",
//                           "function": {
//                               "name": "get_current_conditions",
//                               "description": "Get the current weather conditions for a specific location",
//                               "parameters": {
//                                   "type": "object",
//                                   "properties": {
//                                       "location": {
//                                           "type": "string",
//                                           "description": "The city and state, e.g., San Francisco, CA"
//                                       },
//                                       "unit": {
//                                           "type": "string",
//                                           "enum": ["Celsius", "Fahrenheit"],
//                                           "description": "The temperature unit to use. Infer this from the user's location."
//                                       }
//                                   },
//                                   "required": ["location", "unit"]
//                               }
//                           }
//                       }
//                       
//                       
//                       
//                       Question: what is the weather like in {{{city}}}?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
//                       """;
        
        // var response = await llamaThreeOne.GetChatCompletion(prompt);

        ///-----
         var chatHistory = new ChatHistory();
         // chatHistory.AddSystemMessage("You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original question.");
         chatHistory.AddSystemMessage("""
                                      You are a helpful assistant that takes a question and finds the most appropriate tool or tools to execute, along with the parameters required to run the tool. Respond as JSON using the following schema: {"functionName": "function name", "parameters": [{"parameterName": "name of parameter", "parameterValue": "value of parameter"}]}. 
                                      
                                      The tools are:
                                      [
                                      {
                                          name: "CityToLatLon",
                                          description: "Get the latitude and longitude for a given city",
                                          parameters: [
                                      	    {
                                      		    name: "city",
                                      		    description: "The city to get the latitude and longitude for",
                                      		    type: "string",
                                      		    required: true,
                                      	    },
                                          ],
                                      },
                                      {
                                          name: "WeatherFromLatLon",
                                          description: "Get the weather for a location",
                                          parameters: [
                                      	    {
                                      		    name: "latitude",
                                      		    description: "The latitude of the location",
                                      		    type: "number",
                                      		    required: true,
                                      	    },
                                      	    {
                                      		    name: "longitude",
                                      		    description: "The longitude of the location",
                                      		    type: "number",
                                      		    required: true,
                                      	    },
                                          ],
                                      },
                                      {
                                      	name: "LatLonToCity",
                                      	description: "Get the city name for a given latitude and longitude",
                                      	parameters: [
                                      		{
                                      			name: "latitude",
                                      			description: "The latitude of the location",
                                      			type: "number",
                                      			required: true,
                                      		},
                                      		{
                                      			name: "longitude",
                                      			description: "The longitude of the location",
                                      			type: "number",
                                      			required: true,
                                      		},
                                      	],
                                      },
                                      
                                      {
                                      	name: "WebSearch",
                                      	description: "Search the web for a query",
                                      	parameters: [
                                      		{
                                      			name: "query",
                                      			description: "The query to search for",
                                      			type: "string",
                                      			required: true,
                                      		},
                                      	],
                                      },
                                      {
                                      	name: "WeatherFromLocation",
                                      	description: "Get the weather for a location",
                                      	parameters: [
                                      		{
                                      			name: "location",
                                      			description: "The location to get the weather for",
                                      			type: "string",
                                      			required: true,
                                      		},
                                      	],
                                      }
                                      ]
                                      """);
         
         chatHistory.AddUserMessage($$$"""
                                       What is the weather in {{{city}}}?
                                       """);

         var kernel = (Kernel?)null;
         
         var executionSettings = new PromptSettings
         {
             Temperature = 0.9f,
             ResponseFormat = ResponseFormat.Json,
             FormatRawPrompt = (messages, k, autoInvoke) => LlamaThreeOneService.FormatSimpleLlamaThreeOnePrompt(messages, k, autoInvoke),
             StopSequences = new[] { "<|eom_id|>", "<|eot_id|>" }
         };
         var response = await StaticUtils.RunWithRetries(() =>
             llamaThreeOneChatCompletionService.GetChatMessageContentAsync(chatHistory, executionSettings, kernel));
         
         return response?.Content;
    })
    .WithName("GetWeatherForecastHardcode")
    .WithOpenApi();


app.MapGet("/weatherforecast/hardcode/mistral/{city}", async (string city, 
        [FromKeyedServices("mistral")] IOLlamaInteractionService mistral,
        [FromKeyedServices("mistral")] ISimpleOllamaChatService mistralChatCompletionService) =>
    {
		var chatHistory = new ChatHistory();
        // chatHistory.AddSystemMessage("You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original question.");
        chatHistory.AddSystemMessage("""
                                     You are a helpful assistant that takes a question and finds the most appropriate tool or tools to execute, along with the parameters required to run the tool. Respond as JSON using the following schema: {"functionName": "function name", "parameters": [{"parameterName": "name of parameter", "parameterValue": "value of parameter"}]}. 
                                     
                                     The tools are:
                                     [
                                     {
                                         name: "CityToLatLon",
                                         description: "Get the latitude and longitude for a given city",
                                         parameters: [
                                     	    {
                                     		    name: "city",
                                     		    description: "The city to get the latitude and longitude for",
                                     		    type: "string",
                                     		    required: true,
                                     	    },
                                         ],
                                     },
                                     {
                                         name: "WeatherFromLatLon",
                                         description: "Get the weather for a location",
                                         parameters: [
                                     	    {
                                     		    name: "latitude",
                                     		    description: "The latitude of the location",
                                     		    type: "number",
                                     		    required: true,
                                     	    },
                                     	    {
                                     		    name: "longitude",
                                     		    description: "The longitude of the location",
                                     		    type: "number",
                                     		    required: true,
                                     	    },
                                         ],
                                     },
                                     {
                                     	name: "LatLonToCity",
                                     	description: "Get the city name for a given latitude and longitude",
                                     	parameters: [
                                     		{
                                     			name: "latitude",
                                     			description: "The latitude of the location",
                                     			type: "number",
                                     			required: true,
                                     		},
                                     		{
                                     			name: "longitude",
                                     			description: "The longitude of the location",
                                     			type: "number",
                                     			required: true,
                                     		},
                                     	],
                                     },
                                     
                                     {
                                     	name: "WebSearch",
                                     	description: "Search the web for a query",
                                     	parameters: [
                                     		{
                                     			name: "query",
                                     			description: "The query to search for",
                                     			type: "string",
                                     			required: true,
                                     		},
                                     	],
                                     },
                                     {
                                     	name: "WeatherFromLocation",
                                     	description: "Get the weather for a location",
                                     	parameters: [
                                     		{
                                     			name: "location",
                                     			description: "The location to get the weather for",
                                     			type: "string",
                                     			required: true,
                                     		},
                                     	],
                                     }
                                     ]
                                     """);
        
        chatHistory.AddUserMessage($$$"""
                                      What is the weather in {{{city}}}?
                                      """);

        var kernel = (Kernel?)null;
        
        var executionSettings = new PromptSettings
        {
            Temperature = 0.9f,
            ResponseFormat = ResponseFormat.Json,
            FormatRawPrompt = (messages, k, autoInvoke) => MistralService.FormatMistralPromptWithFunctions(messages, k, autoInvoke),
            StopSequences = new []{ "[/TOOL_CALLS]" }
        };
        var response = await StaticUtils.RunWithRetries(() =>
            mistralChatCompletionService.GetChatMessageContentAsync(chatHistory, executionSettings, kernel));
        
        return response?.Content;
    })
    .WithName("GetWeatherForecastHardcodeMistral")
    .WithOpenApi();

app.MapGet("/weatherforecast/hardcode/mistral/kerneltool/{city}", async (string city, 
		[FromKeyedServices("mistral")] IOLlamaInteractionService mistral,
		[FromKeyedServices("mistral")] ISimpleOllamaChatService mistralChatCompletionService,
		IChatCompletionService kernelChatCompletionService) =>
	{
		var chatHistory = new ChatHistory();
		// chatHistory.AddSystemMessage("You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original question.");
// 		chatHistory.AddSystemMessage("""
// 		                             You are a helpful assistant that takes a question and finds the most appropriate tool or tools to execute, along with the parameters required to run the tool. Respond as JSON using the following schema: {"functionName": "function name", "parameters": [{"parameterName": "name of parameter", "parameterValue": "value of parameter"}]}. 
// 		                             """);
//         
// 		chatHistory.AddUserMessage($$$"""
// 		                              What is the weather in {{{city}}}?
// 		                              """);

		// var kernel = (Kernel?)null;
		// kernel.ImportPluginFromObject(new AssistantTools());
  //       
		// var executionSettings = new PromptSettings
		// {
		// 	Temperature = 0.9f,
		// 	ResponseFormat = ResponseFormat.Json,
		// 	FormatRawPrompt = (messages, k, autoInvoke) => MistralService.FormatMistralPromptWithFunctions(messages, k, autoInvoke),
		// 	StopSequences = new []{ "[/TOOL_CALLS]" }
		// };
		// var response = await StaticUtils.RunWithRetries(() =>
		// 	mistralChatCompletionService.GetChatMessageContentAsync(chatHistory, executionSettings, kernel));
  //       
		// return response?.Content;
		
		
		var kernel = new Kernel();
		kernel.ImportPluginFromObject(new AssistantTools());
		// chatHistory.AddUserMessage($$$"""can you show me a basic for loop in c#""");
		
		// chatHistory.AddSystemMessage("""
		//                              You are a helpful assistant that takes a question and finds the most appropriate tool or tools to execute, along with the parameters required to run the tool. Respond as JSON using the following schema: {"functionName": "function name", "parameters": [{"parameterName": "name of parameter", "parameterValue": "value of parameter"}]}. 
		//                              """);
        
		chatHistory.AddUserMessage($$$"""
		                              What is the weather in {{{city}}}?
		                              """);

		var executionSettings = new OpenAIPromptExecutionSettings { Seed = 0, Temperature = 0, ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions };
		var answer = await kernelChatCompletionService.GetChatMessageContentsAsync(chatHistory, executionSettings, kernel);
		var response = answer.FirstOrDefault();
		
		return response?.Content;
	})
	.WithName("GetWeatherForecastHardcodeMistralKernel")
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
