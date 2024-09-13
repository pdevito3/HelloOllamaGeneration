namespace MinimalAi.Features;

using System.Text.RegularExpressions;
using HelloOllamaGeneration;
using HelloOllamaGeneration.Models;
using Microsoft.Extensions.Options;
using Microsoft.SemanticKernel.ChatCompletion;
using MinimalAi;
using MinimalAi.Models;
using Serilog;

public static class GenerateCategories
{
    public static void MapGenerateCategoriesEndpoint(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapPost("/categories", async (IOLlamaInteractionService oLlamaInteractionService) =>
            {
                return await Handle(oLlamaInteractionService);
            })
        .WithName("GetCategories")
        .WithOpenApi();
    }

    private static async Task<List<Category>> Handle(IOLlamaInteractionService oLlamaInteractionService)
    {
        var numCategories = 10;
        var batchSize = 10;
        Log.Information("Generating {BatchSize} categories...", batchSize);
        var categoryNames = new HashSet<string>();

        var prompt = @$"Generate {batchSize} product category names for an online retailer
                of high-tech outdoor adventure goods and related clothing/electronics/etc.
                Each category name is a single descriptive term, so it does not use the word 'and'.
                Category names should be interesting and novel, e.g., ""Mountain Unicycles"", ""AI Boots"",
                or ""High-volume Water Filtration Plants"", not simply ""Tents"".
                This retailer sells relatively technical products.

                Each category has a list of up to 8 brand names that make products in that category. All brand names are
                purely fictional. Brand names are usually multiple words with spaces and/or special characters, e.g.
                ""Orange Gear"", ""Aqua Tech US"", ""Livewell"", ""E & K"", ""JAXⓇ"".
                Many brand names are used in multiple categories. Some categories have only 2 brands.
                
                The response should be in a JSON format like below with the exact batch count of objects. It is very important you make sure it is in this format and that it is valid JSON. 
                {{ ""categories"": [{{""name"":""Tents"", ""brands"":[""Rosewood"", ""Summit Kings""]}}] }}";
            
        static string ImproveBrandName(string name)
        {
            // Almost invariably, the name is a PascalCase word like AquaTech, even though we told it to use spaces.
            // For variety, convert to "Aqua Tech" or "Aquatech" sometimes.
            return Regex.Replace(name, @"([a-z])([A-Z])", m => Random.Shared.Next(3) switch
            {
                0 => $"{m.Groups[1].Value} {m.Groups[2].Value}",
                1 => $"{m.Groups[1].Value}{m.Groups[2].Value.ToLower()}",
                _ => m.Value
            });
        }
            
        var response = await oLlamaInteractionService.GetAndParseJsonChatCompletion<CategoriesResponse>(prompt, maxTokens: 70 * batchSize);
        foreach (var c in response.Categories)
        {
            if (categoryNames.Add(c.Name))
            {
                c.CategoryId = categoryNames.Count;
                c.Brands = c.Brands.Select(ImproveBrandName).ToArray();
            }
        }

        return response.Categories;
    }
}