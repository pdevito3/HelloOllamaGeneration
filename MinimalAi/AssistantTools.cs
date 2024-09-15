namespace MinimalAi;

using Microsoft.SemanticKernel;
using System.Text;
using System.ComponentModel;
using Microsoft.SemanticKernel.Text;
using Microsoft.SemanticKernel.Embeddings;
using SmartComponents.LocalEmbeddings.SemanticKernel;
using System.Numerics.Tensors;

public class AssistantTools
{
    [KernelFunction, Description("Searches for weather information for a given city.")]
    public async Task<string> GetWeatherForCity([Description("Name of the city that we want to get the weather for")] string city)
    {
        var randomTemperature = Random.Shared.Next(60, 90);
        return $"The weather in {city} is {randomTemperature}°F and sunny.";
    }
}