namespace HelloOllamaGeneration.Generators;

using System.Diagnostics.CodeAnalysis;
using Models;

public class ProductGenerator(IReadOnlyList<Category> categories, IServiceProvider services) : SimpleGeneratorBase<Product>(services)
{
    protected override string DirectoryName => "products";

    protected override object GetId(Product item) => item.ProductId;

    [Experimental("SKEXP0010")]
    protected override async IAsyncEnumerable<Product> GenerateCoreAsync()
    {
        // If there are any products already, assume this covers everything we need
        if (Directory.GetFiles(OutputDirPath).Any())
        {
            yield break;
        }

        var numProducts = 200;
        var batchSize = 5;
        // var numProducts = 10;
        // var batchSize = 2;
        var productId = 0;

        var mappedBatches = MapParallel(Enumerable.Range(0, numProducts / batchSize), async batchIndex =>
        {
            var chosenCategories = Enumerable.Range(0, batchSize)
                .Select(_ => categories[(int)Math.Floor(categories.Count * Random.Shared.NextDouble())])
                .ToList();

            var prompt = @$"Write list of {batchSize} products for an online retailer
            of outdoor adventure goods and related electronics, clothing, and homeware. There is a focus on high-tech products. They match the following category/brand pairs:
            {string.Join(Environment.NewLine, chosenCategories.Select((c, index) => $"- product {(index + 1)}: category {c.Name}, brand: {c.Brands[Random.Shared.Next(c.Brands.Length)]}"))}

            Model names are up to 50 characters long, but usually shorter. Sometimes they include numbers, specs, or product codes.
            Example model names: ""iGPS 220c 64GB"", ""Nomad Camping Stove"", ""UX Polarized Sunglasses (Womens)"", ""40L Backpack, Green""
            Do not repeat the brand name in the model name.

            The description is up to 200 characters long and is the marketing text that will appear on the product page.
            Include the key features and selling points.

            The response should be in a JSON format like below with the exact batch count of objects. It is very important you make sure it is in this format and that it is valid JSON.

            {{
              ""products"": [
                {{ ""id"": 1, ""brand"": ""Garmin"", ""model"": ""iGPS 220c 64GB"", ""description"": ""High-precision GPS with 64GB storage, waterproof, and rugged design."", ""price"": 299.99 }}
              ]
            }}
";

            var response = await GetAndParseJsonChatCompletion<Response>(prompt, maxTokens: 200 * batchSize);
            var batchEntryIndex = 0;
            foreach (var p in response.Products!)
            {
                var category = chosenCategories[batchEntryIndex++];
                p.CategoryId = category.CategoryId;
            }

            return response.Products;
        });

        await foreach (var batch in mappedBatches)
        {
            foreach (var p in batch)
            {
                p.ProductId = ++productId;
                yield return p;
            }
        }
    }

    class Response
    {
        public List<Product>? Products { get; set; }
    }
}