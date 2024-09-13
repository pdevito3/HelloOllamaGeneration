namespace OllamaTools;

using Microsoft.SemanticKernel;

public static class OllamaChatFunction
{
    // When JSON-serialized, needs to produce a structure like this:
    // {
    //   "type": "function",
    //   "function": {
    //     "name": "get_current_weather",
    //     "description": "Get the current weather",
    //     "parameters": {
    //       "type": "object",
    //       "properties": {
    //         "location": {
    //           "type": "string",
    //           "description": "The city or country name"
    //         },
    //         "format": {
    //           "type": "string",
    //           "enum": ["celsius", "fahrenheit"],
    //           "description": "The temperature unit to use. Infer this from the users location."
    //         }
    //       },
    //       "required": ["location", "format"]
    //     }
    //   }
    // }

    public record ToolDescriptor(string Type, FunctionDescriptor Function);
    public record FunctionDescriptor(string Name, string Description, FunctionParameters Parameters);
    public record FunctionParameters(string Type, Dictionary<string, ParameterDescriptor> Properties, string[] Required);
    public record ParameterDescriptor(string Type, string? Description, string[]? Enum);

    public static ToolDescriptor Create(KernelFunctionMetadata metadata)
    {
        return new ToolDescriptor("function", new FunctionDescriptor(metadata.Name, metadata.Description, ToFunctionParameters(metadata)));
    }

    private static FunctionParameters ToFunctionParameters(KernelFunctionMetadata kernelFunction)
    {
        var parameters = kernelFunction.Parameters;
        return new FunctionParameters(
            "object",
            parameters.ToDictionary(p => p.Name!, ToParameterDescriptor),
            parameters.Where(p => p.IsRequired).Select(p => p.Name!).ToArray());
    }

    private static ParameterDescriptor ToParameterDescriptor(KernelParameterMetadata parameterInfo)
        => new ParameterDescriptor(
            ToParameterType(parameterInfo.ParameterType),
            parameterInfo.Description,
            ToEnumValues(parameterInfo?.ParameterType));

    private static string[]? ToEnumValues(Type? type)
        => type is not null && type.IsEnum ? Enum.GetNames(type) : null;

    private static string ToParameterType(Type? parameterType)
    {
        if (parameterType is null)
        {
            return "object";
        }

        parameterType = Nullable.GetUnderlyingType(parameterType) ?? parameterType;
        if (parameterType == typeof(int))
        {
            return "number";
        }
        else if (parameterType == typeof(string))
        {
            return "string";
        }
        else
        {
            throw new NotSupportedException($"Unsupported parameter type {parameterType}");
        }
    }
}