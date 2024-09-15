namespace MinimalAi;

using Newtonsoft.Json;

public class WeatherTools
{
    public class Tool
    {
        public string Name { get; set; }
        public string Description { get; set; }
        public List<ToolParameter> Parameters { get; set; }
    }

    public class ToolParameter
    {
        public string Name { get; set; }
        public string Description { get; set; }
        public string Type { get; set; }
        public bool Required { get; set; }
    }

    public class FunctionParameter
    {
        public string ParameterName { get; set; }
        public string ParameterValue { get; set; }
    }

    public static Tool cityToLatLonTool = new Tool
    {
        Name = "CityToLatLon",
        Description = "Get the latitude and longitude for a given city",
        Parameters = new List<ToolParameter>
        {
            new ToolParameter
            {
                Name = "city",
                Description = "The city to get the latitude and longitude for",
                Type = "string",
                Required = true,
            }
        }
    };

    public static Tool weatherFromLatLonTool = new Tool
    {
        Name = "WeatherFromLatLon",
        Description = "Get the weather for a location",
        Parameters = new List<ToolParameter>
        {
            new ToolParameter
            {
                Name = "latitude",
                Description = "The latitude of the location",
                Type = "number",
                Required = true,
            },
            new ToolParameter
            {
                Name = "longitude",
                Description = "The longitude of the location",
                Type = "number",
                Required = true,
            },
        }
    };

    public static Tool latlonToCityTool = new Tool
    {
        Name = "LatLonToCity",
        Description = "Get the city name for a given latitude and longitude",
        Parameters = new List<ToolParameter>
        {
            new ToolParameter
            {
                Name = "latitude",
                Description = "The latitude of the location",
                Type = "number",
                Required = true,
            },
            new ToolParameter
            {
                Name = "longitude",
                Description = "The longitude of the location",
                Type = "number",
                Required = true,
            },
        }
    };

    public static Tool weatherFromLocationTool = new Tool
    {
        Name = "WeatherFromLocation",
        Description = "Get the weather for a location",
        Parameters = new List<ToolParameter>
        {
            new ToolParameter
            {
                Name = "location",
                Description = "The location to get the weather for",
                Type = "string",
                Required = true,
            },
        }
    };

    public static string toolsString = JsonConvert.SerializeObject(
        new
        {
            tools = new List<Tool>
            {
                weatherFromLocationTool,
                weatherFromLatLonTool,
                latlonToCityTool,
            }
        },
        Formatting.Indented);
    
    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient();
        client.DefaultRequestHeaders.Add("User-Agent", "Demo/1.0 (demo@demo.com)");
        return client;
    }
    
    public static async Task<(string, string)> CityToLatLon(string city)
    {
        using HttpClient client = CreateHttpClient();
        var url = $"https://nominatim.openstreetmap.org/search?q={Uri.EscapeDataString(city)}&format=json";
        var response = await client.GetAsync(url);
        response.EnsureSuccessStatusCode();
        var jsonString = await response.Content.ReadAsStringAsync();
        dynamic json = JsonConvert.DeserializeObject(jsonString);
        return ((string)json[0].lat, (string)json[0].lon);
    }

    public static async Task<string> LatLonToCity(string latitude, string longitude)
    {
        using HttpClient client = CreateHttpClient();
        var url = $"https://nominatim.openstreetmap.org/reverse?lat={latitude}&lon={longitude}&format=json";
        var response = await client.GetAsync(url);
        response.EnsureSuccessStatusCode();
        var jsonString = await response.Content.ReadAsStringAsync();
        dynamic json = JsonConvert.DeserializeObject(jsonString);
        string cityName = json.display_name;
        Console.WriteLine(cityName);
        return cityName;
    }

    public static async Task<string> WeatherFromLatLon(string latitude, string longitude)
    {
        using HttpClient client = CreateHttpClient();
        var url = $"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true&temperature_unit=fahrenheit&wind_speed_unit=mph&forecast_days=1";
        var response = await client.GetAsync(url);
        response.EnsureSuccessStatusCode();
        var jsonString = await response.Content.ReadAsStringAsync();
        dynamic json = JsonConvert.DeserializeObject(jsonString);
        string result = $"{json.current_weather.temperature} degrees Fahrenheit";
        Console.WriteLine(result);
        return result;
    }

    public static async Task<string> WeatherFromLocation(string location)
    {
        var (lat, lon) = await CityToLatLon(location);
        return await WeatherFromLatLon(lat, lon);
    }

    public static string GetValueOfParameter(string parameterName, List<FunctionParameter> parameters)
    {
        return parameters.Find(p => p.ParameterName == parameterName)?.ParameterValue;
    }

    public static async Task<object> ExecuteFunction(string functionName, List<FunctionParameter> parameters)
    {
        return functionName switch
        {
            "WeatherFromLocation" => await WeatherFromLocation(GetValueOfParameter("location", parameters)),
            "WeatherFromLatLon" => await WeatherFromLatLon(GetValueOfParameter("latitude", parameters),
                GetValueOfParameter("longitude", parameters)),
            "LatLonToCity" => await LatLonToCity(GetValueOfParameter("latitude", parameters),
                GetValueOfParameter("longitude", parameters)),
            "CityToLatLon" => await CityToLatLon(GetValueOfParameter("city", parameters)),
            _ => null
        };
    }
    
    public class FunctionRequest
    {
        public string FunctionName { get; set; }
        public List<FunctionParameter> Parameters { get; set; }
    }
}