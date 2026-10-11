using System.Text.Json;
namespace NInferEZ.Converter;
internal sealed class ConversionSettings
{
    public Dictionary<string, object?> Values { get; private set; } = new()
    { ["components"] = new[] { "text" }, ["device"] = "cpu", ["profile"] = "auto", ["rowsPerChunk"] = 512,
      ["maxFileBytes"] = 32000000000L, ["proposal"] = false, ["proposalRows"] = 131072, ["validateTimeoutSeconds"] = 600 };
    public string Text(string key) => Values.TryGetValue(key, out var v) ? v is JsonElement j ? j.ToString() : v?.ToString() ?? "" : "";
    public bool Flag(string key) => bool.TryParse(Text(key), out var v) && v;
    public string[] Components => Values.TryGetValue("components", out var v) ? JsonSerializer.Deserialize<string[]>(JsonSerializer.Serialize(v)) ?? ["text"] : ["text"];
    public Dictionary<string, string> Map(string key)
    {
        try { return Values.TryGetValue(key, out var v) ? JsonSerializer.Deserialize<Dictionary<string, string>>(JsonSerializer.Serialize(v)) ?? new() : new(); }
        catch (JsonException) { return new(); }
    }
    public void Set(string key, object? value) => Values[key] = value;
    public void SetMap(string key, string name, string value) { var map = Map(key); if (string.IsNullOrWhiteSpace(value)) map.Remove(name); else map[name] = value; Set(key, map); }
    public Dictionary<string, object?> Request()
    {
        var request = new Dictionary<string, object?>(Values.Where(p => p.Value is not null && (!(p.Value is string s) || !string.IsNullOrWhiteSpace(s))).ToDictionary(p => p.Key, p => p.Value));
        if (System.IO.Path.GetExtension(Text("source")).Equals(".ninfer", StringComparison.OrdinalIgnoreCase))
        {
            string[] keys = ["source", "output", "profile", "maxFileBytes", "validateEngine", "validateTimeoutSeconds"];
            return request.Where(p => keys.Contains(p.Key)).ToDictionary(p => p.Key, p => p.Value);
        }
        if (!Components.Contains("vision") && request.ContainsKey("resources"))
        { var resources = Map("resources"); resources.Remove("preprocessor_config.json"); resources.Remove("video_preprocessor_config.json"); request["resources"] = resources; }
        return request;
    }
    public void Import(string json)
    {
        var data = JsonSerializer.Deserialize<Dictionary<string, object?>>(json) ?? throw new InvalidOperationException("Request must be a JSON object.");
        foreach (var key in new[] { "expectedPlanSignature", "expectedFileState" }) data.Remove(key);
        Values = data; if (!Values.ContainsKey("components")) Set("components", new[] { "text" });
        if (!Values.ContainsKey("profile")) Set("profile", "auto"); if (!Values.ContainsKey("device")) Set("device", "cpu");
    }
    public string Export() => JsonSerializer.Serialize(Request(), new JsonSerializerOptions { WriteIndented = true });
}
