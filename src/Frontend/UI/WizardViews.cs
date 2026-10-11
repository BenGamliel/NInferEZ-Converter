using System.Text.Json;
using System.Windows;
using System.Windows.Controls;

namespace NInferEZ.Converter;
internal sealed partial class MainWindow
{
    string[] selectedWarnings = [];
    string Description(string profile) => ArtifactInput ? "Repack an existing v3 artifact or upgrade an official v2 artifact while retaining its weight encodings. The v2 upgrader also installs the current official chat template." : profile switch
    {
        "auto" => "Use the architecture's automatic policy. Review the actual mixed encodings before converting.",
        "preserve" => "Retain supported source encodings. Engine-required casts and special-case conversions may still apply.",
        "q4" => "Smaller integer weights. Most main matrices use 4 bits; vocabulary endpoints retain Q8 and sensitive weights stay direct.",
        "q5" => "5-bit main matrices: a middle ground between size and retained precision. Engine exceptions still apply.",
        "q6" => "6-bit main matrices: less compression and more retained precision. Engine exceptions still apply.",
        "q8" => "8-bit main matrices: larger than Q4/Q5/Q6, with smaller quantization steps. Sensitive weights remain direct.",
        "bf16" => "Store direct BF16 weights. Expanding a quantized source cannot recover its original precision.",
        "fp8" => "Row-scaled 8-bit floating-point projections. Sensitive parameters stay in the engine's direct formats.",
        _ => "Apply a built-in or custom conversion recipe. The preview reflects its actual assignments."
    };
    FrameworkElement Field(string key, string title, string kind = "file", string help = "", bool affectsPlan = true) =>
        Ui.PathField(this, title, settings.Text(key), value => Put(key, value, affectsPlan), kind, help);
    FrameworkElement SourceView() => Ui.Column(
        Ui.Card("Choose your model", "Start with local weights. The wizard reads the model and checks what this engine can accept.",
            Field("source", "Model file or checkpoint folder", "fileOrFolder", "GGUF, Safetensors, an index/config.json, a checkpoint folder, or an existing .ninfer. You can also drag a path into the field."),
            Ui.Text("No model files are modified. Missing configuration or tokenizer files can be attached on the next step.", 13, "Muted")),
        Ui.Card("A plan you can inspect", "Choose how to convert after the model is scanned.",
            Ui.Text("1   Check the model and attach missing files\n2   Compare supported profiles and estimated weight sizes\n3   Review encoding changes and possible precision loss\n4   Save and convert with live progress", 15)),
        Ui.Text("Need exact recipes, custom sources or per-weight assignments? Manual setup exposes the full conversion API.", 13, "Muted"));

    FrameworkElement ResourcesView()
    {
        if (discovery.ValueKind != JsonValueKind.Undefined && discovery.GetProperty("source").GetProperty("container").GetString()!.StartsWith("NInfer"))
            return Ui.Card("Existing NInfer artifact", "The configuration and tokenizer already travel inside this model.",
                Ui.Text("This input supports lossless repacking or v2 upgrade. Its existing components and encodings are retained; re-quantization and replacing its resources are not available in this path.", 15),
                Ui.Text("Continue to review the supported path and select the output location/shard size.", 13, "Muted"));
        var stack = new StackPanel();
        stack.Children.Add(Ui.Card("Model resources", "Use files from the exact checkpoint. The wizard checks their presence and the conversion planner validates their compatibility.",
            Ui.PathField(this, "Matching config / tokenizer folder", settings.Text("resourceFolder"), value => { settings.Set("resourceFolder", value); settings.Set("configFolder", value); settings.Set("configFile", ""); Changed(); }, "folder", "A folder can supply all required files at once. Individually selected files take precedence.")));
        if (discovery.ValueKind != JsonValueKind.Undefined && discovery.TryGetProperty("requirements", out var requirements))
        {
            foreach (var item in requirements.EnumerateArray())
            {
                string key = item.GetProperty("key").GetString()!, status = item.GetProperty("status").GetString()!;
                string path = item.GetProperty("path").GetString()!, title = item.GetProperty("label").GetString()!;
                var picker = Ui.PathField(this, title, path, value => { if (key.StartsWith("resource:")) PutMap("resources", key[9..], value); else if (key.StartsWith("source:")) PutMap("sources", key[7..], value); else if (key.StartsWith("shard:")) PutMap("shardOverrides", key[6..], value); else Put(key, value); }, item.GetProperty("kind").GetString()!, item.GetProperty("help").GetString()!);
                stack.Children.Add(Ui.Card((status == "ready" ? "✓  " : "!  ") + title, status == "ready" ? "File found — validated during plan preparation" : "Required: missing or invalid", picker));
            }
        }
        stack.Children.Add(Ui.Card("Optional components", "Text is included. Only select extra components you intend to run; they need compatible weights and resources.", ComponentsEditor(),
            new Expander { Header = "Companion model sources", Content = SourcesEditor() }));
        stack.Children.Add(new Expander { Header = "Recipes, proposal head & processing settings", Content = TuningEditor() });
        return stack;
    }

    FrameworkElement OptionsView()
    {
        var stack = new StackPanel(); stack.Children.Add(Ui.Text("Choose the conversion you want", 25));
        stack.Children.Add(Ui.Text("Each available option is a real prepared plan. Sizes are encoded weights, not total VRAM. Select one to see its actual encodings and exceptions.", 13, "Muted"));
        if (analysis.ValueKind == JsonValueKind.Undefined) { stack.Children.Add(Ui.Text("Settings changed. Return to Required files and scan again.", 14, "Warning")); return stack; }
        foreach (var option in analysis.GetProperty("options").EnumerateArray())
        {
            var copy = option.Clone(); bool supported = option.GetProperty("supported").GetBoolean(); string id = option.GetProperty("profile").GetString()!;
            var card = Ui.Card(id.ToUpperInvariant() + (plan.ValueKind != JsonValueKind.Undefined && settings.Text("profile") == id ? "   •   SELECTED" : ""), Description(id));
            var column = (StackPanel)card.Child;
            if (supported)
            {
                var p = option.GetProperty("plan"); string size = p.TryGetProperty("encodedBytes", out var bytes) ? Ui.Size(bytes.GetInt64()) + " encoded weights" : p.GetProperty("mode").GetString()!;
                column.Children.Add(Ui.Text(size + "  •  " + (option.TryGetProperty("quality", out var q) ? q.GetString() : "Existing encoding"), 13, "Accent"));
                column.Children.Add(Ui.Button("Select " + id.ToUpperInvariant(), () => { SelectOption(copy); Render(); }));
            }
            else column.Children.Add(Ui.Text("Unavailable: " + option.GetProperty("reason").GetString(), 13, "Danger"));
            stack.Children.Add(card);
        }
        if (analysis.TryGetProperty("requirements", out var requirements) && requirements.EnumerateArray().Any(r => r.GetProperty("status").GetString() != "ready"))
            stack.Children.Add(Ui.Text("Some recipes require an additional named source. Go Back to attach the requested files.", 13, "Warning"));
        return stack;
    }
    void SelectOption(JsonElement option)
    {
        if (!option.GetProperty("supported").GetBoolean()) return;
        settings.Set("profile", option.GetProperty("profile").GetString()); plan = option.GetProperty("plan").Clone();
        if (plan.GetProperty("mode").GetString() != "convert" && plan.TryGetProperty("components", out var components) && components.ValueKind == JsonValueKind.Object)
            settings.Set("components", components.EnumerateObject().Select(c => c.Name).ToArray());
        selectedWarnings = option.GetProperty("warnings").EnumerateArray().Select(w => w.GetString()!).ToArray(); banner.Text = "";
    }
    FrameworkElement ReviewView() => Ui.Column(
        Ui.Card("Review & save", "This exact plan will be checked again before conversion. Changed source files or plan assignments require a new scan.",
            Ui.Text(settings.Text("profile").ToUpperInvariant() + "  •  " + string.Join(" + ", settings.Components), 21, "Accent"),
            Ui.Text(Description(settings.Text("profile")), 13, "Muted"),
            Field("output", "Save converted model", "save", "A new .ninfer entry file; existing outputs are never overwritten.", false),
            Ui.Column(selectedWarnings.Select(w => (UIElement)Ui.Text("• " + w, 13, "Warning")).ToArray())),
        Ui.Card("Output settings", "Optional settings for this conversion.",
            OutputNameField(), NumberField("maxFileBytes", "Maximum shard size (bytes)", 8192, long.MaxValue, false),
            Field("validateEngine", "Test text generation with engine EXE", "file", "Optional: loads the model using this engine and checks text generation. Requires compatible GPU/driver and sufficient free VRAM.", false),
            NumberField("validateTimeoutSeconds", "Engine test timeout (seconds)", 1, 86400, false)),
        new Expander { Header = "Inspect the weight plan", Content = WeightDetails() });

    FrameworkElement FinishedView() => Ui.Card("Conversion complete", phase.Text,
        Ui.Text("✓  Saved successfully", 27, "Accent"), Ui.Text(resultPath ?? "", 14),
        Ui.Text("The .conversion.json report is saved beside your model. It records encodings, artifact verification and whether engine text generation was tested.", 14, "Muted"));

    void ContextView()
    {
        var stack = new StackPanel();
        var source = discovery.ValueKind != JsonValueKind.Undefined && discovery.TryGetProperty("source", out var found) ? found : default;
        var model = Ui.Card("Model summary", "Detected from this source"); var column = (StackPanel)model.Child;
        if (source.ValueKind != JsonValueKind.Undefined)
        {
            column.Children.Add(Ui.Text(source.GetProperty("container").GetString()!, 22, "Accent"));
            if (source.TryGetProperty("architecture", out var architecture)) column.Children.Add(Ui.Text(architecture.GetString()!, 13));
            if (source.TryGetProperty("weightBytes", out var bytes)) column.Children.Add(Ui.Text(Ui.Size(bytes.GetInt64()) + " source weights", 14));
            if (source.TryGetProperty("formats", out var formats)) column.Children.Add(Ui.Text(string.Join("\n", formats.EnumerateObject().Select(f => f.Name + "  ×  " + f.Value)), 12, "Muted"));
            if (discovery.TryGetProperty("requirements", out var req)) column.Children.Add(Ui.Text(req.EnumerateArray().Count(r => r.GetProperty("status").GetString() != "ready") + " required files need attention", 12, "Muted"));
            if (discovery.TryGetProperty("issues", out var issues)) foreach (var issue in issues.EnumerateArray()) column.Children.Add(Ui.Text(issue.GetProperty("message").GetString()!, 12, "Danger"));
        }
        else column.Children.Add(Ui.Text("Choose a model to discover its architecture, encodings and required resources.", 14, "Muted"));
        stack.Children.Add(model);
        var preview = Ui.Card("Conversion preview", "Your chosen plan"); column = (StackPanel)preview.Child;
        if (plan.ValueKind == JsonValueKind.Undefined) column.Children.Add(Ui.Text("Select an option after scanning, or check your manual settings. Conversion starts only after your choice.", 14, "Muted"));
        else
        {
            column.Children.Add(Ui.Text(settings.Text("profile").ToUpperInvariant(), 24, "Accent"));
            if (plan.TryGetProperty("encodedBytes", out var size)) column.Children.Add(Ui.Text(Ui.Size(size.GetInt64()) + " encoded weights", 18));
            if (plan.TryGetProperty("formats", out var formats)) column.Children.Add(Ui.Text(string.Join("\n", formats.EnumerateObject().Select(f => f.Name + "  ×  " + f.Value)), 12, "Muted"));
            if (plan.TryGetProperty("operations", out var operations)) column.Children.Add(Ui.Text(string.Join("\n", operations.EnumerateObject().Select(f => ActionName(f.Name) + "  ×  " + f.Value)), 12, "Muted"));
            column.Children.Add(Ui.Text("Counts describe physical weight objects. Estimated weight size excludes file overhead and runtime KV/workspace memory.", 12, "Muted"));
            column.Children.Add(Ui.Text(ArtifactInput ? "Existing numeric encodings are retained without re-quantization." : "Casting or quantization may affect precision. A structural scan cannot score answer quality.", 12, ArtifactInput ? "Muted" : "Warning"));
            column.Children.Add(Ui.Text("GPU execution is checked only if you select an engine test after conversion.", 12, "Muted"));
            if (!wizard) column.Children.Add(Ui.Button("Review weight details", () => { var w = new Window { Title = "Conversion weight plan", Width = 1000, Height = 650, Owner = this, Icon = AppBranding.Icon, Content = Ui.Scroll(Ui.Card("Actual planned weights", "Encodings and actions for this conversion", WeightDetails())) }; w.ShowDialog(); }));
        }
        stack.Children.Add(preview); context.Child = Ui.Scroll(stack);
    }
    static string ActionName(string method) => method switch { "import_encoded" => "Encoded import", "cast_direct" => "Direct / cast", "grouped_absmax" or "grouped_mse" or "fp8_row_maxabs" => "Quantize", _ => method };
    FrameworkElement WeightDetails()
    {
        if (plan.ValueKind == JsonValueKind.Undefined || !plan.TryGetProperty("tensorPlan", out var tensors)) return Ui.Text("This path retains the existing artifact encodings.", 13, "Muted");
        var rows = tensors.EnumerateArray().Select(t => new { Parameter = string.Join(", ", t.GetProperty("parameters").EnumerateArray().Select(n => n.GetString())), Encoding = t.GetProperty("format").GetString(), Action = ActionName(t.GetProperty("method").GetString()!), Size = Ui.Size(t.GetProperty("encodedBytes").GetInt64()) }).ToArray();
        var grid = Ui.Table(rows, ("Parameter / physical group", "Parameter", 4), ("Encoding", "Encoding", 2), ("Action", "Action", 1.5), ("Weights", "Size", 1));
        var search = Ui.Input("", text => grid.ItemsSource = rows.Where(r => r.Parameter.Contains(text, StringComparison.OrdinalIgnoreCase) || r.Encoding!.Contains(text, StringComparison.OrdinalIgnoreCase)).ToArray());
        search.ToolTip = "Filter by parameter name or encoding";
        return Ui.Column(Ui.Text("Filter parameters or encodings", 12, "Muted"), search, grid);
    }
}
