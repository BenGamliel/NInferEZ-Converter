using Microsoft.Win32;
using System.Collections.ObjectModel;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Threading;

namespace NInferEZ.Converter;
internal sealed partial class MainWindow
{
    bool ArtifactInput => Path.GetExtension(settings.Text("source")).Equals(".ninfer", StringComparison.OrdinalIgnoreCase);
    FrameworkElement OutputNameField() => ArtifactInput ? Ui.Text("Existing NInfer metadata, components and encodings are retained. Only output location, shard size and optional engine verification apply.", 13, "Muted") : TextField("name", "Model name in output metadata", false);
    FrameworkElement TextField(string key, string label, bool affectsPlan = true, string help = "") =>
        Ui.Column(Ui.Text(label, 13, "Muted"), Ui.Input(settings.Text(key), value => Put(key, value, affectsPlan)), Ui.Text(help, 12, "Muted"));
    FrameworkElement NumberField(string key, string label, long min, long max, bool affectsPlan = true)
    {
        var input = Ui.Input(settings.Text(key), value => Put(key, long.TryParse(value, out var n) && n >= min && n <= max ? n : value, affectsPlan));
        input.ToolTip = $"Valid range: {min:N0} – {max:N0}. Leave empty to use the backend default.";
        return Ui.Column(Ui.Text(label, 13, "Muted"), input);
    }
    FrameworkElement ComponentsEditor()
    {
        var panel = new StackPanel();
        var descriptions = new Dictionary<string, string> { ["text"] = "Text — required", ["vision"] = "Vision — image/video encoder and matching processors", ["mtp"] = "MTP — compatible multi-token prediction weights", ["dflash"] = "DFlash — compatible draft model", ["dflash2"] = "DFlash2 — compatible draft/selector model" };
        foreach (var (id, text) in descriptions)
        {
            var check = new CheckBox { Content = text, IsChecked = settings.Components.Contains(id), IsEnabled = id != "text" };
            void Update() { var values = settings.Components.ToHashSet(); if (check.IsChecked == true) values.Add(id); else values.Remove(id); values.Add("text"); Put("components", values.ToArray()); }
            check.Checked += (_, _) => Update(); check.Unchecked += (_, _) => Update(); panel.Children.Add(check);
        }
        panel.Children.Add(Ui.Text("Availability is verified from actual tensors when the selected plan is prepared. Optional runtime workloads need their own validation.", 12, "Muted"));
        return panel;
    }
    static readonly string[] knownSources = ["quantized", "vision", "mtp", "dflash", "dflash2"];
    FrameworkElement SourcesEditor()
    {
        var panel = new StackPanel(); var sources = settings.Map("sources");
        foreach (var id in knownSources) panel.Children.Add(Ui.PathField(this, id == "quantized" ? "Quantized text source (optional override)" : id.ToUpperInvariant() + " companion weights", sources.GetValueOrDefault(id, ""), value => PutMap("sources", id, value), "fileOrFolder"));
        var extra = sources.Where(p => !knownSources.Contains(p.Key)).ToDictionary();
        var json = Ui.Input(JsonSerializer.Serialize(extra, new JsonSerializerOptions { WriteIndented = true }), value =>
        {
            try { var extras = JsonSerializer.Deserialize<Dictionary<string, string>>(value) ?? new(); var all = settings.Map("sources").Where(p => knownSources.Contains(p.Key)).ToDictionary(); foreach (var pair in extras) all[pair.Key] = pair.Value; Put("sources", all); }
            catch (JsonException) { settings.Set("sources", value); Changed(); }
        }, true);
        panel.Children.Add(new Expander { Header = "Additional named sources for recipes (JSON)", Content = Ui.Column(Ui.Text("Example: {\"gguf\": \"D:\\\\Models\\\\model.gguf\"}. Use names required by your recipe.", 12, "Muted"), json) });
        return panel;
    }
    FrameworkElement ResourcesEditor()
    {
        var panel = new StackPanel();
        panel.Children.Add(Field("configFolder", "Matching configuration / tokenizer folder", "folder"));
        panel.Children.Add(Field("configFile", "Individual config.json override", "file"));
        panel.Children.Add(Field("resourceFolder", "Tokenizer / frontend resource folder", "folder"));
        var resources = settings.Map("resources");
        foreach (var role in new[] { "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json" })
            panel.Children.Add(Ui.PathField(this, role + (role.Contains("preprocessor") ? " (Vision only)" : ""), resources.GetValueOrDefault(role, ""), value => PutMap("resources", role, value), "file", "Leave empty to use the matching folder/source default."));
        var shards = Ui.Input(JsonSerializer.Serialize(settings.Map("shardOverrides"), new JsonSerializerOptions { WriteIndented = true }), value => { try { Put("shardOverrides", JsonSerializer.Deserialize<Dictionary<string, string>>(value) ?? new()); } catch (JsonException) { Put("shardOverrides", value); } }, true);
        panel.Children.Add(new Expander { Header = "Weight shard replacements (JSON)", Content = Ui.Column(Ui.Text("Map an expected absolute Safetensors shard path to its actual file. The source index remains unchanged. The wizard can fill these entries using individual file pickers.", 12, "Muted"), shards) });
        return panel;
    }
    FrameworkElement TuningEditor()
    {
        var panel = new StackPanel();
        panel.Children.Add(Ui.Text("Recipe name or Python FILE.py[:function]", 13, "Muted"));
        var recipeInput = Ui.Input(settings.Text("recipe"), value => Put("recipe", value));
        var picker = new ComboBox { Margin = new Thickness(0, 0, 0, 10), ItemsSource = capabilities.ValueKind != JsonValueKind.Undefined ? capabilities.GetProperty("recipes").EnumerateArray().Select(r => r.GetString()).ToArray() : Array.Empty<string>() };
        picker.SelectionChanged += (_, _) => { if (picker.SelectedItem is string value) recipeInput.Text = value; };
        panel.Children.Add(Ui.Text("Choose a built-in recipe, or browse/enter a custom one below", 12, "Muted")); panel.Children.Add(picker); panel.Children.Add(recipeInput);
        panel.Children.Add(Ui.Button("Browse recipe…", () => { var d = new OpenFileDialog { Filter = "Python recipe|*.py|All files|*.*" }; if (d.ShowDialog(this) == true) recipeInput.Text = d.FileName; }));
        panel.Children.Add(Field("override", "Python recipe override (applied after base/proposal)", "file", "Custom Python recipes execute with normal file access. Use code you trust."));
        var device = new ComboBox { ItemsSource = new[] { "cpu", "auto" }.Concat(capabilities.ValueKind != JsonValueKind.Undefined && capabilities.GetProperty("devices").EnumerateArray().Any(d => d.GetString() == "cuda") ? new[] { "cuda" } : Array.Empty<string>()).ToArray(), SelectedItem = settings.Text("device"), Margin = new Thickness(0, 0, 0, 12) };
        device.SelectionChanged += (_, _) => Put("device", device.SelectedItem?.ToString());
        panel.Children.Add(Ui.Text("Conversion device — bundled runtime is CPU-only", 13, "Muted")); panel.Children.Add(device);
        panel.Children.Add(NumberField("rowsPerChunk", "Rows per processing chunk", 1, 65536));
        panel.Children.Add(Ui.Text("Smaller chunks reduce working memory; larger chunks may improve throughput. This does not change the selected quantization profile.", 12, "Muted"));
        var proposal = new CheckBox { Content = "Include indexed proposal head", IsChecked = settings.Flag("proposal") };
        proposal.Checked += (_, _) => Put("proposal", true); proposal.Unchecked += (_, _) => Put("proposal", false); panel.Children.Add(proposal);
        panel.Children.Add(Ui.Text("An optional shortlist head for compatible speculative generation paths. Requires enough tokenizer IDs for the requested rows.", 12, "Muted"));
        panel.Children.Add(NumberField("proposalRows", "Proposal shortlist rows", 1, 10000000));
        panel.Children.Add(Field("ranking", "Token ranking file (optional; bundled ranking is the default)", "file"));
        return Ui.Card("Recipes & processing", "Configure the bundled converter's full policy. All selected settings are included in the scan/check.", panel);
    }
    FrameworkElement ManualView()
    {
        var tools = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 8, 0, 8) };
        tools.Children.Add(Ui.Button("Import request…", () => { var d = new OpenFileDialog { Filter = "Conversion request|*.json" }; if (d.ShowDialog(this) == true) { try { settings.Import(File.ReadAllText(d.FileName)); discovery = default; Changed(); Render(); } catch (Exception e) { banner.Text = e.Message; } } }));
        tools.Children.Add(Ui.Button("Export request…", () => { commitEdits?.Invoke(); var d = new SaveFileDialog { Filter = "Conversion request|*.json", FileName = "convert.json" }; if (d.ShowDialog(this) == true) File.WriteAllText(d.FileName, settings.Export()); }));
        var profiles = new ComboBox { ItemsSource = ArtifactInput ? new[] { "auto", "preserve" } : profileIds, SelectedItem = settings.Text("profile"), Margin = new Thickness(0, 0, 0, 12) };
        var explanation = Ui.Text(Description(settings.Text("profile")), 13, "Muted");
        profiles.SelectionChanged += (_, _) => { Put("profile", profiles.SelectedItem?.ToString()); explanation.Text = Description(settings.Text("profile")); };
        if (ArtifactInput)
            return Ui.Column(Ui.Card("Existing NInfer artifact", "Auto / Preserve repack v3 or upgrade official v2 inputs. The original metadata and components are retained.", tools,
                Field("source", "Model file or checkpoint folder", "fileOrFolder"), Field("output", "Save converted model", "save", "", false),
                Ui.Text("Output profile", 13, "Muted"), profiles, OutputNameField(), NumberField("maxFileBytes", "Maximum shard size (bytes)", 8192, long.MaxValue, false)),
                Ui.Card("Engine verification", "Optional text load/warmup/prefill/decode check.", Field("validateEngine", "Matching ninfer-serve.exe", "file", "Requires a compatible GPU/driver and enough available VRAM.", false), NumberField("validateTimeoutSeconds", "Engine test timeout (seconds)", 1, 86400, false)));
        return Ui.Column(
            Ui.Card("Manual setup", "Set every option explicitly, then check the plan before converting. Settings are shared with the wizard.", tools,
                Field("source", "Model file or checkpoint folder", "fileOrFolder"), Field("output", "Save converted model", "save", "", false),
                Ui.Text("Output profile", 13, "Muted"), profiles, explanation, OutputNameField(), NumberField("maxFileBytes", "Maximum shard size (bytes)", 8192, long.MaxValue, false)),
            Ui.Card("Included components", "Text plus optional, compatible companion components.", ComponentsEditor()),
            new Expander { Header = "Model resources — configuration, tokenizer & processors", Content = Ui.Card("Model resources", "Select a folder or attach individual matching files.", ResourcesEditor()) },
            new Expander { Header = "Companion sources — quantized, Vision, MTP, DFlash", Content = Ui.Card("Named sources", "Override components or provide inputs requested by a recipe.", SourcesEditor()) },
            new Expander { Header = "Recipes, device, chunks & proposal head", Content = TuningEditor() },
            new Expander { Header = "Per-weight encoding overrides", Content = TensorOverridesEditor() },
            Ui.Card("Engine verification", "Optional text load/warmup/prefill/decode check. Structural conversion checks always run.", Field("validateEngine", "Matching ninfer-serve.exe", "file", "Requires a compatible GPU/driver and enough available VRAM.", false), NumberField("validateTimeoutSeconds", "Engine test timeout (seconds)", 1, 86400, false)));
    }
    sealed class OverrideRow
    {
        public string Selector { get; set; } = "";
        public string Format { get; set; } = "q8_g32_fp16";
        public string Method { get; set; } = "grouped_absmax";
        public string Layout { get; set; } = "auto";
        public string Policy { get; set; } = "";
        public string Parameters { get; set; } = "{}";
    }
    FrameworkElement TensorOverridesEditor()
    {
        var rows = new ObservableCollection<OverrideRow>();
        if (settings.Values.TryGetValue("tensorOverrides", out var existing))
        {
            var json = JsonSerializer.SerializeToElement(existing);
            if (json.ValueKind == JsonValueKind.Array) foreach (var e in json.EnumerateArray()) rows.Add(new OverrideRow {
                Selector = e.GetProperty("selector").GetString()!, Format = e.TryGetProperty("format", out var f) ? f.GetString()! : "bf16", Method = e.TryGetProperty("method", out var m) ? m.GetString()! : "cast_direct",
                Layout = e.TryGetProperty("layout", out var l) ? l.GetString()! : "auto", Policy = e.TryGetProperty("activationPolicy", out var p) ? p.GetString()! : "", Parameters = e.TryGetProperty("parameters", out var a) ? a.ToString() : "{}" });
        }
        var grid = new DataGrid { ItemsSource = rows, CanUserAddRows = false, CanUserDeleteRows = false, MinHeight = 160, MaxHeight = 320, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto };
        grid.Columns.Add(new DataGridTextColumn { Header = "Parameter / wildcard", Binding = new Binding("Selector"), Width = 260 });
        grid.Columns.Add(new DataGridComboBoxColumn { Header = "Format", SelectedItemBinding = new Binding("Format"), ItemsSource = capabilities.ValueKind != JsonValueKind.Undefined ? capabilities.GetProperty("formats").EnumerateArray().Select(f => f.GetString()).ToArray() : new[] { "bf16", "q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16", "fp8_e4m3fn_row_bf16" }, Width = 180 });
        grid.Columns.Add(new DataGridComboBoxColumn { Header = "Method", SelectedItemBinding = new Binding("Method"), ItemsSource = new[] { "cast_direct", "grouped_absmax", "grouped_mse", "fp8_row_maxabs", "import_encoded" }, Width = 155 });
        grid.Columns.Add(new DataGridTextColumn { Header = "Layout", Binding = new Binding("Layout"), Width = 110 });
        grid.Columns.Add(new DataGridComboBoxColumn { Header = "Activation", SelectedItemBinding = new Binding("Policy"), ItemsSource = new[] { "", "A16Only", "AllowA8", "AllowA4" }, Width = 110 });
        grid.Columns.Add(new DataGridTextColumn { Header = "Method parameters (JSON)", Binding = new Binding("Parameters"), Width = 180 });
        void Save()
        {
            try
            {
                var values = rows.Select(r => new Dictionary<string, object?> { ["selector"] = r.Selector, ["format"] = r.Format, ["method"] = r.Method, ["layout"] = r.Layout, ["activationPolicy"] = r.Policy, ["parameters"] = JsonSerializer.Deserialize<Dictionary<string, object?>>(r.Parameters) ?? new() }).ToArray();
                if (values.Length == 0 && !settings.Values.ContainsKey("tensorOverrides")) return;
                if (!settings.Values.TryGetValue("tensorOverrides", out var previous) || JsonSerializer.Serialize(previous) != JsonSerializer.Serialize(values)) Put("tensorOverrides", values);
            }
            catch (Exception e) { settings.Set("tensorOverrides", "Invalid method parameters JSON"); Changed(); banner.Text = e.Message; }
        }
        grid.CellEditEnding += (_, _) => Dispatcher.BeginInvoke(Save, DispatcherPriority.Background);
        commitEdits = () => { grid.CommitEdit(DataGridEditingUnit.Cell, true); grid.CommitEdit(DataGridEditingUnit.Row, true); Save(); };
        var actions = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 10, 0, 0) };
        actions.Children.Add(Ui.Button("Add assignment", () => { rows.Add(new OverrideRow()); Save(); }));
        actions.Children.Add(Ui.Button("Remove selected", () => { if (grid.SelectedItem is OverrideRow row) { rows.Remove(row); Save(); } }));
        return Ui.Card("Per-weight assignments", "Applied after the profile, proposal and Python override. Invalid shapes, layouts or engine formats are rejected by the planner.",
            Ui.Text("Example selector: text/layers/*/mlp/down. Use the weight preview to find exact logical names. Scroll horizontally for all fields.", 12, "Muted"), grid, actions);
    }
}
