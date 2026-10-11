using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace NInferEZ.Converter;
internal sealed partial class MainWindow : Window
{
    readonly ConversionSettings settings = new();
    readonly BackendClient backend = new();
    readonly Border workspace = new(), context = new();
    readonly TextBlock phase = Ui.Text("Ready • Offline • CPU conversion", 12, "Muted"), banner = Ui.Text("", 13, "Warning");
    readonly ProgressBar progress = new() { Height = 5, Maximum = 100, Foreground = Ui.Brush("Accent"), Background = Ui.Brush("Border"), Margin = new Thickness(0, 6, 0, 10) };
    readonly TextBox log = new() { IsReadOnly = true, AcceptsReturn = true, FontFamily = new FontFamily("Consolas"), FontSize = 11, Height = 130, VerticalScrollBarVisibility = ScrollBarVisibility.Auto, TextWrapping = TextWrapping.Wrap };
    readonly StackPanel steps = new() { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 0, 0, 18) };
    readonly Button wizardButton, manualButton, backButton, nextButton, cancelButton, revealButton;
    readonly Expander logPanel;
    bool wizard = true, busy, closeAfterCancel;
    int step;
    JsonElement discovery, analysis, plan, capabilities;
    string? resultPath;
    Task? capabilitiesTask;
    Action? commitEdits;
    readonly string[] profileIds = ["auto", "preserve", "q4", "q5", "q6", "q8", "bf16", "fp8", "recipe"];
    [DllImport("dwmapi.dll")] static extern int DwmSetWindowAttribute(IntPtr handle, int attribute, ref int value, int size);

    public MainWindow()
    {
        Style = (Style)Application.Current.FindResource(typeof(Window));
        Title = "NInferEZ Converter"; Width = 1260; Height = 930; MinWidth = 1040; MinHeight = 760; WindowStartupLocation = WindowStartupLocation.CenterScreen; Icon = AppBranding.Icon;
        var root = new Grid { Margin = new Thickness(28, 22, 28, 18) };
        foreach (var height in new[] { GridLength.Auto, GridLength.Auto, new GridLength(1, GridUnitType.Star), GridLength.Auto }) root.RowDefinitions.Add(new RowDefinition { Height = height });
        Content = root;
        var header = new Grid { Margin = new Thickness(0, 0, 0, 24) };
        header.ColumnDefinitions.Add(new ColumnDefinition()); header.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var brand = new StackPanel { Orientation = Orientation.Horizontal };
        brand.Children.Add(new Image { Source = AppBranding.Logo, Width = 48, Height = 48, Margin = new Thickness(0, 0, 14, 0) });
        brand.Children.Add(Ui.Column(Ui.Text("NInferEZ Converter", 26), Ui.Text("Local models, ready for your engine", 12, "Muted")));
        header.Children.Add(brand);
        var mode = new StackPanel { Orientation = Orientation.Horizontal, VerticalAlignment = VerticalAlignment.Center };
        wizardButton = Ui.Button("Guided wizard", () => SwitchMode(true)); manualButton = Ui.Button("Manual setup", () => SwitchMode(false));
        mode.Children.Add(wizardButton); mode.Children.Add(manualButton); Grid.SetColumn(mode, 1); header.Children.Add(mode); root.Children.Add(header);
        Grid.SetRow(steps, 1); root.Children.Add(steps);
        var middle = new Grid(); middle.ColumnDefinitions.Add(new ColumnDefinition()); middle.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(340) });
        workspace.Margin = new Thickness(0, 0, 18, 0); middle.Children.Add(workspace); Grid.SetColumn(context, 1); middle.Children.Add(context); Grid.SetRow(middle, 2); root.Children.Add(middle);
        var footer = new StackPanel { Margin = new Thickness(0, 14, 0, 0) }; footer.Children.Add(banner); footer.Children.Add(phase); footer.Children.Add(progress);
        logPanel = new Expander { Header = "Activity log", Content = log, Margin = new Thickness(0, 0, 0, 12) }; footer.Children.Add(logPanel);
        var actions = new Grid(); actions.ColumnDefinitions.Add(new ColumnDefinition()); actions.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var left = new StackPanel { Orientation = Orientation.Horizontal };
        backButton = Ui.Button("Back", Back); revealButton = Ui.Button("Show output", Reveal); revealButton.Visibility = Visibility.Collapsed;
        left.Children.Add(backButton); left.Children.Add(revealButton); actions.Children.Add(left);
        var right = new StackPanel { Orientation = Orientation.Horizontal }; cancelButton = Ui.Button("Cancel operation", CancelOperation);
        nextButton = Ui.Button("Continue", () => _ = Next(), true); right.Children.Add(cancelButton); right.Children.Add(nextButton); Grid.SetColumn(right, 1); actions.Children.Add(right); footer.Children.Add(actions); Grid.SetRow(footer, 3); root.Children.Add(footer);
        SourceInitialized += (_, _) => { int dark = 1; DwmSetWindowAttribute(new WindowInteropHelper(this).Handle, 20, ref dark, sizeof(int)); };
        Loaded += (_, _) => { capabilitiesTask = LoadCapabilities(); };
        Closing += (_, e) => { if (!busy) return; e.Cancel = true; if (MessageBox.Show(this, "Cancel this operation and close after cleanup?", "Operation running", MessageBoxButton.YesNo) == MessageBoxResult.Yes) { closeAfterCancel = true; backend.Cancel(); } };
        Render();
    }
    async Task LoadCapabilities()
    {
        try { capabilities = await new BackendClient().Run("capabilities", new(), _ => { }, Append); if (!busy && !wizard) Render(); }
        catch (Exception e) { Append(e.Message); banner.Text = "Backend unavailable. Extract the complete Portable ZIP."; }
    }
    void CancelOperation() { backend.Cancel(); phase.Text = "Cancelling and cleaning up…"; cancelButton.IsEnabled = false; }
    void Append(string text) => Dispatcher.BeginInvoke(() => { if (log.Text.Length > 90000) log.Text = log.Text[^60000..]; log.AppendText(text + Environment.NewLine); log.ScrollToEnd(); });
    void Changed(bool affectsPlan = true)
    {
        if (affectsPlan) { plan = default; analysis = default; selectedWarnings = []; resultPath = null; revealButton.Visibility = Visibility.Collapsed; }
        banner.Text = affectsPlan ? "Settings changed. Scan/check again to review the resulting plan." : ""; ContextView(); UpdateActions();
    }
    void Put(string key, object? value, bool affectsPlan = true) { settings.Set(key, value); if (key == "source") discovery = default; Changed(affectsPlan); }
    void PutMap(string key, string name, string value) { settings.SetMap(key, name, value); Changed(); }
    void SwitchMode(bool guided) { if (busy) return; commitEdits?.Invoke(); wizard = guided; if (guided) step = plan.ValueKind == JsonValueKind.Undefined ? 0 : 3; banner.Text = ""; Render(); }
    void Back() { if (busy) return; if (wizard && step > 0) { step--; Render(); } }
    void Render()
    {
        commitEdits = null;
        wizardButton.Background = Ui.Brush(wizard ? "AccentSoft" : "SurfaceAlt"); manualButton.Background = Ui.Brush(wizard ? "SurfaceAlt" : "AccentSoft");
        steps.Children.Clear();
        if (wizard)
        {
            string[] names = ["1  Source", "2  Required files", "3  Choose an option", "4  Review & save"];
            for (int i = 0; i < names.Length; i++) steps.Children.Add(new Border { Background = Ui.Brush(i == Math.Min(step, 3) ? "AccentSoft" : "Surface"), CornerRadius = new CornerRadius(8), Padding = new Thickness(14, 8, 14, 0), Margin = new Thickness(0, 0, 10, 0), Child = Ui.Text(names[i], 13, i == Math.Min(step, 3) ? "Accent" : "Muted") });
            workspace.Child = Ui.Scroll(step switch { 0 => SourceView(), 1 => ResourcesView(), 2 => OptionsView(), 3 => ReviewView(), _ => FinishedView() });
        }
        else { steps.Children.Add(Ui.Text("MANUAL SETUP  /  all conversion settings", 13, "Accent")); workspace.Child = Ui.Scroll(ManualView()); }
        ContextView(); UpdateActions();
    }
    void UpdateActions()
    {
        workspace.IsEnabled = !busy; wizardButton.IsEnabled = manualButton.IsEnabled = !busy;
        backButton.IsEnabled = !busy && wizard && step is > 0 and < 4; cancelButton.IsEnabled = busy;
        nextButton.Content = wizard ? step switch { 0 => "Check model →", 1 => "Scan conversion options →", 2 => "Review selection →", 3 => "Convert model", _ => "Convert another model" } : plan.ValueKind == JsonValueKind.Undefined ? "Check & preview" : "Convert model";
        nextButton.IsEnabled = !busy && (!wizard || step != 2 || plan.ValueKind != JsonValueKind.Undefined);
    }
    async Task Next()
    {
        if (busy) return;
        try
        {
            commitEdits?.Invoke();
            banner.Text = "";
            if (!wizard) { if (plan.ValueKind == JsonValueKind.Undefined) { await CheckManual(); Render(); } else await Convert(); return; }
            switch (step)
            {
                case 0: discovery = await Execute("probe"); step = 1; break;
                case 1:
                    analysis = await Execute("analyze"); discovery = analysis;
                    if (analysis.GetProperty("options").GetArrayLength() == 0) { step = 1; banner.Text = analysis.GetProperty("message").GetString(); }
                    else { step = 2; plan = default; } break;
                case 2: step = 3; SuggestOutput(); break;
                case 3: await Convert(); break;
                default: step = 0; plan = analysis = discovery = default; resultPath = null; settings.Values.Remove("output"); settings.Values.Remove("name"); revealButton.Visibility = Visibility.Collapsed; break;
            }
            Render();
        }
        catch (Exception e) { banner.Text = e.Message; Append(e.ToString()); UpdateActions(); }
    }
    async Task<JsonElement> Execute(string command, Dictionary<string, object?>? request = null)
    {
        busy = true; progress.IsIndeterminate = true; progress.Value = 0; UpdateActions(); phase.Text = "Preparing " + command + "…";
        try { return await backend.Run(command, request ?? settings.Request(), OnEvent, Append); }
        finally { busy = false; progress.IsIndeterminate = false; UpdateActions(); if (closeAfterCancel) Close(); }
    }
    void OnEvent(JsonElement e)
    {
        string? type = e.GetProperty("type").GetString();
        if (type == "stage") { phase.Text = e.GetProperty("message").GetString(); Append(phase.Text ?? ""); }
        if (type == "progress") { progress.IsIndeterminate = false; progress.Value = e.GetProperty("percent").GetDouble(); }
        if (type == "option") Append(e.GetProperty("profile").GetString() + ": " + (e.GetProperty("supported").GetBoolean() ? "plan available" : e.GetProperty("reason").GetString()));
        if (type == "validation") Append("Engine test passed.");
    }
    void SuggestOutput()
    {
        if (!string.IsNullOrWhiteSpace(settings.Text("output"))) return;
        var source = settings.Text("source"); var root = Directory.Exists(source) ? source : Path.GetDirectoryName(source)!;
        settings.Set("output", Path.Combine(root, (Directory.Exists(source) ? new DirectoryInfo(source).Name : Path.GetFileNameWithoutExtension(source)) + "-converted.ninfer"));
    }
    async Task CheckManual()
    {
        discovery = await Execute("probe");
        if (!discovery.GetProperty("ready").GetBoolean()) { banner.Text = "Required files are missing or invalid. See the model checklist; attach them in Model resources."; ContextView(); return; }
        plan = await Execute("inspect");
        if (plan.GetProperty("mode").GetString() != "convert" && plan.TryGetProperty("components", out var components) && components.ValueKind == JsonValueKind.Object)
            settings.Set("components", components.EnumerateObject().Select(c => c.Name).ToArray());
        SuggestOutput(); phase.Text = "Plan checked. Review the preview before converting."; ContextView(); UpdateActions();
    }
    async Task Convert()
    {
        if (plan.ValueKind == JsonValueKind.Undefined) throw new InvalidOperationException("Scan/check and select a plan first.");
        if (string.IsNullOrWhiteSpace(settings.Text("output"))) throw new InvalidOperationException("Choose an output filename.");
        var request = settings.Request();
        if (plan.TryGetProperty("planSignature", out var signature)) request["expectedPlanSignature"] = signature.GetString();
        if (discovery.TryGetProperty("fileState", out var state)) request["expectedFileState"] = state;
        var result = await Execute("convert", request); resultPath = result.GetProperty("output").GetString();
        phase.Text = result.GetProperty("runtimeVerified").GetBoolean() ? "Complete • Text generation tested in engine" : "Complete • Artifact structure verified";
        progress.Value = 100; revealButton.Visibility = Visibility.Visible; step = 4; Render();
    }
    void Reveal() { if (resultPath is not null) { var p = new ProcessStartInfo("explorer.exe") { UseShellExecute = true }; p.ArgumentList.Add("/select," + resultPath); Process.Start(p); } }
    public void Snapshot(string path)
    {
        UpdateLayout(); var dpi = VisualTreeHelper.GetDpi(this);
        var bitmap = new RenderTargetBitmap((int)Math.Ceiling(ActualWidth * dpi.DpiScaleX), (int)Math.Ceiling(ActualHeight * dpi.DpiScaleY), 96 * dpi.DpiScaleX, 96 * dpi.DpiScaleY, PixelFormats.Pbgra32); bitmap.Render(this);
        var encoder = new PngBitmapEncoder(); encoder.Frames.Add(BitmapFrame.Create(bitmap)); using var file = File.Create(Path.GetFullPath(path)); encoder.Save(file);
    }
    public async Task SmokeConvert(string source, string output, string screenshot)
    {
        if (capabilitiesTask is not null) await capabilitiesTask;
        settings.Set("source", source); settings.Set("output", output); await Next();
        await Dispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle); Snapshot(screenshot + ".resources.png");
        if (!discovery.GetProperty("ready").GetBoolean()) throw new InvalidOperationException(discovery.ToString());
        await Next();
        if (plan.ValueKind != JsonValueKind.Undefined || nextButton.IsEnabled) throw new InvalidOperationException("Wizard selected a profile without user choice.");
        SelectOption(analysis.GetProperty("options").EnumerateArray().First(o => o.GetProperty("supported").GetBoolean() && o.GetProperty("profile").GetString() == (Path.GetExtension(source) == ".ninfer" ? "preserve" : "q4")));
        Render(); await Dispatcher.InvokeAsync(() => { }, DispatcherPriority.ApplicationIdle); Snapshot(screenshot + ".options.png");
        await Next(); Snapshot(screenshot + ".review.png");
        wizard = false; Render(); Snapshot(screenshot + ".manual.png"); wizard = true; step = 3;
        await Next(); if (!File.Exists(output) || progress.Value != 100 || busy || step != 4) throw new InvalidOperationException("GUI conversion did not complete: " + banner.Text);
    }
    public async Task SmokeManual(string requestFile, string screenshot)
    {
        if (capabilitiesTask is not null) await capabilitiesTask;
        settings.Import(File.ReadAllText(requestFile)); wizard = false; Render();
        await Next(); if (plan.ValueKind == JsonValueKind.Undefined) throw new InvalidOperationException("Manual plan failed: " + banner.Text);
        Snapshot(screenshot + ".preview.png"); await Next();
        if (!File.Exists(settings.Text("output")) || progress.Value != 100 || busy) throw new InvalidOperationException("Manual conversion failed: " + banner.Text);
    }
    public async Task SmokeRecovery(string source, string resources, string output, string screenshot)
    {
        if (capabilitiesTask is not null) await capabilitiesTask;
        settings.Set("source", source); settings.Set("output", output); await Next();
        if (discovery.GetProperty("ready").GetBoolean()) throw new InvalidOperationException("Expected missing resource requirements.");
        Snapshot(screenshot + ".missing.png");
        settings.Set("resourceFolder", resources); settings.Set("configFolder", resources); Changed(); await Next();
        if (step != 2) throw new InvalidOperationException("Resource recovery failed: " + banner.Text);
        SelectOption(analysis.GetProperty("options").EnumerateArray().First(o => o.GetProperty("profile").GetString() == "q5" && o.GetProperty("supported").GetBoolean()));
        await Next(); await Next(); if (!File.Exists(output) || step != 4) throw new InvalidOperationException("Recovered wizard conversion failed: " + banner.Text);
    }
}
