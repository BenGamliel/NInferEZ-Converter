using Microsoft.Win32;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Media;

namespace NInferEZ.Converter;
internal static class Ui
{
    public static Brush Brush(string key) => (Brush)Application.Current.FindResource(key);
    public static TextBlock Text(string text, double size = 14, string color = "Text") => new() { Text = text, FontSize = size, Foreground = Brush(color), Margin = new Thickness(0, 0, 0, 8), TextWrapping = TextWrapping.Wrap };
    public static Button Button(string text, Action action, bool primary = false)
    {
        var button = new Button { Content = text, Margin = new Thickness(0, 0, 8, 0) };
        if (primary) button.Style = (Style)Application.Current.FindResource("Primary");
        button.Click += (_, _) => action(); return button;
    }
    public static Border Card(string title, string caption, params UIElement[] content)
    {
        var stack = new StackPanel(); stack.Children.Add(Text(title, 20));
        if (!string.IsNullOrEmpty(caption)) stack.Children.Add(Text(caption, 13, "Muted"));
        foreach (var element in content) stack.Children.Add(element);
        return new Border { Background = Brush("Surface"), BorderBrush = Brush("Border"), BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(14), Padding = new Thickness(20), Margin = new Thickness(0, 0, 0, 14), Child = stack };
    }
    public static ScrollViewer Scroll(UIElement child) => new() { Content = child, VerticalScrollBarVisibility = ScrollBarVisibility.Auto, HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled, Padding = new Thickness(0, 0, 10, 0) };
    public static StackPanel Column(params UIElement[] children) { var panel = new StackPanel(); foreach (var child in children) panel.Children.Add(child); return panel; }
    public static TextBox Input(string value, Action<string> changed, bool multiline = false)
    {
        var box = new TextBox { Text = value, AcceptsReturn = multiline, TextWrapping = multiline ? TextWrapping.Wrap : TextWrapping.NoWrap, MinHeight = multiline ? 110 : 38, Margin = new Thickness(0, 0, 0, 12), VerticalScrollBarVisibility = multiline ? ScrollBarVisibility.Auto : ScrollBarVisibility.Hidden };
        box.TextChanged += (_, _) => changed(box.Text); return box;
    }
    public static FrameworkElement PathField(Window owner, string label, string value, Action<string> changed, string kind = "file", string help = "")
    {
        var row = new Grid { Margin = new Thickness(0, 0, 0, 12) };
        row.ColumnDefinitions.Add(new ColumnDefinition()); row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var box = Input(value, changed); box.Margin = new Thickness(0, 0, 8, 0); box.ToolTip = help;
        box.AllowDrop = true;
        box.PreviewDragOver += (_, e) => { e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.Copy : DragDropEffects.None; e.Handled = true; };
        box.Drop += (_, e) => { if (e.Data.GetData(DataFormats.FileDrop) is string[] files && files.Length > 0) box.Text = files[0]; };
        row.Children.Add(box);
        var buttons = new StackPanel { Orientation = Orientation.Horizontal };
        void Pick(bool folder)
        {
            if (folder) { var dialog = new OpenFolderDialog { Title = label }; if (dialog.ShowDialog(owner) == true) box.Text = dialog.FolderName; }
            else { var dialog = new OpenFileDialog { Title = label, Filter = kind == "save" ? "NInfer model|*.ninfer" : "All files|*.*" }; if (dialog.ShowDialog(owner) == true) box.Text = dialog.FileName; }
        }
        if (kind == "save") buttons.Children.Add(Button("Save as…", () => { var d = new SaveFileDialog { Title = label, Filter = "NInfer model|*.ninfer", DefaultExt = "ninfer", AddExtension = true, FileName = Path.GetFileName(box.Text) }; if (d.ShowDialog(owner) == true) box.Text = d.FileName; }));
        else { buttons.Children.Add(Button(kind == "folder" ? "Folder…" : "File…", () => Pick(kind == "folder"))); if (kind == "fileOrFolder") buttons.Children.Add(Button("Folder…", () => Pick(true))); }
        Grid.SetColumn(buttons, 1); row.Children.Add(buttons);
        var stack = Column(Text(label, 13, "Muted"), row); if (!string.IsNullOrEmpty(help)) stack.Children.Add(Text(help, 12, "Muted")); return stack;
    }
    public static DataGrid Table(IEnumerable<object> rows, params (string title, string path, double width)[] columns)
    {
        var grid = new DataGrid { ItemsSource = rows, IsReadOnly = true, CanUserAddRows = false, CanUserDeleteRows = false, SelectionMode = DataGridSelectionMode.Single, SelectionUnit = DataGridSelectionUnit.FullRow, MinHeight = 110, MaxHeight = 340 };
        foreach (var (title, path, width) in columns) grid.Columns.Add(new DataGridTextColumn { Header = title, Binding = new Binding(path), Width = new DataGridLength(width, DataGridLengthUnitType.Star) });
        return grid;
    }
    public static string Size(long bytes) => bytes >= 1073741824 ? $"{bytes / 1073741824d:F2} GiB" : bytes >= 1048576 ? $"{bytes / 1048576d:F1} MiB" : $"{bytes / 1024d:F1} KiB";
}
