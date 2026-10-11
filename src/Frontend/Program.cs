using System.IO;
using System.Windows;
using System.Windows.Media.Imaging;

namespace NInferEZ.Converter;

internal static class AppBranding
{
    public static readonly BitmapSource Logo = Load("Branding.Logo");
    public static readonly BitmapSource Icon = Load("Branding.Icon");
    static BitmapSource Load(string name)
    {
        using var stream = typeof(AppBranding).Assembly.GetManifestResourceStream(name)!;
        var decoder = BitmapDecoder.Create(stream, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad);
        var image = decoder.Frames.OrderByDescending(f => f.PixelWidth).First(); image.Freeze(); return image;
    }
}
internal static class Program
{
    [STAThread]
    static void Main(string[] args)
    {
        var app = new Application();
        app.Resources.MergedDictionaries.Add(new ResourceDictionary { Source = new Uri("/NInferEZ.Converter;component/UI/Theme.xaml", UriKind.Relative) });
        var window = new MainWindow();
        app.DispatcherUnhandledException += (_, e) => { MessageBox.Show(window, e.Exception.Message, "NInferEZ Converter"); e.Handled = true; };
        if (args.Length == 2 && args[0] == "--smoke-ui" || args.Length == 4 && args[0] == "--smoke-convert" || args.Length == 3 && args[0] == "--smoke-request" || args.Length == 5 && args[0] == "--smoke-recovery")
        {
            window.ContentRendered += async (_, _) =>
            {
                try { if (args[0] == "--smoke-convert") await window.SmokeConvert(args[1], args[2], args[3]); if (args[0] == "--smoke-request") await window.SmokeManual(args[1], args[2]); if (args[0] == "--smoke-recovery") await window.SmokeRecovery(args[1], args[2], args[3], args[4]); window.Snapshot(args[^1]); }
                catch (Exception e) { File.WriteAllText(args[^1] + ".error.txt", e.ToString()); Environment.ExitCode = 1; }
                finally { window.Close(); }
            };
        }
        app.Run(window);
    }
}
