using System.Diagnostics;
using System.Text;
using System.Text.Json;

Console.InputEncoding = new UTF8Encoding(false);
Console.OutputEncoding = new UTF8Encoding(false);
var root = AppContext.BaseDirectory;
var python = Path.Combine(root, "runtime", "python", "python.exe");
var backend = Path.Combine(root, "app", "backend", "app.py");
if (!File.Exists(python) || !File.Exists(backend))
{
    Console.WriteLine(JsonSerializer.Serialize(new { type = "error", success = false, message = "Extract the complete Portable ZIP before running the backend." }));
    return 1;
}
var start = new ProcessStartInfo(python)
{
    UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = Environment.CurrentDirectory,
    RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
    StandardInputEncoding = new UTF8Encoding(false), StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8
};
foreach (var value in new[] { "-I", "-B", "-u", backend }) start.ArgumentList.Add(value);
foreach (var value in args.Length == 0 ? new[] { "--help" } : args) start.ArgumentList.Add(value);
// Ignore global Python paths and user packages. All dependencies are app-local.
start.Environment["PYTHONUTF8"] = "1";
using var child = Process.Start(start)!;
var output = Task.Run(async () => { while (await child.StandardOutput.ReadLineAsync() is { } line) Console.WriteLine(line); });
var error = Task.Run(async () => { while (await child.StandardError.ReadLineAsync() is { } line) Console.Error.WriteLine(line); });
_ = Task.Run(async () =>
{
    try
    {
        while (await Console.In.ReadLineAsync() is { } line)
        {
            await child.StandardInput.WriteLineAsync(line);
            await child.StandardInput.FlushAsync();
        }
        child.StandardInput.Close();
    }
    catch (Exception e) when (e is IOException or ObjectDisposedException or InvalidOperationException) { }
});
Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    try { child.StandardInput.WriteLine("cancel"); child.StandardInput.Flush(); } catch (IOException) { }
};
await child.WaitForExitAsync();
await Task.WhenAll(output, error);
return child.ExitCode;
