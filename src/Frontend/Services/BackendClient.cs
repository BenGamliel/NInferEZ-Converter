using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
namespace NInferEZ.Converter;
internal sealed class BackendClient
{
    Process? active;
    public void Cancel() { try { active?.StandardInput.WriteLine("cancel"); active?.StandardInput.Flush(); } catch (Exception e) when (e is IOException or InvalidOperationException or ObjectDisposedException) { } }
    public async Task<JsonElement> Run(string command, Dictionary<string, object?> request, Action<JsonElement> onEvent, Action<string> log)
    {
        var start = new ProcessStartInfo(Path.Combine(AppContext.BaseDirectory, "NInferEZ.Converter.Backend.exe"))
        { UseShellExecute = false, CreateNoWindow = true, RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
          StandardInputEncoding = new UTF8Encoding(false), StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
          WorkingDirectory = AppContext.BaseDirectory };
        start.ArgumentList.Add(command); active = Process.Start(start) ?? throw new IOException("Could not start the backend. Extract the complete Portable ZIP.");
        var process = active;
        try
        {
            if (command != "capabilities") { await process.StandardInput.WriteLineAsync(JsonSerializer.Serialize(request)); await process.StandardInput.FlushAsync(); }
            var diagnostics = new StringBuilder();
            var errors = Task.Run(async () => { while (await process.StandardError.ReadLineAsync() is { } line) { if (diagnostics.Length < 16000) diagnostics.AppendLine(line); log(line); } });
            JsonElement result = default; string? error = null; bool cancelled = false;
            while (await process.StandardOutput.ReadLineAsync() is { } line)
            {
                using var document = JsonDocument.Parse(line); var e = document.RootElement.Clone(); var type = e.GetProperty("type").GetString();
                if (type is "result" or "capabilities") result = e;
                if (type is "error" or "cancelled") { error = e.GetProperty("message").GetString(); cancelled = type == "cancelled"; }
                onEvent(e);
            }
            await process.WaitForExitAsync(); await errors;
            if (cancelled) throw new OperationCanceledException(error);
            if (process.ExitCode != 0 || error is not null || result.ValueKind == JsonValueKind.Undefined) throw new InvalidOperationException(error ?? diagnostics.ToString());
            return result;
        }
        finally { if (!process.HasExited) { Cancel(); process.StandardInput.Close(); await process.WaitForExitAsync(); } process.Dispose(); active = null; }
    }
}
