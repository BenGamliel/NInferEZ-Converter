# NInferEZ Converter

חלצו את כל החבילה והפעילו **NInferEZ.Converter.exe**.
בחרו **Guided wizard**: בחרו מודל, השלימו קבצים חסרים, סרקו את האפשרויות,
בחרו פרופיל לאחר עיון בגודל ובהסבר האיכות, ואז בחרו יעד והמירו.
**Manual setup** מאפשר להגדיר משאבים, רכיבים, מתכונים והגדרות לכל משקל.
הוראות בעברית: [START-HE.md](docs/START-HE.md).

Extract the complete package and open **NInferEZ.Converter.exe**.
The **Guided wizard** scans a model, requests missing files and presents supported
conversion plans with size estimates and precision notes. Select a plan, review
it and choose a new output path before converting. **Manual setup** exposes
resources, companion components, recipes and per-weight settings.

| Location | Contents |
| --- | --- |
| `NInferEZ.Converter.exe` | Desktop interface |
| `NInferEZ.Converter.Backend.exe` | Independent command-line / JSON backend |
| `app/` | Converter code, engine adapters and inspector |
| `runtime/` | App-local Python and CPU conversion dependencies |
| `docs/` | Usage, backend API, support details and validation evidence |
| `licenses/` | Third-party redistribution notices |

Keep the complete layout together. Conversion works offline without installing
Python or .NET. The .NET runtime is bundled inside the two EXEs; Windows may cache
their native runtime files in the current user's temporary folder on first launch.

- [Full documentation](docs/README.md)
- [Backend JSON API](docs/BACKEND.md)
- [Supported sources and validation](docs/QA.md)
- Distribution hashes and runtime metadata: `docs/distribution/`
