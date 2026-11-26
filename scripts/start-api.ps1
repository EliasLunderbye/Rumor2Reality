$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'The project virtual environment is missing. See docs/DEVELOPMENT.md.'
}
if (-not $env:SEMANTIC_SCHOLAR_API_KEY) {
    $secret = Read-Host 'Semantic Scholar API key' -AsSecureString
    $env:SEMANTIC_SCHOLAR_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
}
if (-not $env:SEMANTIC_SCHOLAR_API_KEY) {
    throw 'A Semantic Scholar API key is required by this launcher.'
}
Push-Location -LiteralPath $project
$previousIndex = $env:RUMOR2REALITY_VECTOR_STORE
try {
    Remove-Item Env:RUMOR2REALITY_VECTOR_STORE -ErrorAction SilentlyContinue
    & $python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
    if ($LASTEXITCODE -ne 0) { throw "Uvicorn exited with code $LASTEXITCODE." }
}
finally {
    $env:RUMOR2REALITY_VECTOR_STORE = $previousIndex
    Pop-Location
}
