# Publishes MyUncleVision to GitHub (public) and attaches the manuscript PDF to release v0.2.1 (v0.1.0 and v0.2.0 may already be released).
# Run from PowerShell:   powershell -ExecutionPolicy Bypass -File "C:\Users\Leon\Downloads\MyUncleVision_work\repo\MyUncleVision\publish_to_github.ps1"
# Needs: git and the GitHub CLI (gh), logged in as sandlerleon (gh auth status).
# NOTE: the repository is public. The HeroX pack (docs/herox) is a DRAFT that stays in the repo only as an example application.
#       If you prefer that, change --public to --private below.
$ErrorActionPreference = "Continue"
$Repo = "sandlerleon/MyUncleVision"
$Dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Dir
gh auth status
if ($LASTEXITCODE -ne 0) { throw "gh is not logged in (run: gh auth login)" }
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "main") { throw "expected branch main, found $branch" }
if (git status --porcelain) { throw "working tree has uncommitted changes" }
function Test-Gh([string[]]$ghArgs) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & gh @ghArgs 2>&1 | Out-Null; return ($LASTEXITCODE -eq 0) } finally { $ErrorActionPreference = $prev }
}
if (-not (Test-Gh @("repo", "view", $Repo))) {
    gh repo create $Repo --public --description "Reusable lattice-based framework for unsupervised anomaly discovery in images (histopathology first); computer-vision sister of MyUncle"
    if ($LASTEXITCODE -ne 0) { throw "gh repo create failed" }
}
gh repo edit $Repo --description "Reusable lattice-based framework for unsupervised anomaly discovery in images (histopathology first); computer-vision sister of MyUncle" 2>&1 | Out-Null
if ((git remote) -notcontains "origin") { git remote add origin "https://github.com/$Repo.git" }
git push -u origin main
if ($LASTEXITCODE -ne 0) { throw "git push (main) failed" }
git push origin v0.1.0 --force
if ($LASTEXITCODE -ne 0) { throw "git push (tag v0.1.0) failed" }
git push origin v0.1.1 --force
if ($LASTEXITCODE -ne 0) { throw "git push (tag v0.1.1) failed" }
git push origin v0.2.0 --force
if ($LASTEXITCODE -ne 0) { throw "git push (tag v0.2.0) failed" }
git push origin v0.2.1 --force
if ($LASTEXITCODE -ne 0) { throw "git push (tag v0.2.1) failed" }
if (-not (Test-Gh @("release", "view", "v0.2.0", "--repo", $Repo))) {
    gh release create v0.2.0 --repo $Repo --title "v0.2.0" `
        --notes "v0.2.0: frozen settings, held-out synthetic cohort, ablations and baselines, robustness suite, lattice smoothing as default score, Docker end-to-end tested. Synthetic evaluation only."
}
if (-not (Test-Gh @("release", "view", "v0.2.1", "--repo", $Repo))) {
    gh release create v0.2.1 "docs\MyUncleVision_Manuscript_CMPB.pdf" --repo $Repo --title "v0.2.1" `
        --notes "v0.2.1: MyUncleVision as a reusable framework for any image collection (not tied to one challenge): Python API (discover_images, discover_arrays), your own encoder (CallableEncoder / --encoder-module), demo command, docs/USAGE.md; HeroX material moved to docs/herox. Evaluated on synthetic slides only; real-histopathology validation is planned future work. Zenodo (all versions): software 10.5281/zenodo.24066260, manuscript 10.5281/zenodo.24066256. MyUncle core vendored unmodified."
    if ($LASTEXITCODE -ne 0) { throw "gh release create failed" }
}
git rev-parse HEAD
Write-Host "== done: https://github.com/$Repo (commit ID above)" -ForegroundColor Green
