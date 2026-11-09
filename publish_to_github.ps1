# Publishes MyUncleVision to GitHub (public) and attaches the manuscript PDF to release v0.2.0 (v0.1.0 is already released).
# Run from PowerShell:   powershell -ExecutionPolicy Bypass -File "C:\Users\Leon\Downloads\MyUncleVision_work\repo\MyUncleVision\publish_to_github.ps1"
# Needs: git and the GitHub CLI (gh), logged in as sandlerleon (gh auth status).
# NOTE: a public repository discloses the method before the HeroX deadline. HeroX allows the repository to stay private (it needs read access only for shortlisted teams).
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
    gh repo create $Repo --public --description "Entropy-guided, label-free discovery of unusual tissue organization in whole-slide images (MyUncle maintenance-lattice front end); HeroX Kidney AI Discovery Challenge"
    if ($LASTEXITCODE -ne 0) { throw "gh repo create failed" }
}
if ((git remote) -notcontains "origin") { git remote add origin "https://github.com/$Repo.git" }
git push -u origin main
if ($LASTEXITCODE -ne 0) { throw "git push (main) failed" }
git push origin v0.1.0 --force
if ($LASTEXITCODE -ne 0) { throw "git push (tag v0.1.0) failed" }
git push origin v0.1.1 --force
if ($LASTEXITCODE -ne 0) { throw "git push (tag v0.1.1) failed" }
git push origin v0.2.0 --force
if ($LASTEXITCODE -ne 0) { throw "git push (tag v0.2.0) failed" }
if (-not (Test-Gh @("release", "view", "v0.2.0", "--repo", $Repo))) {
    gh release create v0.2.0 "docs\MyUncleVision_Manuscript_CMPB.pdf" --repo $Repo --title "v0.2.0" `
        --notes "v0.2.0: frozen settings, held-out synthetic cohort, ablations and baselines (Isolation Forest, LOF, kNN, Mahalanobis), robustness suite, multi-scale appearance with lattice smoothing as the default score, scenario analysis opt-in and exploratory, Docker end-to-end tested. Evaluated on synthetic slides only; not yet run on KPMP images. Zenodo: software 10.5281/zenodo.24066260, manuscript 10.5281/zenodo.24066256 (all versions). MyUncle core vendored unmodified."
    if ($LASTEXITCODE -ne 0) { throw "gh release create failed" }
}
git rev-parse HEAD
Write-Host "== done: https://github.com/$Repo (commit ID above)" -ForegroundColor Green
