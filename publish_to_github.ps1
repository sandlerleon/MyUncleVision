# Publishes MyUncleVision to GitHub (public) and attaches the manuscript PDF to release v0.3.0 (earlier releases are created if missing).
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
git push origin v0.2.2 --force
if ($LASTEXITCODE -ne 0) { throw "git push (tag v0.2.2) failed" }
git push origin v0.2.3 --force
if ($LASTEXITCODE -ne 0) { throw "git push (tag v0.2.3) failed" }
git push origin v0.2.4 --force
if ($LASTEXITCODE -ne 0) { throw "git push (tag v0.2.4) failed" }
git push origin v0.3.0 --force
if ($LASTEXITCODE -ne 0) { throw "git push (tag v0.3.0) failed" }
if (-not (Test-Gh @("release", "view", "v0.2.1", "--repo", $Repo))) {
    gh release create v0.2.1 --repo $Repo --title "v0.2.1" --notes "v0.2.1: reusable framework for any image collection: Python API, own encoders, demo command, docs/USAGE.md; HeroX material in docs/herox. Synthetic evaluation only."
}
if (-not (Test-Gh @("release", "view", "v0.2.2", "--repo", $Repo))) {
    gh release create v0.2.2 "docs\MyUncleVision_Manuscript_PRL.pdf" --repo $Repo --title "v0.2.2" --notes "v0.2.2: manuscript retargeted to Pattern Recognition Letters (Elsevier); real-histopathology validation stated as future work."
}
if (-not (Test-Gh @("release", "view", "v0.2.3", "--repo", $Repo))) {
    gh release create v0.2.3 --repo $Repo --title "v0.2.3" --notes "v0.2.3: bridge between MyUncle and MyUncleVision (bridge.py, docs/BRIDGE.md, bridge command)."
}
if (-not (Test-Gh @("release", "view", "v0.2.4", "--repo", $Repo))) {
    gh release create v0.2.4 "docs\MyUncleVision_Manuscript_PRL.pdf" --repo $Repo --title "v0.2.4" `
        --notes "v0.2.4: second testbed on lattices simulated with MyUncle (bridge_bench.py, results/bridge_bench_*.json; same frozen settings and baselines; mixed results), added to the Pattern Recognition Letters manuscript (Section 4.5, Table 4). Zenodo (all versions): software 10.5281/zenodo.24066260, manuscript 10.5281/zenodo.24066256. MyUncle core vendored unmodified."
    if ($LASTEXITCODE -ne 0) { throw "gh release create failed" }
}
if (-not (Test-Gh @("release", "view", "v0.3.0", "--repo", $Repo))) {
    gh release create v0.3.0 "docs\prl\main.pdf" --repo $Repo --title "v0.3.0" `
        --notes "v0.3.0: manuscript condensed and reframed for Pattern Recognition Letters (docs/prl: LaTeX, PDF, highlights, cover letter; attached PDF), ChemRxiv-framed version (docs/MyUncleVision_Manuscript_ChemRxiv.*), MyUncle bridge testbed results, framework-first CITATION. Evidence is synthetic and simulated only. Zenodo (all versions): software 10.5281/zenodo.24066260, manuscript 10.5281/zenodo.24066256."
    if ($LASTEXITCODE -ne 0) { throw "gh release create failed" }
}
git rev-parse HEAD
Write-Host "== done: https://github.com/$Repo (commit ID above)" -ForegroundColor Green
