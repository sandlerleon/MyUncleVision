# Publishes MyUncleVision v1.1.0 (Knowledge-Based Systems study) to GitHub: branch main, all tags (incl. the freeze tags kbs-freeze, kbs-freeze-2, kbs-freeze-3) and release v1.1.0.
# Run BEFORE submitting to KBS (the manuscript's Data availability statement cites these tags and the release):
#   powershell -ExecutionPolicy Bypass -File "C:\Users\Leon\Downloads\MyUncleVision_work\repo\MyUncleVision\publish_kbs_to_github.ps1"
# Needs: git and the GitHub CLI (gh), logged in as sandlerleon (gh auth status).
$ErrorActionPreference = "Continue"
$Repo = "sandlerleon/MyUncleVision"
$Dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Dir
gh auth status
if ($LASTEXITCODE -ne 0) { throw "gh is not logged in (run: gh auth login)" }
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "main") { throw "expected branch main, found $branch" }
if (git status --porcelain) { throw "working tree has uncommitted changes" }
git push origin main
if ($LASTEXITCODE -ne 0) { throw "git push (main) failed" }
git push origin --tags
if ($LASTEXITCODE -ne 0) { throw "git push (tags) failed" }
function Test-Gh([string[]]$ghArgs) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & gh @ghArgs 2>&1 | Out-Null; return ($LASTEXITCODE -eq 0) } finally { $ErrorActionPreference = $prev }
}
if (-not (Test-Gh @("release", "view", "v1.1.0", "--repo", $Repo))) {
    gh release create v1.1.0 "docs\kbs\main.pdf" --repo $Repo --title "v1.1.0: detectability study (Knowledge-Based Systems manuscript)" `
        --notes "Detectability and complementarity of feature and spatial-order anomalies in embedding lattices. Modules kbs_study, kbs_stage2, kbs_stage3, kbs_robust, kbs_props2; three prospectively frozen tests (tags kbs-freeze, kbs-freeze-2, kbs-freeze-3; freeze files results/kbs*_frozen.json). The stage-1 rule failed; the stage-2 rule max(isolation forest, Shekhar) was tested on G3 and the stationary G4. Synthetic data only."
    if ($LASTEXITCODE -ne 0) { throw "gh release create failed" }
}
Write-Host "== done: https://github.com/$Repo/releases/tag/v1.1.0" -ForegroundColor Green
