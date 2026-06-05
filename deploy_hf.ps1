# deploy_hf.ps1
# Automates pushing your Deepfake Detection application to Hugging Face Spaces.

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "   Hugging Face Spaces Deployment Automation   " -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

$username = Read-Host "Enter your Hugging Face username"
$spaceName = Read-Host "Enter your Hugging Face Space name (e.g. sentry-app)"

if ([string]::IsNullOrEmpty($username) -or [string]::IsNullOrEmpty($spaceName)) {
    Write-Host "Error: Username and Space name cannot be empty." -ForegroundColor Red
    Exit
}

$remoteUrl = "https://huggingface.co/spaces/$username/$spaceName"

Write-Host ""
Write-Host "1. Removing existing 'hf' remote if present..." -ForegroundColor Yellow
git remote remove hf 2>$null

Write-Host "2. Adding git remote 'hf' pointing to: $remoteUrl" -ForegroundColor Yellow
git remote add hf $remoteUrl

Write-Host ""
Write-Host "----------------------------------------------" -ForegroundColor Cyan
Write-Host "IMPORTANT CREDENTIALS NOTE:" -ForegroundColor Yellow
Write-Host "When git prompts you for authentication:" -ForegroundColor Yellow
Write-Host " - Username: Use your Hugging Face username ($username)" -ForegroundColor Gray
Write-Host " - Password: Use your Hugging Face Access Token (with 'WRITE' permission)." -ForegroundColor Gray
Write-Host "             Get one at: https://huggingface.co/settings/tokens" -ForegroundColor Gray
Write-Host "----------------------------------------------" -ForegroundColor Cyan
Write-Host ""

$confirm = Read-Host "Do you want to proceed and push to Hugging Face now? (y/n)"
if ($confirm -ne "y") {
    Write-Host "Deployment aborted." -ForegroundColor Red
    Exit
}

Write-Host ""
Write-Host "3. Pushing local 'master' branch to remote Hugging Face 'main' branch..." -ForegroundColor Yellow
Write-Host "This will upload your Git LFS model weights (~680MB) and code files." -ForegroundColor Yellow
Write-Host "This might take several minutes depending on your upload speed..." -ForegroundColor Yellow
Write-Host ""

git push -f hf master:main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host "SUCCESS: Code and models pushed successfully!" -ForegroundColor Green
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host "Your Space is now building. Visit and monitor the progress at:" -ForegroundColor Yellow
    Write-Host "https://huggingface.co/spaces/$username/$spaceName" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "ERROR: Push failed. Please check your credentials / network connection." -ForegroundColor Red
}
