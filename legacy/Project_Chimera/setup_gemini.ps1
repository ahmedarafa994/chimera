# Setup Google Gemini API Key
Write-Host "Setting up Google Gemini API Key..." -ForegroundColor Green
$env:GOOGLE_API_KEY = "AIzaSyB_72w51dCfTKdjUnLPV--_IqAUc8N78k4"
$env:GOOGLE_MODEL = "gemini-1.5-flash"
Write-Host "Google Gemini configured!" -ForegroundColor Green
Write-Host ""
Write-Host "Starting API server..." -ForegroundColor Cyan
python api_server.py
