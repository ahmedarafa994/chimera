@echo off
echo Setting up Google Gemini API Key...
set GOOGLE_API_KEY=AIzaSyB_72w51dCfTKdjUnLPV--_IqAUc8N78k4
set GOOGLE_MODEL=gemini-1.5-flash
echo Google Gemini configured!
echo.
echo Starting API server...
python api_server.py
