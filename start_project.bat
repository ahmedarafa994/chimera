@echo off
REM ============================================================
REM Chimera Project Startup Script
REM Starts both the Python backend and React frontend servers
REM With enhanced health checks and service verification
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   CHIMERA PROJECT LAUNCHER
echo ========================================
echo.

REM Set the project root directory
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

echo [INFO] Project root: %PROJECT_ROOT%
echo.

REM ============================================================
REM Check Prerequisites
REM ============================================================

echo [STEP 1/5] Checking prerequisites...
echo.

REM Check if Python is available
where py >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please install Python and try again.
    echo         Download from: https://python.org/downloads/
    pause
    exit /b 1
)
echo   [OK] Python found

REM Check if Node.js is available
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js not found. Please install Node.js and try again.
    echo         Download from: https://nodejs.org/
    pause
    exit /b 1
)
echo   [OK] Node.js found
echo.

REM ============================================================
REM Check Port Availability
REM ============================================================

echo [STEP 2/5] Checking port availability...
echo.

REM Check if port 8001 is in use
netstat -ano | findstr ":8001.*LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [WARNING] Port 8001 is already in use!
    echo          Backend may already be running or another service is using this port.
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "!CONTINUE!"=="y" (
        echo [INFO] Exiting. Please free port 8001 and try again.
        pause
        exit /b 1
    )
) else (
    echo   [OK] Port 8001 is available
)

REM Check if port 3001 is in use
netstat -ano | findstr ":3001.*LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [WARNING] Port 3001 is already in use!
    echo          Frontend may already be running or another service is using this port.
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "!CONTINUE!"=="y" (
        echo [INFO] Exiting. Please free port 3001 and try again.
        pause
        exit /b 1
    )
) else (
    echo   [OK] Port 3001 is available
)
echo.

REM ============================================================
REM Start Backend Server
REM ============================================================

echo [STEP 3/5] Starting Backend Server (Python/FastAPI)...
echo [INFO] Backend will run on http://localhost:8001
echo.

REM Start the backend server in a new window
start "Chimera Backend" cmd /k "cd /d %PROJECT_ROOT%backend-api && py run.py"

REM Wait for backend to initialize with health check
echo [INFO] Waiting for backend to initialize...
set /a TIMEOUT=60
set /a ELAPSED=0
set /a INTERVAL=2

:WAIT_BACKEND
if %ELAPSED% GEQ %TIMEOUT% (
    echo.
    echo [ERROR] Backend failed to start within %TIMEOUT% seconds!
    echo         Please check the backend terminal window for errors.
    echo.
    echo Possible issues:
    echo   - Missing Python dependencies: pip install -r requirements.txt
    echo   - Port 8001 already in use
    echo   - Configuration errors in backend-api/
    pause
    exit /b 1
)

REM Try to connect to health endpoint
curl -s -o nul -w "%%{http_code}" http://localhost:8001/health 2>nul | findstr "200" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo.
    echo   [OK] Backend is healthy!
    goto BACKEND_READY
)

REM Show progress
set /a ELAPSED=%ELAPSED%+%INTERVAL%
echo   Waiting... (%ELAPSED%/%TIMEOUT%s)
timeout /t %INTERVAL% /nobreak >nul
goto WAIT_BACKEND

:BACKEND_READY
echo.

REM ============================================================
REM Start Frontend Server
REM ============================================================

echo [STEP 4/5] Starting Frontend Server (Next.js)...
echo [INFO] Frontend will run on http://localhost:3001
echo.

REM Start the frontend server in a new window
start "Chimera Frontend" cmd /k "cd /d %PROJECT_ROOT%frontend && npm run dev"

REM Wait for frontend to initialize
echo [INFO] Waiting for frontend to initialize...
set /a ELAPSED=0

:WAIT_FRONTEND
if %ELAPSED% GEQ %TIMEOUT% (
    echo.
    echo [ERROR] Frontend failed to start within %TIMEOUT% seconds!
    echo         Please check the frontend terminal window for errors.
    echo.
    echo Possible issues:
    echo   - Missing Node.js dependencies: npm install
    echo   - Port 3001 already in use
    echo   - Build errors in frontend code
    pause
    exit /b 1
)

REM Try to connect to frontend
curl -s -o nul http://localhost:3001 2>nul
if %ERRORLEVEL% EQU 0 (
    echo.
    echo   [OK] Frontend is ready!
    goto FRONTEND_READY
)

REM Show progress
set /a ELAPSED=%ELAPSED%+%INTERVAL%
echo   Waiting... (%ELAPSED%/%TIMEOUT%s)
timeout /t %INTERVAL% /nobreak >nul
goto WAIT_FRONTEND

:FRONTEND_READY
echo.

REM ============================================================
REM Display Status Summary
REM ============================================================

echo [STEP 5/5] Startup Complete!
echo.
echo ========================================
echo   SERVICE STATUS
echo ========================================
echo.
echo   +------------------+---------------------------+--------+
echo   ^| Service          ^| URL                       ^| Status ^|
echo   +------------------+---------------------------+--------+
echo   ^| Backend API      ^| http://localhost:8001     ^| [OK]   ^|
echo   ^| Frontend UI      ^| http://localhost:3001     ^| [OK]   ^|
echo   ^| API Docs         ^| http://localhost:8001/docs^| [OK]   ^|
echo   +------------------+---------------------------+--------+
echo.
echo   Quick Links:
echo     - Dashboard:     http://localhost:3001/dashboard
echo     - Jailbreak:     http://localhost:3001/dashboard/jailbreak
echo     - GPTFuzz:       http://localhost:3001/dashboard/gptfuzz
echo     - Health Check:  http://localhost:8001/health
echo.
echo ========================================
echo.
echo [INFO] Both servers are running in separate windows.
echo [INFO] Press any key to open the Dashboard in your browser...
echo.

pause

REM Open the dashboard in the default browser
start http://localhost:3001/dashboard

echo.
echo [INFO] Browser opened. You can close this window.
echo [INFO] To stop the servers, close the Backend and Frontend terminal windows.
echo.

endlocal
