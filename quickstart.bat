@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo === WildVector quickstart ===
echo.

rem --- Find a supported Python (3.11-3.13) via the "py" launcher ---
set "PYLAUNCHER="
for %%V in (3.13 3.12 3.11) do (
    if not defined PYLAUNCHER (
        py -%%V --version >nul 2>&1
        if not errorlevel 1 set "PYLAUNCHER=py -%%V"
    )
)

rem --- Fall back to a bare "python" on PATH if it's already a supported version ---
if not defined PYLAUNCHER (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
    echo !PYVER! | findstr /r "^3\.1[123]\." >nul
    if not errorlevel 1 set "PYLAUNCHER=python"
)

if not defined PYLAUNCHER (
    echo Could not find Python 3.11-3.13.
    echo Install one from https://www.python.org/downloads/ and re-run this script.
    goto :error
)
echo Using Python: %PYLAUNCHER%

rem --- Create the virtual environment if it doesn't exist yet ---
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PYLAUNCHER% -m venv .venv
    if errorlevel 1 goto :error
) else (
    echo Virtual environment already exists, reusing .venv
)
set "VENV_PY=.venv\Scripts\python.exe"

rem --- Install dependencies (pip skips anything already satisfied) ---
echo.
echo Installing dependencies...
"%VENV_PY%" -m pip install --upgrade pip >nul
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

rem --- Build the local catalog, only for steps not already done ---
echo.
if not exist "data\catalog\wildvector.duckdb" (
    echo Downloading migration telemetry for the three built-in animals...
    "%VENV_PY%" scripts\refresh_catalog.py --continue-on-error
    if errorlevel 1 goto :error
) else (
    echo Catalog already built, skipping refresh_catalog.py.
)

if not exist "data\catalog\environment.parquet" (
    echo Fetching historical weather/environment data...
    "%VENV_PY%" scripts\refresh_environment.py
    if errorlevel 1 goto :error
) else (
    echo Environment data already present, skipping refresh_environment.py.
)

if not exist "data\catalog\outcome-models" (
    echo Training weather-scenario outcome models...
    "%VENV_PY%" scripts\train_population_models.py
    if errorlevel 1 goto :error
) else (
    echo Outcome models already trained, skipping train_population_models.py.
)

if not exist "data\catalog\corridors.parquet" (
    echo Precomputing population corridors...
    "%VENV_PY%" scripts\precompute_corridors.py
)

echo.
echo === Launching WildVector (close this window or press Ctrl+C to stop) ===
"%VENV_PY%" -m streamlit run app.py
goto :end

:error
echo.
echo Quickstart failed. See the output above for details.
pause
exit /b 1

:end
pause
