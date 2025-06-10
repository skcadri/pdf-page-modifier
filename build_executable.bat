@echo off
title Building PDF Page Modifier Executable
echo ========================================
echo Building PDF Page Modifier Executable
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.7 or higher from https://python.org
    pause
    exit /b 1
)

echo Installing/updating PyInstaller...
pip install pyinstaller

echo.
echo Building executable...
echo This may take a few minutes...
echo.

REM Clean previous builds
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
if exist "PDF_Page_Modifier.spec" del "PDF_Page_Modifier.spec"

REM Build the executable
pyinstaller --onefile --windowed --name "PDF_Page_Modifier" pdf_page_modifier.py

if exist "dist\PDF_Page_Modifier.exe" (
    echo.
    echo ✅ Build successful!
    echo.
    echo Executable location: dist\PDF_Page_Modifier.exe
    echo File size: 
    for %%A in ("dist\PDF_Page_Modifier.exe") do echo %%~zA bytes
    echo.
    echo You can now distribute the dist\PDF_Page_Modifier.exe file
    echo Users can run it without installing Python or any dependencies!
    echo.
    
    REM Optional: Test the executable
    choice /M "Would you like to test the executable now"
    if errorlevel 2 goto :end
    echo.
    echo Launching PDF_Page_Modifier.exe...
    start "" "dist\PDF_Page_Modifier.exe"
) else (
    echo.
    echo ❌ Build failed!
    echo Check the output above for errors.
)

:end
echo.
pause 