@echo off
REM Activation script for Windows
REM Usage: activate.bat

echo Activating AdventureWorks virtual environment...
call venv\Scripts\activate.bat

echo.
echo Virtual environment activated!
echo.
echo Quick commands:
echo   - python verify_setup.py      (Verify installation)
echo   - jupyter notebook             (Start Jupyter)
echo   - pip list                     (List installed packages)
echo   - deactivate                   (Exit virtual environment)
echo.
