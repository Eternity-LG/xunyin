@echo off
chcp 65001 >nul
echo 正在启动迅音...
echo.
cd /d "%~dp0"
python xunyin.py
if errorlevel 1 (
    echo.
    echo 启动失败，请检查 Python 是否安装
    pause
)
