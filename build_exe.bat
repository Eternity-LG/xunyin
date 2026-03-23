@echo off
chcp 65001 >nul
echo 正在打包迅音...
echo.

:: 清理旧文件
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

:: 打包
pyinstaller xunyin.spec --clean

:: 复制必要文件
echo.
echo 复制模型文件...
if not exist "dist\迅音\models" mkdir "dist\迅音\models"

echo.
echo 打包完成！
echo 输出目录: dist\迅音\
echo.
pause
