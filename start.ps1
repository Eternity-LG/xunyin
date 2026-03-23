# 迅音启动脚本
Write-Host "正在启动迅音..." -ForegroundColor Green
Write-Host ""

Set-Location $PSScriptRoot

try {
    python xunyin.py
} catch {
    Write-Host "启动失败: $_" -ForegroundColor Red
    Write-Host "请检查 Python 是否安装并添加到 PATH" -ForegroundColor Yellow
    Read-Host "按 Enter 键退出"
}
