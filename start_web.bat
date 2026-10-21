@echo off
setlocal
chcp 65001 >nul
title 文档问答助手 - 网页服务

cd /d "%~dp0"
if errorlevel 1 (
    echo 无法进入项目目录，请检查启动文件所在的位置。
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo 没有找到项目的 Python 环境，请先按 README 安装依赖。
    pause
    exit /b 1
)

if not exist "web_app\app.py" (
    echo 没有找到网页入口，请将这个启动文件保留在项目根目录。
    pause
    exit /b 1
)

echo 正在启动文档问答助手，稍后会自动打开浏览器。
echo 使用期间请保留这个窗口，结束使用时关闭窗口即可停止服务。
echo 如果浏览器没有自动打开，请访问下方显示的 Local URL。

".venv\Scripts\python.exe" -m streamlit run "web_app\app.py" --server.headless=false
set "WEB_START_EXIT_CODE=%ERRORLEVEL%"

if not "%WEB_START_EXIT_CODE%"=="0" (
    echo 网页服务已退出，请查看上面的错误信息及 README 中的运行说明。
    pause
)

exit /b %WEB_START_EXIT_CODE%
