@echo off
REM Arranca la API en local + túnel público Cloudflare en una sola ventana.
REM Te da una URL https aleatoria tipo https://xxx-yyy.trycloudflare.com
REM
REM Requisitos (instalar UNA vez):
REM   winget install Cloudflare.cloudflared
REM
REM Uso:
REM   doble click en este .bat
REM
REM La URL aparece en consola. Cópiala y mete en BolsilloIaClient.kt -> baseUrl.
REM Cierra la ventana para parar todo.

cd /d "%~dp0"

echo === arrancando API en localhost:8123 ===
start /B py main.py

echo.
echo === esperando 5s a que arranque ===
timeout /t 5 /nobreak >nul

echo.
echo === abriendo tunel Cloudflare ===
echo (la URL publica saldra abajo, copiala)
echo.
cloudflared tunnel --url http://localhost:8123

REM Cuando cierres cloudflared, mata uvicorn:
taskkill /F /IM python.exe >nul 2>&1
