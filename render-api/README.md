# Bolsillo IA Relay

API privada con cadena de fallback entre varios proveedores.
**Todas las claves van en variables de entorno / `secrets.properties` — nunca en el código.**

## Secrets

1. Copia `../secrets.defaults.properties` → `../secrets.properties` (gitignored).
2. Rellena las claves reales.
3. En producción (Render / Railway / Fly / GitHub Actions) define las mismas variables de entorno.

Variables mínimas:
- `API_KEY_CLIENTE` — auth del cliente Android (`X-API-Key`)
- `EDITOR_KEY` — auth del panel editor
- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` — proveedores
- `SUPABASE_URL`, `SUPABASE_ANON` — persistencia (opcional)
- `APPWRITE_*` — fallback de persistencia (opcional)

## Opción A — Túnel Cloudflare desde tu PC

1. `winget install Cloudflare.cloudflared`
2. Doble click en `tunel-local.bat`
3. Copia la URL (`https://xxxx.trycloudflare.com`) a `RELAY_URL_*` en `secrets.properties` / UI

## Opción B — Fly.io

```
cd render-api
flyctl launch --no-deploy --copy-config --name bolsillo-ia-<unico>
flyctl secrets set API_KEY_CLIENTE=... ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=...
flyctl deploy
```

## Local

```
cd render-api
# asegúrate de tener ../secrets.properties o un .env
py -m pip install -r requirements.txt
py main.py
```

## Endpoints

Todos requieren header `X-API-Key: <API_KEY_CLIENTE>` (excepto `/` y `/health`).

- `GET  /` → ping + lista de proveedores
- `GET  /health` → estado + jobs en memoria
- `POST /ask` → encola job
- `GET /result/{job_id}` → estado actual
- `DELETE /result/{job_id}` → libera memoria
