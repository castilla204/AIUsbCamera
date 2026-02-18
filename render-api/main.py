import sys
import subprocess
import importlib
import re
import os

# Carga secrets locales (.env / secrets.properties) sin hardcodear en el repo.
try:
    from dotenv import load_dotenv
    from pathlib import Path as _SecretsPath
    _here = _SecretsPath(__file__).resolve().parent
    load_dotenv(_here / ".env")
    load_dotenv(_here.parent / "secrets.properties")
    load_dotenv(_here.parent / ".env")
except Exception:
    pass
import gc
import html as html_lib
import uuid
import time
import threading
import traceback
import base64
import binascii
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple

# ─── Reducir stack size por thread (CRÍTICO en Render free 512MB) ───────────
# Python por defecto crea cada thread con 8MB de stack virtual. Con ~25 threads
# concurrentes (9 OCRs + 6 analyzers + 5 daemons + fuse_bg + tavily + workers
# de uvicorn), el reserve total es ~200MB — sobre 512MB de RAM eso AGOTABA el
# free tier de Render durante un pipeline VIDEO (OOM kill confirmado en prod).
#
# 1MB es el mínimo seguro:
#   • POSIX permite >= 16KB, pero cv2/numpy/optical flow Farneback pueden
#     usar varios cientos de KB de stack en operaciones recursivas internas.
#   • PaddleOCR y libs grandes lo requieren para inicialización.
#   • Glibc reserva PTHREAD_STACK_MIN (16384) absoluto; CPython tiene su propio
#     mínimo (~32KB para el frame del intérprete).
#   • 1MB = 7MB ahorrados por thread × 25 threads = ~175MB liberados.
#
# IMPORTANTE: esta llamada AFECTA SOLO a threads creados DESPUÉS. Tiene que ir
# ANTES de cualquier threading.Thread() de los daemons.
_DEFAULT_THREAD_STACK_BYTES = int(os.environ.get("THREAD_STACK_BYTES", str(1024 * 1024)))
try:
    threading.stack_size(_DEFAULT_THREAD_STACK_BYTES)
except (ValueError, RuntimeError) as _stack_exc:
    # ValueError: plataforma no acepta ese tamaño. RuntimeError: stack_size
    # no soportado por esta build de Python. En ambos casos seguimos con el
    # default del SO — operación funcional pero más memoria.
    print(f"[bootstrap] WARN: no se pudo bajar thread stack a {_DEFAULT_THREAD_STACK_BYTES} bytes: {_stack_exc}",
          flush=True)

# OpenCV en Render con 1 vCPU: limitar a 1 hilo interno. Con un solo core, el
# multi-threading de cv2 (Farneback, resize, Canny…) solo añade overhead de
# scheduling y, con varios jobs de vídeo a la vez, sus hilos internos pelean por
# el único core. setNumThreads(1) lo hace determinista y evita saturarlo. cv2 se
# importa lazy en el pipeline; esto adelanta la carga una sola vez y fija el
# límite global. Si cv2 no está instalado, el pipeline degrada (no es fatal).
try:
    import cv2 as _cv2_boot
    _cv2_boot.setNumThreads(1)
except Exception as _cv2_exc:
    print(f"[bootstrap] cv2.setNumThreads(1) no aplicado: {_cv2_exc}", flush=True)

# ─── Bootstrap de dependencias ───────────────────────────────────────────────
# Dos categorías: críticas (sin ellas el server NO puede arrancar) y opcionales
# (mejoran calidad pero hay fallback). Si la instalación de una OPCIONAL falla
# (sin internet, mirror caído, sandbox sin pip), seguimos arrancando: el server
# degrada elegantemente. Solo las CRÍTICAS abortan el arranque.
_REQUIRED_CRITICAL = [
    ("fastapi",   "fastapi==0.115.0"),
    ("uvicorn",   "uvicorn[standard]==0.30.6"),
    ("anthropic", "anthropic>=0.52.0"),
    ("pydantic",  "pydantic==2.9.2"),
    ("httpx",     "httpx==0.27.2"),
]
_REQUIRED_OPTIONAL = [
    # rapidfuzz: similitud léxica multi-señal (token_set_ratio, partial_ratio,
    # token_sort_ratio). Usado por _similarity() en fusión OCR. Si falta,
    # _similarity() cae a difflib (igual que antes). Wheel binario ~1MB.
    ("rapidfuzz", "rapidfuzz>=3.0.0"),
    # opencv-python-headless: para extraer el frame medio del video MP4 y
    # pasarlo a Claude (Anthropic) como OCR. Anthropic NO soporta video nativo,
    # solo imágenes; con cv2 el servidor extrae el frame automáticamente sin
    # que el cliente tenga que mandar image_b64. Wheel binario ~50MB en headless
    # (sin GUI de OpenCV — perfecto para servidor). Si falla la instalación o
    # el wheel no está disponible para esta arquitectura, Claude OCR queda
    # fuera de la fusión y los otros 4 proveedores siguen votando.
    ("cv2",       "opencv-python-headless>=4.9.0"),
]

def _asegurar_deps():
    # 1) Críticas: si faltan e instalación falla → reraise (no se puede arrancar)
    faltan_criticas = []
    for mod, pkg in _REQUIRED_CRITICAL:
        try:
            importlib.import_module(mod)
        except ImportError:
            faltan_criticas.append(pkg)
    if faltan_criticas:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", *faltan_criticas],
                timeout=180,
            )
            importlib.invalidate_caches()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            print(f"[bootstrap] FATAL: deps críticas no instalables: {e}", flush=True)
            raise

    # 2) Opcionales: si faltan, intentamos instalar pero NO abortamos si falla.
    faltan_opt = []
    for mod, pkg in _REQUIRED_OPTIONAL:
        try:
            importlib.import_module(mod)
        except ImportError:
            faltan_opt.append((mod, pkg))
    if faltan_opt:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 *(pkg for _, pkg in faltan_opt)],
                timeout=120,
            )
            importlib.invalidate_caches()
        except Exception as e:
            # No fatal: anotar y seguir. Las funciones que usen estas deps
            # deben tener fallback via try/except ImportError.
            mods = ", ".join(m for m, _ in faltan_opt)
            print(f"[bootstrap] WARN: deps opcionales sin instalar ({mods}): {e}", flush=True)

_asegurar_deps()

import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="[relay] %(message)s")
logger = logging.getLogger("relay")

# ─── Log persistente de errores ──────────────────────────────────────────────
# Capturamos cada log de nivel ERROR/CRITICAL en una cola circular + disco.
# Permite ver desde el panel qué falló incluso después de un reinicio.
from collections import deque

ERROR_LOG_FILE = Path(os.environ.get("ERROR_LOG_FILE", "relay_errors.json"))
ERROR_LOG_MAX  = int(os.environ.get("ERROR_LOG_MAX", "200"))
_ERROR_LOG: deque = deque(maxlen=ERROR_LOG_MAX)
_ERROR_LOG_LOCK = threading.Lock()
_ERROR_LOG_DIRTY = threading.Event()


class _ErrorCapturer(logging.Handler):
    """Captura logs de nivel ERROR+ en deque + flag dirty para persistir.

    Deduplica: si el mismo mensaje aparece dentro de DEDUP_WINDOW segundos,
    se incrementa `count` del entry existente en vez de crear uno nuevo.
    Pensado para el caso "Supabase caída 2h" en el que el daemon sync (cada
    60s) loguea el mismo error 120 veces — sin dedup llenaríamos el deque
    de 200 con entries idénticos y enterraríamos errores genuinos distintos.
    """
    DEDUP_WINDOW_S = 300       # 5 min: dentro de esto, mismo msg = bump count
    # B18: lookback completo del deque (no solo 30) para que un mensaje
    # repetido tras una ráfaga de errores DISTINTOS siga deduplicándose. Antes
    # con LOOKBACK=30, un patrón "30 errores únicos + repeat del primero" no
    # detectaba el duplicado porque ya había salido del rango. Resultado: el
    # deque de 200 se llenaba de pares (error_orig, repetición) en vez de
    # acumular contador.
    DEDUP_LOOKBACK = ERROR_LOG_MAX

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)[:500]
            now = int(record.created)
            with _ERROR_LOG_LOCK:
                # Mirar TODAS las entries de la ventana DEDUP_WINDOW_S — si una
                # tiene el mismo msg, incrementamos su contador.
                try:
                    snapshot = list(_ERROR_LOG)
                    for entry in reversed(snapshot):
                        last_t = entry.get("last_t", entry.get("t", 0))
                        if (now - last_t) >= self.DEDUP_WINDOW_S:
                            # Las entries están en orden temporal; en cuanto una
                            # cae fuera de la ventana, todas las anteriores también.
                            break
                        if entry.get("msg") == msg:
                            entry["count"]  = entry.get("count", 1) + 1
                            entry["last_t"] = now
                            _ERROR_LOG_DIRTY.set()
                            return
                except Exception:
                    pass  # si el dedup falla, seguimos al insert normal
                _ERROR_LOG.append({
                    "t":      now,
                    "last_t": now,
                    "level":  record.levelname,
                    "msg":    msg,
                    "thread": record.threadName,
                    "count":  1,
                })
            _ERROR_LOG_DIRTY.set()
        except Exception:
            pass

_err_handler = _ErrorCapturer()
_err_handler.setLevel(logging.ERROR)
_err_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(_err_handler)


# ─── Capturer GLOBAL para excepciones que escapen de cualquier camino ────────
# 3 puentes adicionales para asegurar 100% visibilidad de errores en /api/errors:
#
#   1. threading.excepthook (Python 3.8+): cualquier excepción NO capturada en
#      un thread daemon. Sin esto, los hilos OCR/analyzer/daemon morían
#      silenciosamente y el operador NO veía la causa.
#
#   2. sys.excepthook: excepciones del thread MAIN que no se capturan (raras,
#      pero posibles si algún @app.on_event explota durante startup).
#
#   3. warnings.warn capturado al logger: warnings de Python (DeprecationWarning,
#      ResourceWarning, etc.) que indiquen bugs (sockets sin cerrar, etc.).
#
# Adjuntamos también el handler al logger "uvicorn.error" porque uvicorn NO
# propaga al root logger por defecto desde 0.30.
def _setup_extra_capturers():
    # 1) Threading excepthook
    def _thread_excepthook(args):
        # args: ExceptHookArgs con exc_type, exc_value, exc_traceback, thread
        try:
            t_name = getattr(args.thread, "name", "?") if args.thread else "?"
            exc_type = args.exc_type.__name__ if args.exc_type else "?"
            exc_val = args.exc_value
            logger.error(
                "💥 Thread '%s' lanzó excepción no manejada: %s: %s",
                t_name, exc_type, exc_val,
            )
            if args.exc_traceback:
                traceback.print_exception(args.exc_type, args.exc_value,
                                           args.exc_traceback, file=sys.stdout)
        except Exception:
            pass  # no entrar en loop infinito

    try:
        # threading.excepthook existe desde Python 3.8
        threading.excepthook = _thread_excepthook
    except Exception as exc:
        print(f"[bootstrap] WARN: no se pudo instalar threading.excepthook: {exc}",
              flush=True)

    # 2) sys.excepthook para excepciones del thread main
    _prev_sys_hook = sys.excepthook
    def _main_excepthook(exc_type, exc_value, exc_traceback):
        try:
            logger.error(
                "💥 Main thread lanzó excepción no manejada: %s: %s",
                exc_type.__name__ if exc_type else "?", exc_value,
            )
        except Exception:
            pass
        # Encadenar al handler previo para no romper el comportamiento default
        try: _prev_sys_hook(exc_type, exc_value, exc_traceback)
        except Exception: pass
    sys.excepthook = _main_excepthook

    # 3) Adjuntar _err_handler también al logger uvicorn.error (que NO propaga
    #    al root desde uvicorn 0.30). Sin esto, errores de startup/binding o
    #    fallos del worker loop se perderían.
    for uv_logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access",
                           "fastapi", "asyncio"):
        try:
            uv_log = logging.getLogger(uv_logger_name)
            # Evitar duplicados si ya tenía el handler.
            if _err_handler not in uv_log.handlers:
                uv_log.addHandler(_err_handler)
        except Exception:
            pass

_setup_extra_capturers()


def _save_errors_to_file():
    with _ERROR_LOG_LOCK:
        data = list(_ERROR_LOG)
    try:
        tmp = ERROR_LOG_FILE.with_suffix(ERROR_LOG_FILE.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(ERROR_LOG_FILE)
    except Exception:
        pass   # no logueamos aquí para no entrar en loop infinito


def _load_errors_from_file():
    if not ERROR_LOG_FILE.exists():
        return 0
    try:
        data = json.loads(ERROR_LOG_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return 0
        with _ERROR_LOG_LOCK:
            for e in data[-ERROR_LOG_MAX:]:
                if isinstance(e, dict): _ERROR_LOG.append(e)
            return len(_ERROR_LOG)
    except Exception:
        return 0

# ─── Configuración ────────────────────────────────────────────────────────────
# Modelos potentes por DEFECTO (fallback). Editables desde el panel y persistidos.
CLAUDE_MODEL        = "claude-opus-4-7"          # web search nativo + code exec + thinking 8k
OPENAI_MODEL        = "gpt-5.5"                  # web search + reasoning configurable + RAG
# ─── Tuning de GPT-5.5 (Responses API agéntica) ──────────────────────────────
# reasoning effort: "low" (rápido, ~10s), "medium" (~30s, equilibrio), "high"
# (~60-90s, deep research suave). NO usar "xhigh" salvo que el cliente acepte
# esperas largas — supera fácilmente los 2 min. "medium" es el sweet spot.
OPENAI_REASONING_EFFORT     = "medium"
# Límite de búsquedas en el loop agéntico. El modelo decide cuándo parar, pero
# este tope evita que se quede dando vueltas en consultas ambiguas. 4 búsquedas
# suelen cubrir el peor caso de un examen oposición (1 BOE + 1 consolidado +
# 1 jurisprudencia + 1 contraste); más de 4 raras veces aporta y se acerca al
# tope de 2 min. Si quieres deep research duro, sube a 10 y effort=high.
OPENAI_MAX_TOOL_CALLS       = 4
# Lista de dominios autorizados (filters.allowed_domains, max 100). Vacío = todo
# internet. Para oposiciones lo correcto es restringir a fuentes oficiales:
#   ["boe.es", "noticias.juridicas.com", "tribunalconstitucional.es", ...]
# Si pasas lista vacía, OpenAI usa búsqueda libre (lo de hoy).
OPENAI_ALLOWED_DOMAINS: list = []
# Timeout estricto de 2 min para la llamada HTTP a /responses. Si el modelo se
# enrolla con effort=high + muchas búsquedas + cita larga, cortamos a 120s.
# Esto es ADEMÁS del _PROVIDER_TIMEOUTS["gpt"]=120 que protege el hilo entero.
OPENAI_WEB_SEARCH_HTTP_TIMEOUT_S = 120.0
GEMINI_MODEL        = "gemini-3.1-pro-preview"   # nº1 razonamiento 2026 (GPQA ~94); 2.5-pro obsoleto. Si el slug -preview falla en el panel/logs → revertir a "gemini-2.5-pro"
DEEPSEEK_MODEL      = "deepseek-v4-pro"          # Anthropic-compat, solo texto
MISTRAL_MODEL       = "mistral-large-latest"     # Mistral.ai, OpenAI-compat, analyzer texto
# Modelos OCR de VIDEO (solo se usan cuando el cliente manda video_b64)
QWEN_VIDEO_MODEL    = "qwen3-vl-plus"            # Alibaba, hasta 20 min video, second-level accuracy
GEMINI_VIDEO_MODEL  = "gemini-3.1-pro-preview"   # video understanding; el fallback de thinkingConfig lo protege. Revertir a "gemini-2.5-pro" si falla
KIMI_VIDEO_MODEL    = "kimi-k2.6"                # Moonshot, video via data URL base64, MoonViT encoder
# Claude no soporta video nativo, pero sí imagen multimodal. Usamos `req.image_b64`
# si el cliente lo manda (frame representativo), o extraemos el frame medio del
# video con cv2 si está disponible. Si nada está disponible, este OCR queda fuera.
CLAUDE_VIDEO_OCR_MODEL = "claude-opus-4-7"       # mismo que analyzer; vision para frames del video
# GPT-4o (o GPT-5.5 si se sube). Igual que Claude: no soporta video nativo, así
# que usa image_b64 o frame extraído con cv2. Reusa OPENAI_API_KEY del analyzer.
OPENAI_VIDEO_OCR_MODEL = "gpt-4o"                # rápido/barato; cámbiar a gpt-5.5 para máxima calidad
# NVIDIA Llama 3.3 Nemotron Super 49B — modelo de RAZONAMIENTO de texto puro
# servido en NVIDIA NIM Cloud (integrate.api.nvidia.com, OpenAI-compat).
# Soporta toggle de reasoning vía system prompt: "detailed thinking on/off".
# Es texto-only (no procesa imagen) — entra como analyzer de fase 2 sobre el
# OCR fusionado, igual que DeepSeek/Mistral. Antes ocupaba este slot el OCR
# Nemotron Nano 12B v2 VL, que rendía mal en hojas de examen en español.
NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
# MiniMax-M2 vía Novita (OpenAI-compat) — analyzer de RAZONAMIENTO potente y
# RÁPIDO (~10B activos, razonamiento de frontera). Endpoint Novita:
# https://api.novita.ai/openai/v1. Texto-only (recibe el OCR fusionado).
MINIMAX_MODEL = "minimax/minimax-m2"
# Mistral OCR 3 (API DEDICADA de OCR documental, NO un LLM general con vision).
# Endpoint distinto: POST /v1/ocr (no chat/completions). Procesa imagen base64
# o URL pública y devuelve `pages[i].markdown`. 99%+ accuracy en 11+ idiomas
# según Mistral; top en OmniDocBench v1.5 entre las APIs comerciales. Le
# mandamos solo el MEJOR frame extraído del video (top_k=1) — Mistral OCR es
# por documento, no acepta multi-imagen en una sola llamada.
MISTRAL_OCR_MODEL = "mistral-ocr-2512"
# DeepSeek-OCR (modelo OCR-dedicated de DeepSeek, lanzado oct 2025; v2 ene 2026).
# OJO: NO está en api.deepseek.com (que SOLO sirve texto). Se sirve en TERCEROS
# OpenAI-compatibles. Por defecto lo apuntamos a NOVITA (el usuario tiene cuenta):
# `deepseek/deepseek-ocr`. Alternativa: SiliconFlow (`deepseek-ai/DeepSeek-OCR`).
# Usa NOVITA_API_KEY (o DEEPSEEK_OCR_API_KEY) + DEEPSEEK_OCR_BASE_URL; si no hay
# key, este OCR queda fuera sin error fatal.
DEEPSEEK_OCR_MODEL = "deepseek/deepseek-ocr-2"   # id de Novita (v2: +3.7 pts OmniDocBench v1.5 ≈ 91)
# GLM-OCR (Z.AI / Zhipu) — modelo 0.9B OCR-dedicated. Top OmniDocBench v1.5
# (94.62) y +19 puntos vs PaddleOCR-VL en OCRBench (text recognition).
# Endpoint dedicado /layout_parsing — requiere Z_AI_API_KEY nueva. Si no
# está configurada, el OCR queda fuera sin error fatal.
GLM_OCR_MODEL = "glm-ocr"

# MiMo (Xiaomi) — OpenAI-compatible en https://api.xiaomimimo.com/v1 con header
# `api-key: <MIMO_API_KEY>` (no Bearer). Anthropic-compat en /anthropic/v1/messages.
# v2.5: full-modal (texto+imagen+audio+video), 1M ctx, 128K output. RPM 100, TPM 10M.
# Misma key sirve para analyzer general (MIMO_MODEL) y OCR de video (MIMO_VIDEO_MODEL):
# Xiaomi factura por tokens, no por endpoint, así que reusar el modelo es óptimo.
MIMO_MODEL          = "mimo-v2.5"                # analyzer texto/imagen
MIMO_VIDEO_MODEL    = "mimo-v2.5"                # OCR de video (mismo modelo, capability full-modal)

# ─── Tavily Search API (paso intermedio OCR → razonamiento) ────────────────────
# Docs oficiales: https://docs.tavily.com/documentation/api-reference/endpoint/search
# Endpoint: POST https://api.tavily.com/search · header Authorization: Bearer <key>.
# Por cada pregunta del examen disparamos 1 búsqueda en paralelo. Las primeras
# 2-3 fuentes (title + url + content) se inyectan al prompt del analyzer como
# bloque <internet_context> para enriquecer el razonamiento.
#
# CRÍTICA DE DISEÑO:
#   - max_results bajo (3) → prompt no explota; el razonador no se ahoga.
#   - search_depth "basic" → respuesta <2s típica; "advanced" tarda 4-6s.
#   - include_answer=False → ahorra latencia (Tavily no resume él, lo hace la IA).
#   - Deadline TOTAL absoluto (TAVILY_TOTAL_DEADLINE_S): si los hilos no terminan
#     a tiempo, el pipeline sigue SIN <internet_context> (no bloqueamos al usuario).
TAVILY_SEARCH_URL   = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS  = 3      # fuentes por pregunta — pocas, calidad > cantidad
TAVILY_SEARCH_DEPTH = "basic"  # "basic" (rápido) | "advanced" (más profundo, +2-4s)
# Timeout de UNA llamada HTTP individual a Tavily (por pregunta).
TAVILY_HTTP_TIMEOUT_S = 6.0
# Deadline TOTAL del paso de enriquecimiento (todas las preguntas en paralelo).
# Si no termina dentro de este margen, el pipeline sigue SIN internet_context.
# 8s cubre 90% de casos con max_results=3 y depth=basic; aún así dejamos margen.
TAVILY_TOTAL_DEADLINE_S = 8.0
# Cap defensivo: si una pregunta tiene MUCHÍSIMO texto, recortamos el query a
# este nº de chars antes de mandarlo (Tavily no acepta queries >400 chars bien).
TAVILY_QUERY_MAX_CHARS = 380
# Cap por fuente al inyectarla al prompt (evita inflar el contexto del analyzer).
TAVILY_SNIPPET_MAX_CHARS = 350
# Si está apagado por config, el paso se SALTA (pipeline OCR→razonamiento intacto).
TAVILY_ENABLED      = True

# ─── Prompt único OCR de video ────────────────────────────────────────────────
# Idéntico para los 3 proveedores (Qwen / Gemini / Kimi). El objetivo es que las
# 3 transcripciones tengan el MISMO formato canónico, para que la fase 2 pueda
# fusionarlas por consenso sin que ningún analyzer confunda 3 lecturas con 3 exámenes.
#
# Técnicas aplicadas (investigación 2025-2026):
#   - VLMs siguen instrucciones de formato muy bien si das estructura explícita.
#   - Un único few-shot ejemplo > varios (sobre-prompting degrada accuracy).
#   - "Inter-model agreement converges, errors diverge" (Consensus Entropy 2025):
#     si los 3 OCRs usan el MISMO formato canónico, la fusión es trivial.
#   - Marcar incertidumbre explícita es CRÍTICO para que el analyzer fusione bien.
OCR_VIDEO_PROMPT = """<task>
Transcribir el texto visible de UNA hoja de examen tipo test filmada en video.
</task>

<context>
La cámara puede temblar, cambiar de foco, o capturar distintas partes de la hoja
en frames distintos. Combina TODO lo que veas en los distintos frames para producir
UNA transcripción única, ordenada y coherente. No es necesario referir frames
concretos: la salida es UN solo bloque de texto.
</context>

<output_format>
**Marcadores de sección (CRÍTICO para la fusión multi-OCR).** Si en la hoja aparece
un encabezado o título que separa grupos de preguntas (p.ej. "BLOQUE IV",
"Preguntas de reserva", "PARTE 2", "Sección B", "Reserva"...), emite ANTES del
primer número de ese grupo una línea con este formato exacto:

## SECCION: <nombre del bloque tal cual aparece>

Si la hoja no tiene secciones explícitas, omite la línea SECCION (no la inventes).

**Preguntas.** Para cada pregunta detectada, produce EXACTAMENTE este patrón
(separador: línea en blanco):

N. <enunciado literal de la pregunta>
A) <opción A>
B) <opción B>
C) <opción C>
D) <opción D>

**N debe ser el número ORIGINAL que aparece en la hoja** (no renumerar). Si la
hoja muestra "25." entonces escribe "25.", no "1.". Si en una sección nueva
("Preguntas de reserva", "BLOQUE IV"…) la numeración se reinicia a 1, respétalo
tal cual (la línea SECCION delante ya las distingue).

Si una pregunta tiene menos opciones visibles (p.ej. solo A y B), transcribe solo
esas. Mantén el orden en que aparecen en la hoja.

**SIN LÍNEAS EN BLANCO DENTRO DE UNA PREGUNTA — REGLA CRÍTICA.**
Toda la pregunta (línea "N. …" + sus opciones A/B/C/D + cualquier fragmento
de código, HTML, XSD, XML, CSS, JSON, IPs, URLs, etc. que aparezca dentro
del enunciado o de una opción) va en UN ÚNICO BLOQUE COMPACTO. Las líneas
en blanco SÓLO separan preguntas distintas, o un "## SECCION:" del bloque
siguiente. JAMÁS metas una línea en blanco entre el enunciado y "A)", ni
entre "A)" y "B)", ni a mitad del código de una opción.

**Código/HTML/XSD/CSS en una opción → UNA SOLA LÍNEA por opción.**
Si la opción A contiene 10 líneas de XSD o HTML en el original, colapsa
TODO ese contenido a una sola línea "A) <xsd:element …><xsd:complexType>…"
separando trozos con un espacio. Lo mismo para B/C/D. Nunca repartas el
contenido de una opción en varias líneas físicas.

**Tokens numéricos jamás aislados.** Direcciones IP (10.1.0.0, 10.9.0.0),
versiones de estándares (802.11r, 802.11p), velocidades (1 Gbps, 100 Mbps),
códigos numéricos (0000-123456-ACA), prefijos CIDR (/16), patrones regex
con \\d{n}, y cualquier número o identificador que aparezca en el examen
SIEMPRE va inline dentro del enunciado o de la opción correspondiente.
Nunca los pongas en una línea aislada rodeada de blancos — eso engaña al
parser y crea preguntas fantasma.
</output_format>

<precision_rules>
  • LITERAL: respeta puntuación, acentos, símbolos (≥, ², %, ±, °, …).
  • Palabra/trozo ilegible o dudoso → escribe [?] en su lugar. NO inventes.
  • Pregunta parcialmente legible → transcribe lo legible + "[...?]" al final.
  • IGNORA marcas a mano (círculos, subrayados, palomitas, tachones, letras
    rellenadas por el usuario). Esas marcas NO entran en la transcripción.
  • Si dos partes del enunciado aparecen en frames distintos, únelas con un
    espacio si la sintaxis lo permite; usa [...?] si hay un hueco que no ves.
</precision_rules>

<forbidden>
  • Razonar o decir cuál opción es correcta.
  • Cualquier preámbulo, encabezado, conclusión, comentario o meta-explicación
    (excepto las líneas "## SECCION: …" pedidas arriba).
  • Reordenar, agrupar, fusionar o resumir preguntas.
  • Inventar opciones, palabras, números o secciones que no veas claras.
  • Renumerar las preguntas (debe verse el N original de la hoja).
  • Bloques de código, formato JSON, o cualquier salida que no sea texto plano.
</forbidden>

<example>
ENTRADA: hoja con la pregunta 25, luego una sección "Preguntas de reserva" con 2
preguntas numeradas 1-2, luego "BLOQUE IV" con 1 pregunta (numerada 1).

SALIDA CORRECTA:
25. Un licornio es una tecnología de apoyo empleada por:
A) Los usuarios ciegos.
B) Los usuarios con discapacidad motriz.
C) Los usuarios sordos.
D) No es una tecnología de apoyo.

## SECCION: Preguntas de reserva

1. Indique en cuál de los siguientes lenguajes [...] herede de varias superclases:
A) Java.
B) Python.
C) Visual Basic .NET.
D) C#.

2. ¿Cuál de las siguientes normas se relaciona con SQL?
A) IEEE 1394.
B) ISO 9100.
C) ISO/IEC 9075.
D) IEEE 754.

## SECCION: BLOQUE IV

1. En un sistema operativo UNIX, ¿qué hace [...]?
A) ...
B) ...
C) ...
D) ...
</example>

<example_codigo>
ENTRADA: una pregunta con XSD/HTML/CSS multilínea en el enunciado y en las
opciones. En la hoja, el código ocupa varias líneas físicas con saltos.

SALIDA CORRECTA (todo el código colapsado a una sola línea por opción, SIN
líneas en blanco dentro del bloque de la pregunta):
3. El identificador asociado a cada Unidad se denomina CodigoDIR3. Dada la definición XSD <xsd:simpleType name="CodigoDIR3"><xsd:restriction base="xsd:string"><xsd:pattern value="\\(\\d{4}\\)-\\(\\d{6}\\)-\\(([AC]){3}\\)" /></xsd:restriction></xsd:simpleType>, indique cuál validaría el esquema:
A) <CodigoDIR3>(0000)-(123456)-(ACA)</CodigoDIR3>
B) <CodigoDIR3>(0000)-(123456)-AAA</CodigoDIR3>
C) <CodigoDIR3>(0000)-(123456)-(ABC)</CodigoDIR3>
D) <CodigoDIR3>(0000)-(123345)-(AAC)</CodigoDIR3>

SALIDA INCORRECTA (NO HAGAS ESTO — partir el bloque con líneas en blanco
crea preguntas fantasma "10.x" / "802.x" / "0000-…" en el parser):
3. El identificador asociado…

<xsd:simpleType name="CodigoDIR3">
<xsd:restriction base="xsd:string">

<xsd:pattern value="…" />

A) <CodigoDIR3>(0000)-(123456)-(ACA)</CodigoDIR3>

B) <CodigoDIR3>(0000)-(123456)-AAA</CodigoDIR3>
</example_codigo>

<example_red>
ENTRADA: pregunta de redes con IPs, prefijos CIDR y velocidades/estándares
en las opciones (10.1.0.0, /16, 802.11r, 100 Mbps, 1 Gbps, etc.).

SALIDA CORRECTA (los tokens numéricos siempre INLINE, nunca en línea propia):
1. Con un direccionamiento privado /16, la facultad 1 tendrá la dirección 10.1.0.0 y consecutivamente hasta la facultad 9 con 10.9.0.0. ¿Cuál es el número de hosts por subred?
A) 2^16
B) (2^16) - 1
C) (2^16) - 2
D) (2^16) - 3

3. La red wifi debe soportar configuración remota de clientes. ¿Cuál norma elige?
A) 802.11r
B) 802.11p
C) 802.11j
D) 802.11v
</example_red>

Empieza tu respuesta directamente por la primera línea de contenido (puede ser
una línea "## SECCION: …" o directamente "N. …" si no hay sección). Sin saludo,
sin preámbulo, sin explicaciones. Si no detectas NINGUNA pregunta legible,
responde una única línea: "NO_LEGIBLE"."""

DYNAMIC_CONFIG = {
    # API keys (primaria + respaldo)
    "ANTHROPIC_API_KEY":         os.environ.get("ANTHROPIC_API_KEY",         ""),
    "OPENAI_API_KEY":            os.environ.get("OPENAI_API_KEY",            ""),
    "GEMINI_API_KEY":            os.environ.get("GEMINI_API_KEY",            ""),
    "DEEPSEEK_API_KEY":          os.environ.get("DEEPSEEK_API_KEY",          ""),
    # DeepSeek-OCR NO está en api.deepseek.com (solo texto): se sirve en terceros
    # (SiliconFlow por defecto / Novita). Key + base URL propios; si la key está
    # vacía, el OCR de DeepSeek queda fuera sin error fatal (como GLM sin Z_AI key).
    "DEEPSEEK_OCR_API_KEY":      os.environ.get("DEEPSEEK_OCR_API_KEY",      ""),
    "DEEPSEEK_OCR_API_KEY_BACKUP": os.environ.get("DEEPSEEK_OCR_API_KEY_BACKUP", ""),
    "DEEPSEEK_OCR_BASE_URL":     os.environ.get("DEEPSEEK_OCR_BASE_URL",     "https://api.novita.ai/openai/v1"),
    # Z.AI / Zhipu — solo desde env / panel / persistencia remota.
    "Z_AI_API_KEY":              os.environ.get("Z_AI_API_KEY",              ""),
    "Z_AI_API_KEY_BACKUP":       os.environ.get("Z_AI_API_KEY_BACKUP",       ""),
    "MISTRAL_API_KEY":            os.environ.get("MISTRAL_API_KEY",            ""),
    "KIMI_API_KEY":              os.environ.get("KIMI_API_KEY",              ""),
    "QWEN_API_KEY":              os.environ.get("QWEN_API_KEY",              ""),
    # MiMo (Xiaomi) — solo desde env / panel / persistencia remota. Sin default en código.
    "MIMO_API_KEY":              os.environ.get("MIMO_API_KEY",              ""),
    # NVIDIA NIM Cloud — solo desde env / panel / persistencia remota.
    "NVIDIA_API_KEY":            os.environ.get("NVIDIA_API_KEY",            ""),
    # Novita AI — solo desde env / panel / persistencia remota.
    "NOVITA_API_KEY":            os.environ.get("NOVITA_API_KEY",            ""),
    "NOVITA_API_KEY_BACKUP":     os.environ.get("NOVITA_API_KEY_BACKUP",     ""),
    # Tavily Search API — solo desde env / panel / persistencia remota.
    "TAVILY_API_KEY":            os.environ.get("TAVILY_API_KEY",            ""),
    "TAVILY_API_KEY_BACKUP":     os.environ.get("TAVILY_API_KEY_BACKUP",     ""),
    "ANTHROPIC_API_KEY_BACKUP":  os.environ.get("ANTHROPIC_API_KEY_BACKUP",  ""),
    "OPENAI_API_KEY_BACKUP":     os.environ.get("OPENAI_API_KEY_BACKUP",     ""),
    "GEMINI_API_KEY_BACKUP":     os.environ.get("GEMINI_API_KEY_BACKUP",     ""),
    "DEEPSEEK_API_KEY_BACKUP":   os.environ.get("DEEPSEEK_API_KEY_BACKUP",   ""),
    "MISTRAL_API_KEY_BACKUP":    os.environ.get("MISTRAL_API_KEY_BACKUP",    ""),
    "KIMI_API_KEY_BACKUP":       os.environ.get("KIMI_API_KEY_BACKUP",       ""),
    "QWEN_API_KEY_BACKUP":       os.environ.get("QWEN_API_KEY_BACKUP",       ""),
    "MIMO_API_KEY_BACKUP":       os.environ.get("MIMO_API_KEY_BACKUP",       ""),
    "NVIDIA_API_KEY_BACKUP":     os.environ.get("NVIDIA_API_KEY_BACKUP",     ""),
    # Modelos: editables desde el panel; los defaults arriba son los fallbacks
    "OPENAI_MODEL":              OPENAI_MODEL,
    "CLAUDE_MODEL":              CLAUDE_MODEL,
    "GEMINI_MODEL":              GEMINI_MODEL,
    "DEEPSEEK_MODEL":            DEEPSEEK_MODEL,
    "MISTRAL_MODEL":             MISTRAL_MODEL,
    "MIMO_MODEL":                MIMO_MODEL,
    "QWEN_VIDEO_MODEL":          QWEN_VIDEO_MODEL,
    "GEMINI_VIDEO_MODEL":        GEMINI_VIDEO_MODEL,
    "KIMI_VIDEO_MODEL":          KIMI_VIDEO_MODEL,
    "MIMO_VIDEO_MODEL":          MIMO_VIDEO_MODEL,
    "CLAUDE_VIDEO_OCR_MODEL":    CLAUDE_VIDEO_OCR_MODEL,
    "OPENAI_VIDEO_OCR_MODEL":    OPENAI_VIDEO_OCR_MODEL,
    # NVIDIA Nemotron como ANALYZER de texto (fase 2). El OCR-video Nemotron
    # Nano 12B v2 VL anterior se eliminó del pipeline por baja calidad en
    # hojas de examen en español.
    "NVIDIA_MODEL":              NVIDIA_MODEL,
    # Tuning agéntico de GPT-5.5 (Responses API · web_search GA)
    "OPENAI_REASONING_EFFORT":   OPENAI_REASONING_EFFORT,
    "OPENAI_MAX_TOOL_CALLS":     OPENAI_MAX_TOOL_CALLS,
    "OPENAI_ALLOWED_DOMAINS":    OPENAI_ALLOWED_DOMAINS,
    # Tavily tuning (per-question search step entre OCR y analyzers)
    "TAVILY_ENABLED":            TAVILY_ENABLED,
    "TAVILY_MAX_RESULTS":        TAVILY_MAX_RESULTS,
    "TAVILY_SEARCH_DEPTH":       TAVILY_SEARCH_DEPTH,
    "TAVILY_HTTP_TIMEOUT_S":     TAVILY_HTTP_TIMEOUT_S,
    "TAVILY_TOTAL_DEADLINE_S":   TAVILY_TOTAL_DEADLINE_S,
    # Pesos en la fusión por votación (un peso 2 = vale como 2 IAs básicas).
    # Usado en fusionar(): cada IA aporta su voto ponderado por estos pesos.
    # ── Pesos ANALYZERS (fase 2 — análisis del texto OCR fusionado) ──
    "ANTHROPIC_WEIGHT":          1,
    "OPENAI_WEIGHT":             1,
    "GEMINI_WEIGHT":             1,
    "DEEPSEEK_WEIGHT":           1,
    "MISTRAL_WEIGHT":            1,
    # NVIDIA Llama-3.3 Nemotron Super 49B — texto-only, reasoning-toggle ("detailed
    # thinking on"). Peso 1 igual que el resto: top-tier en reasoning benchmarks
    # 2026, similar a DeepSeek-R1 según evals de NVIDIA.
    "NVIDIA_WEIGHT":             1,
    "KIMI_WEIGHT":               1,
    # MiMo arranca con peso 0 = no participa en la fusión hasta que se conecte
    # el caller (fusionar() no la conoce todavía). Subir a 1 cuando se integre.
    "MIMO_WEIGHT":               0,
    # ── Meta-Judge (árbitro final post-fusión, opcional) ──
    # Activado por defecto. Recibe el OCR fusionado + las respuestas A/B/C/D de
    # los 5 analyzers y produce el veredicto final. Si la llamada falla por
    # cualquier motivo (network, rate limit, JSON malformado, modelo lento),
    # se hace fallback automático a la fusión local determinística.
    # Modelo recomendado: gpt-5 (o gpt-5.5) con reasoning_effort="high".
    "META_JUDGE_ENABLED":         True,
    "META_JUDGE_MODEL":           "gpt-5",
    "META_JUDGE_REASONING_EFFORT":"high",      # low | medium | high
    "META_JUDGE_TIMEOUT_S":       60.0,
    # ── Pesos OCR de VIDEO (fase 1 — transcripción del enunciado) ──
    # Defaults tier-based según evidencia 2026 de accuracy en ES.
    # Escala 0..10 (0=ignorar). Editables vía panel /panel, persistidos en
    # Supabase/Appwrite igual que el resto de pesos.
    "MISTRAL_OCR_WEIGHT":        5,   # TIER 1 — Mistral OCR 3 99%+ ES
    "GEMINI_OCR_WEIGHT":         4,   # TIER 1 — ELO 1665 OCR Arena
    "QWEN_OCR_WEIGHT":           4,   # TIER 1 — Qwen3-VL 32 idiomas
    "GLM_OCR_WEIGHT":            4,   # TIER 2 — SOTA OmniDocBench
    "ANTHROPIC_OCR_WEIGHT":      3,   # TIER 2 — Claude latin-strong
    "OPENAI_OCR_WEIGHT":         3,   # TIER 2 — GPT-4o
    "DEEPSEEK_OCR_WEIGHT":       2,   # TIER 3 — foco chino-inglés
    "KIMI_OCR_WEIGHT":           1,   # TIER 4 — foco chino
    "MIMO_OCR_WEIGHT":           1,   # TIER 4 — "modest OCR" según vendor
    # ── Topaz Labs Image API (post-stacking enhancement opcional) ──
    # Wonder 3 = all-in-one sharpen + upscale + denoise (release abril 2026).
    # Si la API key está, el pipeline VIDEO genera una imagen "casi-perfecta"
    # uniendo el stacking local (15 frames + Optical Flow + anti-ghosting +
    # CLAHE + unsharp) con un pase final de Wonder 3. Coste ~$0.05-0.10/imagen,
    # latencia 10-30 s. Si el endpoint falla (timeout/429/5xx), la imagen
    # del stacking local se usa como fallback transparente.
    "TOPAZ_API_KEY":             os.environ.get("TOPAZ_API_KEY",             ""),
    "TOPAZ_ENABLED":             True,
    "TOPAZ_MODEL":               "Wonder 3",   # fallback automático a "Wonder 2" si 400
    "TOPAZ_OUTPUT_HEIGHT":       0,             # 0 = mantener resolución del input
    "TOPAZ_TIMEOUT_S":           60.0,          # tope global async (POST + poll + GET)
}
CONFIG_LOCK = threading.Lock()

# ─── Persistencia de configuración ────────────────────────────────────────────
# La config (API keys, modelos, pesos) NO se guarda en disco — el FS de
# Render/Railway free tier es efímero (cold start lo borra). Las fuentes de
# verdad son Supabase (principal) + Appwrite (fallback). Si ambas caen al
# arrancar, se usan los defaults hardcoded en DYNAMIC_CONFIG / env vars.
_CONFIG_KEYS_PERSISTED = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY", "KIMI_API_KEY", "QWEN_API_KEY", "MIMO_API_KEY",
    "ANTHROPIC_API_KEY_BACKUP", "OPENAI_API_KEY_BACKUP", "GEMINI_API_KEY_BACKUP",
    "DEEPSEEK_API_KEY_BACKUP", "MISTRAL_API_KEY_BACKUP", "KIMI_API_KEY_BACKUP",
    "QWEN_API_KEY_BACKUP", "MIMO_API_KEY_BACKUP",
    # Modelos: también se persisten para sobrevivir reinicios
    "CLAUDE_MODEL", "OPENAI_MODEL", "GEMINI_MODEL", "DEEPSEEK_MODEL", "MISTRAL_MODEL",
    "MIMO_MODEL",
    "QWEN_VIDEO_MODEL", "GEMINI_VIDEO_MODEL", "KIMI_VIDEO_MODEL", "MIMO_VIDEO_MODEL",
    "CLAUDE_VIDEO_OCR_MODEL", "OPENAI_VIDEO_OCR_MODEL",
    "NVIDIA_API_KEY", "NVIDIA_API_KEY_BACKUP", "NVIDIA_MODEL",
    "Z_AI_API_KEY", "Z_AI_API_KEY_BACKUP",
    "OPENAI_REASONING_EFFORT", "OPENAI_MAX_TOOL_CALLS", "OPENAI_ALLOWED_DOMAINS",
    # Tavily (key + tuning del paso intermedio web search)
    "TAVILY_API_KEY", "TAVILY_API_KEY_BACKUP",
    "TAVILY_ENABLED", "TAVILY_MAX_RESULTS", "TAVILY_SEARCH_DEPTH",
    "TAVILY_HTTP_TIMEOUT_S", "TAVILY_TOTAL_DEADLINE_S",
    # Pesos por IA (votación ponderada en fusionar). Incluir TODOS los pesos
    # que el panel pueda editar — si falta uno aquí, su valor NO se sube a
    # Supabase/Appwrite y se pierde al reiniciar (el bug de MIMO_WEIGHT).
    "ANTHROPIC_WEIGHT", "OPENAI_WEIGHT", "GEMINI_WEIGHT", "DEEPSEEK_WEIGHT",
    "MISTRAL_WEIGHT", "NVIDIA_WEIGHT", "KIMI_WEIGHT", "MIMO_WEIGHT",
    # Pesos OCR de video (fase 1) — usados en _vote_text del fusionador OCR.
    "MISTRAL_OCR_WEIGHT", "GEMINI_OCR_WEIGHT", "QWEN_OCR_WEIGHT", "GLM_OCR_WEIGHT",
    "ANTHROPIC_OCR_WEIGHT", "OPENAI_OCR_WEIGHT", "DEEPSEEK_OCR_WEIGHT",
    "KIMI_OCR_WEIGHT", "MIMO_OCR_WEIGHT",
    # Meta-Judge (árbitro final post-fusión)
    "META_JUDGE_ENABLED", "META_JUDGE_MODEL",
    "META_JUDGE_REASONING_EFFORT", "META_JUDGE_TIMEOUT_S",
    # Topaz Labs (super-resolution + denoise post-stacking, opcional)
    "TOPAZ_API_KEY", "TOPAZ_ENABLED", "TOPAZ_MODEL",
    "TOPAZ_OUTPUT_HEIGHT", "TOPAZ_TIMEOUT_S",
)


def _safe_write_json(path: Path, data: dict, keep_backup: bool = True) -> bool:
    """Escritura atómica con backup: si el JSON anterior era válido, se conserva
    como .bak para recuperarse de corrupciones. tmp + fsync + verify + rename.

    Verificación post-escritura: releemos y parseamos el .tmp ANTES de promoverlo
    a final. Si el contenido grabado está corrupto (FS al límite, disco con bad
    sector), preferimos abortar y conservar el archivo principal antiguo (válido)
    a sobrescribir con basura. Sin esta verificación, una escritura corrupta
    rotaba el archivo bueno a .bak y dejaba ambos archivos rotos tras 2 fallos
    consecutivos."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        body = json.dumps(data, indent=2, ensure_ascii=False)
        # Write + fsync para garantizar que llega al disco antes del rename
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(body)
            try: os.fsync(f.fileno())
            except (OSError, AttributeError): pass
        # Verificación post-escritura: si el archivo es enorme (>5 MB),
        # saltamos el re-parse (cuesta CPU/RAM y no detecta más bugs que el
        # propio json.dumps). Para archivos normales, releemos y parseamos.
        try:
            st_size = tmp.stat().st_size if tmp.exists() else 0
        except OSError:
            st_size = 0
        if st_size <= 5 * 1024 * 1024:
            try:
                json.loads(tmp.read_text(encoding="utf-8"))
            except Exception as verify_exc:
                logger.error("escritura corrupta detectada en %s: %s — no se promueve",
                             tmp, verify_exc)
                return False
        # Backup del fichero anterior antes de sobrescribir
        if keep_backup and path.exists():
            try: path.replace(path.with_suffix(path.suffix + ".bak"))
            except Exception as e: logger.debug("backup %s: %s", path, e)
        tmp.replace(path)
        return True
    except Exception as exc:
        logger.error("No se pudo escribir %s: %s", path, exc)
        return False
    finally:
        # Garantía: si llegamos aquí con .tmp aún en disco (escritura corrupta,
        # OOM, OSError de espacio, race que sobrevivió al replace), lo barremos.
        # Mejor un cleanup ruidoso que dejar .tmp huérfanos consumiendo inodos.
        try:
            if tmp.exists(): tmp.unlink()
        except Exception:
            pass


def _safe_float(v, default: float = 0.0) -> float:
    """B4 + B7: coerce a float defensivo. Tolera None, int, float, string
    numérico, string ISO timestamp. Si nada funciona, devuelve `default` en
    vez de propagar TypeError → evita que `gc_jobs` o `_watchdog_loop` se
    cuelguen por un timestamp corrupto en una entrada vieja de Supabase."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except (TypeError, ValueError):
            try:
                from datetime import datetime
                return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
            except Exception:
                return default
    return default


def _safe_read_json(path: Path) -> Optional[dict]:
    """Lee JSON con fallback al .bak si el principal está corrupto."""
    for candidate in (path, path.with_suffix(path.suffix + ".bak")):
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if candidate != path:
                    logger.warning("Recuperando desde backup %s (fichero principal corrupto)", candidate)
                return data
        except Exception as exc:
            logger.error("JSON corrupto %s: %s", candidate, exc)
    return None


# IMPORTANTE: TODA clave que represente un peso DEBE estar aquí. Si falta:
#   - _apply_db_value() la lee como string en lugar de int → NO la aplica
#     porque el valor en DB viene como int (returna False sin escribir).
#   - _supabase_save_config / _appwrite_save_config usan default "" en lugar
#     de 1, lo que produce snapshots inconsistentes.
# El bug histórico: MIMO_WEIGHT no estaba aquí, así que al reiniciar el server
# volvía siempre a 0 aunque el usuario lo hubiera puesto a 1 desde el panel.
_WEIGHT_KEYS = (
    # Analyzers (fase 2)
    "ANTHROPIC_WEIGHT", "OPENAI_WEIGHT", "GEMINI_WEIGHT",
    "DEEPSEEK_WEIGHT", "MISTRAL_WEIGHT", "NVIDIA_WEIGHT", "KIMI_WEIGHT", "MIMO_WEIGHT",
    # OCRs de video (fase 1)
    "MISTRAL_OCR_WEIGHT", "GEMINI_OCR_WEIGHT", "QWEN_OCR_WEIGHT", "GLM_OCR_WEIGHT",
    "ANTHROPIC_OCR_WEIGHT", "OPENAI_OCR_WEIGHT", "DEEPSEEK_OCR_WEIGHT",
    "KIMI_OCR_WEIGHT", "MIMO_OCR_WEIGHT",
)

# Keys PRIMARIAS de IA: si la DB las trae vacías, NO pisamos el valor en memoria
# (que viene de la env var del hosting). Esto garantiza el orden de prioridad:
#   1º Supabase (con valor real)   →  pisa
#   2º Appwrite (con valor real)   →  pisa si Supabase no la tiene
#   3º Env var del hosting         →  última red de seguridad
# Sin esta protección, un "" en la DB borraría una key válida de las env vars.
# Las _BACKUP keys NO se incluyen aquí: queremos poder vaciarlas desde el panel
# (un backup vacío es legítimo "este proveedor no tiene plan B").
_PROTECTED_FROM_EMPTY = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY", "KIMI_API_KEY", "QWEN_API_KEY", "MIMO_API_KEY",
)


def _apply_db_value(k: str, v) -> bool:
    """Aplica un valor leído de la DB a DYNAMIC_CONFIG con las reglas de fallback.
    Devuelve True si efectivamente sobrescribió el valor."""
    # Weights: int con clamp; valores inválidos se ignoran.
    if k in _WEIGHT_KEYS:
        try:
            DYNAMIC_CONFIG[k] = max(0, int(v))
            return True
        except (TypeError, ValueError):
            return False
    # Tavily numeric/bool tuning — parsear desde string si la DB devuelve string.
    if k == "TAVILY_ENABLED":
        if isinstance(v, bool):
            DYNAMIC_CONFIG[k] = v
            return True
        if isinstance(v, str):
            DYNAMIC_CONFIG[k] = v.strip().lower() in ("1", "true", "yes", "on")
            return True
        return False
    if k in ("TAVILY_MAX_RESULTS",):
        try:
            DYNAMIC_CONFIG[k] = max(1, min(10, int(v)))
            return True
        except (TypeError, ValueError):
            return False
    if k in ("TAVILY_HTTP_TIMEOUT_S", "TAVILY_TOTAL_DEADLINE_S"):
        try:
            DYNAMIC_CONFIG[k] = max(1.0, min(60.0, float(v)))
            return True
        except (TypeError, ValueError):
            return False
    if k == "META_JUDGE_ENABLED":
        if isinstance(v, bool):
            DYNAMIC_CONFIG[k] = v
            return True
        if isinstance(v, str):
            DYNAMIC_CONFIG[k] = v.strip().lower() in ("1", "true", "yes", "on")
            return True
        return False
    if k == "META_JUDGE_TIMEOUT_S":
        try:
            DYNAMIC_CONFIG[k] = max(10.0, min(180.0, float(v)))
            return True
        except (TypeError, ValueError):
            return False
    if k == "META_JUDGE_REASONING_EFFORT":
        if isinstance(v, str) and v.strip().lower() in ("low", "medium", "high"):
            DYNAMIC_CONFIG[k] = v.strip().lower()
            return True
        return False
    # META_JUDGE_MODEL es string libre — cae al handler genérico de strings abajo.
    # Strings: las primarias se protegen contra "" (mantenemos env var).
    if isinstance(v, str):
        if k in _PROTECTED_FROM_EMPTY and not v.strip():
            return False
        DYNAMIC_CONFIG[k] = v
        return True
    return False

# ─── Supabase (config compartida entre Render y Railway) ──────────────────────
# Defaults hardcoded: ambas instancias (Render y Railway) leen/escriben en la
# misma tabla, así un cambio hecho desde el móvil se propaga a ambos relays.
# Las env vars permiten override si hace falta mover a otro proyecto.
SUPABASE_URL  = os.environ.get("SUPABASE_URL",  "").rstrip("/")
SUPABASE_ANON = os.environ.get("SUPABASE_ANON", "")
SUPABASE_HEADERS = {
    "apikey":        SUPABASE_ANON,
    "Authorization": f"Bearer {SUPABASE_ANON}",
    "Content-Type":  "application/json",
}
_SUPABASE_LAST_OK = 0.0

# Cliente HTTP dedicado para llamadas a DB (Supabase + Appwrite). Reusa
# conexiones keep-alive y evita socket churn: el daemon de sync llama cada 60s
# y crear/cerrar un httpx.Client cada vez desperdicia sockets bajo carga.
# Timeout corto: las DBs deben responder rápido o tratamos como fallo.
DB_HTTP_CLIENT = httpx.Client(
    # C3: subir pool de 2.0 a 5.0 — flusher + sync_loop + saves desde panel
    # pueden coincidir; con pool=2.0s los 4ª conn esperaba <2s y abortaba con
    # PoolTimeout. Más capacidad de conexiones también.
    timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0),
    limits=httpx.Limits(max_keepalive_connections=8, max_connections=16,
                         keepalive_expiry=60.0),
)


def _supabase_load_config() -> int:
    """Carga la fila id=1 de relay_config y sobreescribe DYNAMIC_CONFIG.
    Devuelve nº de claves aplicadas, o 0 si falló/no había datos."""
    global _SUPABASE_LAST_OK
    try:
        url = f"{SUPABASE_URL}/rest/v1/relay_config?id=eq.1&select=data"
        r = DB_HTTP_CLIENT.get(url, headers=SUPABASE_HEADERS)
        if r.status_code != 200:
            logger.error("Supabase load HTTP %s: %s", r.status_code, r.text[:200])
            return 0
        rows = r.json()
        if not rows: return 0
        data = rows[0].get("data") or {}
        count = 0
        with CONFIG_LOCK:
            for k in _CONFIG_KEYS_PERSISTED:
                if k not in data: continue
                if _apply_db_value(k, data[k]):
                    count += 1
        _SUPABASE_LAST_OK = time.time()
        return count
    except Exception as exc:
        logger.error("Supabase load excepción: %s", exc)
        return 0


def _supabase_save_config() -> bool:
    """Sube DYNAMIC_CONFIG (campos persistidos) a la fila id=1 de relay_config."""
    global _SUPABASE_LAST_OK
    try:
        snapshot = {}
        with CONFIG_LOCK:
            for k in _CONFIG_KEYS_PERSISTED:
                default = 1 if k in _WEIGHT_KEYS else ""
                snapshot[k] = DYNAMIC_CONFIG.get(k, default)
        url = f"{SUPABASE_URL}/rest/v1/relay_config?id=eq.1"
        payload = {"data": snapshot}
        headers = {**SUPABASE_HEADERS, "Prefer": "return=minimal"}
        r = DB_HTTP_CLIENT.patch(url, headers=headers, json=payload)
        if r.status_code not in (200, 204):
            logger.error("Supabase save HTTP %s: %s", r.status_code, r.text[:200])
            return False
        _SUPABASE_LAST_OK = time.time()
        return True
    except Exception as exc:
        logger.error("Supabase save excepción: %s", exc)
        return False


# ─── Appwrite (fallback de Supabase) ──────────────────────────────────────────
# Si Supabase cae (rate limit, downtime, suspensión free tier), Appwrite mantiene
# la config sincronizada entre Render y Railway. Patrón idéntico: un único
# documento "relay_config" en la colección "config" contiene la config completa
# serializada como JSON string en el atributo "data".
#
# Credenciales solo vía env / secrets.properties (nunca hardcodear en el repo).
APPWRITE_ENDPOINT      = os.environ.get("APPWRITE_ENDPOINT",      "https://fra.cloud.appwrite.io/v1").rstrip("/")
APPWRITE_PROJECT_ID    = os.environ.get("APPWRITE_PROJECT_ID",    "")
APPWRITE_API_KEY       = os.environ.get("APPWRITE_API_KEY",       "")
APPWRITE_DATABASE_ID   = os.environ.get("APPWRITE_DATABASE_ID",   "")
APPWRITE_COLLECTION_ID = os.environ.get("APPWRITE_COLLECTION_ID", "config")
APPWRITE_DOCUMENT_ID   = os.environ.get("APPWRITE_DOCUMENT_ID",   "relay_config")

_APPWRITE_LAST_OK = 0.0


def _appwrite_enabled() -> bool:
    """True si hay credenciales suficientes para hablar con Appwrite."""
    return bool(APPWRITE_ENDPOINT and APPWRITE_PROJECT_ID and APPWRITE_API_KEY
                and APPWRITE_DATABASE_ID and APPWRITE_COLLECTION_ID)


def _appwrite_headers() -> dict:
    return {
        "X-Appwrite-Project":         APPWRITE_PROJECT_ID,
        "X-Appwrite-Key":             APPWRITE_API_KEY,
        "X-Appwrite-Response-Format": "1.9.4",
        "Content-Type":               "application/json",
    }


def _appwrite_doc_url() -> str:
    return (f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}"
            f"/collections/{APPWRITE_COLLECTION_ID}/documents/{APPWRITE_DOCUMENT_ID}")


def _appwrite_load_config() -> int:
    """Carga el documento de Appwrite y sobreescribe DYNAMIC_CONFIG.
    Devuelve nº de claves aplicadas, o 0 si falló/no había datos/no está configurado."""
    global _APPWRITE_LAST_OK
    if not _appwrite_enabled():
        return 0
    try:
        r = DB_HTTP_CLIENT.get(_appwrite_doc_url(), headers=_appwrite_headers())
        if r.status_code != 200:
            logger.error("Appwrite load HTTP %s: %s", r.status_code, r.text[:200])
            return 0
        doc = r.json()
        raw = doc.get("data")
        if not raw: return 0
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict): return 0
        count = 0
        with CONFIG_LOCK:
            for k in _CONFIG_KEYS_PERSISTED:
                if k not in data: continue
                if _apply_db_value(k, data[k]):
                    count += 1
        _APPWRITE_LAST_OK = time.time()
        return count
    except Exception as exc:
        logger.error("Appwrite load excepción: %s", exc)
        return 0


def _appwrite_save_config() -> bool:
    """Sube DYNAMIC_CONFIG al documento de Appwrite (serializado como JSON string)."""
    global _APPWRITE_LAST_OK
    if not _appwrite_enabled():
        return False
    try:
        snapshot = {}
        with CONFIG_LOCK:
            for k in _CONFIG_KEYS_PERSISTED:
                default = 1 if k in _WEIGHT_KEYS else ""
                snapshot[k] = DYNAMIC_CONFIG.get(k, default)
        payload = {"data": {"data": json.dumps(snapshot, ensure_ascii=False)}}
        r = DB_HTTP_CLIENT.patch(_appwrite_doc_url(), headers=_appwrite_headers(), json=payload)
        if r.status_code not in (200, 204):
            logger.error("Appwrite save HTTP %s: %s", r.status_code, r.text[:200])
            return False
        _APPWRITE_LAST_OK = time.time()
        return True
    except Exception as exc:
        logger.error("Appwrite save excepción: %s", exc)
        return False


# Flag: cuando Supabase escribe FALLÓ pero Appwrite OK, la fila de Supabase
# queda con datos antiguos. El sync_loop debe ignorarla hasta que un save
# posterior la actualice — si no, sobreescribe los cambios nuevos al refrescar.
_SUPABASE_STALE = False


def _save_config_to_file() -> bool:
    """Persiste DYNAMIC_CONFIG en Supabase (principal) + Appwrite (fallback).
    Devuelve True si AL MENOS una DB aceptó el cambio.

    NO se guarda en disco local: el FS de Render/Railway free tier es efímero
    (cold start lo borra) y crearía la falsa sensación de persistencia. Las
    fuentes de verdad son las 2 DBs cloud; si ambas caen, al reinicio se
    cargarán los defaults hardcoded en DYNAMIC_CONFIG."""
    global _SUPABASE_STALE
    supa_ok = _supabase_save_config()
    # Write-through también a Appwrite si está configurado: así un fallo
    # posterior de Supabase no pierde los datos recientes.
    appw_ok = _appwrite_save_config() if _appwrite_enabled() else False
    # Si Supabase falló pero Appwrite tiene los datos nuevos, marcamos Supabase
    # como "stale" para que el sync loop no nos sobreescriba con la versión vieja.
    if not supa_ok and appw_ok:
        _SUPABASE_STALE = True
        logger.error("⚠️ Supabase save failed; relying on Appwrite — marking Supabase as stale")
    elif not supa_ok and not appw_ok:
        # Ambas DBs cayeron al guardar: el cambio que el usuario acaba de hacer
        # se PERDERÁ al reiniciar el server. Esto es crítico — debe verse.
        logger.error("💥 CRÍTICO: save fail en Supabase Y Appwrite — el cambio se perderá al reiniciar")
    elif supa_ok:
        # Supabase aceptó: ya está sincronizado, podemos volver a leer de ahí.
        _SUPABASE_STALE = False
    # B20: si AL MENOS UNA DB aceptó, marcamos el write para que el sync_loop
    # no nos sobreescriba con datos viejos de la otra instancia durante los
    # próximos N segundos. La función `_mark_local_config_write` no existe
    # todavía cuando se carga este módulo (declarada más abajo), así que
    # comprobamos su existencia para evitar NameError en arranque.
    if supa_ok or appw_ok:
        try:
            _mark_local_config_write()
        except NameError:
            pass  # caso de arranque temprano antes de definir la función
    return supa_ok or appw_ok


# Cargar config AL ARRANCAR: 1º Supabase (fuente compartida Render+Railway),
# 2º Appwrite (fallback si Supabase cae), 3º defaults hardcoded / env vars.
_loaded_supa = _supabase_load_config()
if _loaded_supa > 0:
    logger.info("☁️ Config cargada desde Supabase (%d claves)", _loaded_supa)
else:
    _loaded_appw = _appwrite_load_config() if _appwrite_enabled() else 0
    if _loaded_appw > 0:
        logger.info("🔷 Config cargada desde Appwrite (%d claves) — Supabase no disponible", _loaded_appw)
    else:
        # CRÍTICO en arranque: sin DBs el server corre solo con env vars/hardcoded.
        # Los cambios desde panel SÍ se aceptarán pero no podrán persistirse hasta
        # que vuelva alguna DB. Visible en /api/errors para que el operador actúe.
        logger.error("💥 CRÍTICO: ambas DBs caídas al arrancar — corriendo con defaults hardcoded / env vars")


# ─── Persistencia de JOBS ─────────────────────────────────────────────────────
# Los jobs sobreviven a un cold-start/redeploy → la historia del panel no se vacía.
# /reset (que envía el móvil al iniciar sistema) borra todo y deja el relay vacío.
# Las imágenes base64 NO se persisten (peso × 50 jobs ≈ MB innecesarios).
#
# IMPORTANTE: JOBS y LOCK se declaran AQUÍ (no más abajo) porque
# _load_jobs_from_file() los usa al arrancar el módulo. Si están definidos
# después de la llamada al loader, un arranque con relay_jobs.json existente
# crashea con NameError. En Render no se ve (FS efímero) pero en local sí.
JOBS: dict = {}
LOCK = threading.Lock()
JOBS_FILE = Path(os.environ.get("JOBS_FILE", "relay_jobs.json"))
_JOBS_DIRTY = threading.Event()

# ─── Contexto de imagen STICKY (server-side) ──────────────────────────────────
# El relay RECUERDA la última imagen-contexto (diagrama del caso práctico) marcada
# con la casilla 🖼️ del panel —o la última que mandó el móvil— y la adjunta él
# mismo a cada /ask posterior que NO traiga la suya. Antes el flujo dependía de
# que el MÓVIL guardara la imagen (ctx_diagram.jpg) y la reenviara en cada /ask
# (round-trip relay→móvil→relay); ahora el servidor es autónomo: no depende del
# móvil y ahorra mandar la imagen abajo+arriba por radio.
#
# Vive SOLO en RAM (un redeploy la pierde — el panel muestra el estado para
# re-marcarla; persistir un blob de ~150-500 KB inflaría el cap de Supabase).
# Expira por TTL y se limpia con /reset o con el botón del panel.
STICKY_CONTEXT_TTL_S = int(os.environ.get("STICKY_CONTEXT_TTL_S", "10800"))  # 3h
_STICKY_CONTEXT_LOCK = threading.Lock()
_STICKY_CONTEXT: dict = {}   # {"b64","mime","set_at","source_job"}


def _set_sticky_context(b64: Optional[str], mime: Optional[str], source_job: str = "") -> None:
    """Fija/reemplaza la imagen-contexto sticky. No-op si b64 vacío o demasiado
    grande. NUNCA lanza.

    Cap de 12 MB (= mismo que el guard de edit_result): un sticky enorme se
    re-inyectaría en CADA /ask posterior y se mandaría a todas las IAs de visión
    → saturaría tokens/ancho de banda en cada examen. Mejor no hacerlo sticky."""
    if not b64:
        return
    if len(b64) > 12 * 1024 * 1024:
        logger.warning("🖼️ contexto sticky ignorado: imagen demasiado grande (%d KB)",
                       len(b64) * 3 // 4 // 1024)
        return
    try:
        with _STICKY_CONTEXT_LOCK:
            _STICKY_CONTEXT.clear()
            _STICKY_CONTEXT.update({
                "b64":        b64,
                "mime":       mime or "image/jpeg",
                "set_at":     time.time(),
                "source_job": source_job or "",
            })
    except Exception:
        pass


def _get_sticky_context() -> Optional[dict]:
    """Devuelve el contexto sticky vigente (copia) o None si no hay / expiró por
    TTL. Auto-purga al expirar. NUNCA lanza."""
    try:
        with _STICKY_CONTEXT_LOCK:
            if not _STICKY_CONTEXT.get("b64"):
                return None
            if (STICKY_CONTEXT_TTL_S > 0
                    and (time.time() - _STICKY_CONTEXT.get("set_at", 0)) > STICKY_CONTEXT_TTL_S):
                _STICKY_CONTEXT.clear()
                return None
            return dict(_STICKY_CONTEXT)
    except Exception:
        return None


def _clear_sticky_context() -> bool:
    """Limpia el contexto sticky. Devuelve True si había algo. NUNCA lanza."""
    try:
        with _STICKY_CONTEXT_LOCK:
            had = bool(_STICKY_CONTEXT.get("b64"))
            _STICKY_CONTEXT.clear()
        return had
    except Exception:
        return False


def _mark_jobs_dirty():
    """Marca que JOBS ha cambiado para que el flusher escriba en background."""
    _JOBS_DIRTY.set()


# Campos con prefijo "_" que SÍ deben persistir entre restarts. El resto de
# campos "_*" (como _timer_started, _touched, _cancelled) son estado transitorio
# de runtime y se recrean en cada job nuevo.
#   _providers: la lista de analyzers que corren en este job. Sin ella, el panel
#               cae al fallback Object.keys(responses) que da resultados raros
#               cuando faltan IAs (puede pintar IAs que nunca corrieron).
_PERSIST_UNDERSCORE_KEYS = ("_providers",)

# Cap "blando" para responses[*].raw en disco. El archivo en disco aguanta más
# que las DBs (no hay límite de 1 MB por fila), pero raws de 50 KB × 5 IAs ×
# 150 jobs = 37 MB que `_safe_write_json` tendría que dumps+read+parse cada
# flush bajo el LOCK del flusher. Cap a 8000 chars conserva ~lo justo para
# debug (preview + final answer) sin inflar el archivo.
_DISK_MAX_RAW_CHARS = 8000
# Telemetría del flusher: contador para loguear el tamaño cada N flushes.
_flusher_metrics = {"n": 0, "last_size": 0, "last_ms": 0}


def _snapshot_jobs_for_disk() -> dict:
    """Construye un snapshot SERIALIZABLE e INDEPENDIENTE de JOBS para escribir
    a disco. Diseño:
      1. Tomar LOCK y hacer copy() shallow de JOBS.items() — barato.
      2. Para cada job, descender un nivel y deepcopy() los sub-dicts que
         puedan mutarse (responses, ocr_results, phase_history, intentos).
         json.dumps() ya no verá referencias compartidas con el dict vivo,
         así que ningún hilo concurrente puede romper la serialización.
      3. Aplicar el mismo stripping que `_strip_job_for_supabase` pero con
         cap MAYOR (8000 chars de raw) — disco aguanta más que JSONB de 1MB.

    Sin esto, `json.dumps` corre fuera del LOCK con dicts vivos: un `responses[X] = entry`
    desde otro hilo en mitad de la serialización lanza
    'RuntimeError: dictionary changed size during iteration' (A1)."""
    import copy
    with LOCK:
        shallow = list(JOBS.items())
    out: dict = {}
    for jid, j in shallow:
        try:
            clean: dict = {}
            for k, v in j.items():
                # Blobs base64 pesados: NUNCA se persisten (ni a disco ni a DB).
                # Son efímeros (viven en RAM solo para los últimos N jobs vía la
                # retención por recencia de gc_jobs). Persistirlos bloateaba el
                # JSON de Supabase/Appwrite (>1MB → shrink que tiraba jobs) y, peor,
                # al reiniciar recargaba TODOS esos blobs a RAM. Conservamos los
                # flags has_*/source/count (metadatos ligeros) para el histórico.
                if k in ("img", "video", "extracted_frames",
                         "fused_image_b64", "fused_image_local_b64",
                         "context_image", "context_image_provided_b64"):
                    continue
                if k.startswith("_") and k not in _PERSIST_UNDERSCORE_KEYS:
                    continue
                # Deepcopy de las estructuras anidadas mutables. Los strings/
                # nums/bools son inmutables, no hace falta copiarlos.
                if k == "responses" and isinstance(v, dict):
                    slim = {}
                    for prov, r in v.items():
                        if not isinstance(r, dict):
                            slim[prov] = r
                            continue
                        rr = copy.deepcopy(r)
                        raw = rr.get("raw")
                        if isinstance(raw, str) and len(raw) > _DISK_MAX_RAW_CHARS:
                            rr["raw"] = raw[:_DISK_MAX_RAW_CHARS]
                            rr["raw_truncated_from"] = len(raw)
                        slim[prov] = rr
                    clean[k] = slim
                elif k == "ocr_results" and isinstance(v, dict):
                    slim = {}
                    for prov, r in v.items():
                        if not isinstance(r, dict):
                            slim[prov] = r
                            continue
                        rr = copy.deepcopy(r)
                        txt = rr.get("text")
                        if isinstance(txt, str) and len(txt) > _DISK_MAX_RAW_CHARS:
                            rr["text"] = txt[:_DISK_MAX_RAW_CHARS]
                            rr["text_truncated_from"] = len(txt)
                        slim[prov] = rr
                    clean[k] = slim
                elif k == "phase_history" and isinstance(v, list):
                    # Cap defensivo: si por algún bug crece sin freno (A6),
                    # truncamos al persistir. El history retiene los últimos 100
                    # eventos — suficiente para depurar y previene archivos enormes.
                    clean[k] = copy.deepcopy(v[-100:]) if len(v) > 100 else copy.deepcopy(v)
                elif isinstance(v, (dict, list)):
                    # Cualquier otro contenedor: deepcopy para aislar.
                    clean[k] = copy.deepcopy(v)
                else:
                    clean[k] = v
            out[jid] = clean
        except Exception as exc:
            # Si UN job falla al serializar, NO bloquea el snapshot entero —
            # lo dejamos fuera con un marker para diagnóstico.
            logger.error("snapshot_jobs[%s] falló: %s — omitido del flush", jid[:8], exc)
    return out


def _save_jobs_to_file():
    """Snapshot atómico de JOBS sin las imágenes base64. Con backup .bak.
    Filtra campos "_*" salvo los marcados como persistibles en _PERSIST_UNDERSCORE_KEYS.
    El snapshot es DEEPCOPY (vía _snapshot_jobs_for_disk) — json.dumps no toca
    refs vivas, así que es resistente a mutaciones concurrentes (A1)."""
    t0 = time.monotonic()
    snapshot = _snapshot_jobs_for_disk()
    _safe_write_json(JOBS_FILE, snapshot)
    # Telemetría cada 10 flushes (sin spammar el log)
    _flusher_metrics["n"] += 1
    _flusher_metrics["last_ms"] = int((time.monotonic() - t0) * 1000)
    try:
        _flusher_metrics["last_size"] = JOBS_FILE.stat().st_size if JOBS_FILE.exists() else 0
    except OSError:
        _flusher_metrics["last_size"] = 0
    if _flusher_metrics["n"] % 10 == 0:
        logger.info("📁 flusher: %d jobs · %d KB · %d ms",
                    len(snapshot), _flusher_metrics["last_size"] // 1024,
                    _flusher_metrics["last_ms"])


def _load_jobs_from_file() -> int:
    """Carga JOBS desde disco (con fallback a .bak). Jobs in-progress se marcan
    como error porque sus hilos murieron en el restart."""
    data = _safe_read_json(JOBS_FILE)
    if data is None:
        return 0
    # C4: si el contenido del JSON no es dict, loguear y retornar 0 (en vez de
    # silencio). Antes podía pasar una lista corrupta y el caller asumía vacío.
    if not isinstance(data, dict):
        logger.error("📁 %s tiene tipo inesperado %s — ignorado",
                     JOBS_FILE, type(data).__name__)
        return 0
    rescued, terminated, skipped = 0, 0, 0
    terminated_ids: list = []
    with LOCK:
        for jid, j in data.items():
            if not isinstance(j, dict):
                skipped += 1
                continue
            # B4: coerce timestamps a float — si Supabase/Appwrite lo devuelven
            # como string ISO o int, `now - j["created"]` lanzaría TypeError en
            # gc_jobs y el daemon entraría en backoff exponencial sin que JOBS
            # se limpie nunca.
            _coerce_job_floats(j)
            if j.get("status") in ("pending", "awaiting_review"):
                j["status"]  = "error"
                j["error"]   = "Relay reiniciado durante el procesamiento"
                j["finished"] = time.time()
                terminated += 1
                terminated_ids.append(jid)
            j["img"]   = None  # perdida en el reinicio (no se persiste)
            j["video"] = None  # idem para video MP4 base64
            JOBS[jid] = j
            rescued += 1
    if terminated:
        # logger.error (no info) → entra al deque _ERROR_LOG visible en el panel.
        # Antes era info y el operador veía el job como ERROR sin saber por qué.
        logger.error(
            "📁 %d job(s) en curso marcados como error tras restart (job_ids: %s)",
            terminated, ", ".join(jid[:8] for jid in terminated_ids[:10]),
        )
    if skipped:
        # ERROR: el snapshot persistido tiene entradas corruptas — el operador
        # necesita verlo en VER ERRORES para detectar bugs en serialización
        # o problemas de almacenamiento (disco corrupto, FS efímero borrando).
        logger.error("📁 %d entrada(s) ignorada(s) por no ser dict (snapshot corrupto?)", skipped)
    return rescued


def _coerce_job_floats(j: dict) -> None:
    """B4: convierte timestamps a float (in-place). Tolera string ISO, int,
    float, o None. Si no se puede coerce, deja el campo como está y deja que
    el caller lo trate (gc_jobs salta jobs cuyo `created` no sea numérico)."""
    for k in ("created", "finished", "review_deadline", "reviewed_at"):
        v = j.get(k)
        if v is None or isinstance(v, (int, float)):
            continue
        try:
            j[k] = float(v)
        except (TypeError, ValueError):
            # Si es un string ISO (poco probable pero defensive), intentar parsearlo
            if isinstance(v, str):
                try:
                    from datetime import datetime
                    j[k] = datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
                except Exception:
                    j[k] = 0.0
            else:
                j[k] = 0.0


# ─── JOBS en Supabase (sobrevive re-deploys del FS efímero de Render) ─────────
# El FS de Render free tier se borra en cada deploy → JOBS_FILE en disco no es
# fiable. Replicamos JOBS a la tabla relay_jobs (single-row, id=1, data jsonb).
# Stripping agresivo para no superar el límite práctico de ~1MB por fila:
#   - img/video (ya no se persisten ni a disco)
#   - responses[*].raw → truncado a 2000 chars (preview suficiente)
#   - ocr_results[*].text → solo metadatos (chars/ok/ms/error)
#   - tavily_block → eliminado (tavily_stats se mantiene, que es lo que el panel
#     necesita para el desplegable 🔎).
# Best-effort: si falla la DB el job sigue funcionando con persistencia local.
SUPABASE_JOBS_URL = f"{SUPABASE_URL}/rest/v1/relay_jobs?id=eq.1"
_SUPABASE_JOBS_MAX_RAW = 2000     # chars de responses[*].raw a conservar
_SUPABASE_JOBS_MAX_BYTES = 900_000  # umbral suave (~900KB) antes de avisar


def _strip_job_for_supabase(j: dict) -> dict:
    """Snapshot ligero pero útil para DBs cloud (Supabase + Appwrite).
    Conserva metadatos + responses con raw truncado. Drop de
    img/video/ocr_text/tavily_block. El panel puede redibujar el historial sin
    estos campos; el expediente queda parcial pero suficiente para diagnóstico.

    CRÍTICO: hace deepcopy() de los contenedores anidados (responses[*],
    ocr_results[*], phase_history). Sin deepcopy, después de soltar el LOCK
    el caller puede ver mutaciones concurrentes al serializar con json.dumps,
    causando 'RuntimeError: dictionary changed size during iteration' (A1)."""
    import copy
    out: dict = {}
    for k, v in j.items():
        if k in ("img", "video"):
            continue
        if k.startswith("_") and k not in _PERSIST_UNDERSCORE_KEYS:
            continue
        if k == "tavily_block":
            continue
        if k == "responses" and isinstance(v, dict):
            slim = {}
            for prov, r in v.items():
                if not isinstance(r, dict):
                    slim[prov] = r
                    continue
                rr = copy.deepcopy(r)
                raw = rr.get("raw")
                if isinstance(raw, str) and len(raw) > _SUPABASE_JOBS_MAX_RAW:
                    rr["raw"] = raw[:_SUPABASE_JOBS_MAX_RAW]
                    rr["raw_truncated_from"] = len(raw)
                slim[prov] = rr
            out[k] = slim
            continue
        if k == "ocr_results" and isinstance(v, dict):
            slim = {}
            for prov, r in v.items():
                if not isinstance(r, dict):
                    slim[prov] = r
                    continue
                rr = {kk: copy.deepcopy(vv) for kk, vv in r.items() if kk != "text"}
                txt = r.get("text")
                if isinstance(txt, str):
                    rr["chars"] = len(txt)
                slim[prov] = rr
            out[k] = slim
            continue
        if k == "phase_history" and isinstance(v, list):
            out[k] = copy.deepcopy(v[-100:]) if len(v) > 100 else copy.deepcopy(v)
            continue
        if isinstance(v, (dict, list)):
            out[k] = copy.deepcopy(v)
            continue
        out[k] = v
    return out


def _shrink_snapshot_to_fit(snapshot: dict, max_bytes: int) -> tuple:
    """B1: si el snapshot serializado supera `max_bytes`, dropea los jobs más
    antiguos (por `created` ascendente, los más viejos primero) hasta caber.
    Retorna (snapshot_recortado, n_dropeados).

    Razón: el PATCH a Supabase con JSONB >1MB devuelve 400/413 y el flush
    "ha fallado" en silencio. Mejor un snapshot parcial actualizado que un
    rechazo total que deje la fila stale."""
    try:
        body = json.dumps(snapshot, ensure_ascii=False)
    except Exception:
        return snapshot, 0
    if len(body) <= max_bytes:
        return snapshot, 0
    # Ordenar por created ascendente (más antiguos primero); dropear hasta caber.
    ordered = sorted(snapshot.items(),
                     key=lambda kv: kv[1].get("created", 0) if isinstance(kv[1], dict) else 0)
    dropped = 0
    while len(body) > max_bytes and ordered:
        ordered.pop(0)  # quita el más antiguo
        dropped += 1
        try:
            body = json.dumps(dict(ordered), ensure_ascii=False)
        except Exception:
            break
    return dict(ordered), dropped


def _supabase_save_jobs() -> bool:
    """Vuelca JOBS (versión slim) a la fila id=1 de relay_jobs. Best-effort.

    B1: si supera el límite práctico de ~900KB, dropea los jobs más antiguos
    hasta caber (no abandona — mantiene los recientes en la DB).
    B2: si la fila id=1 no existe (PATCH devuelve 200 vacío), hace POST con
    upsert para crearla."""
    global _SUPABASE_LAST_OK
    try:
        with LOCK:
            snapshot = {jid: _strip_job_for_supabase(j) for jid, j in JOBS.items()}
        snapshot, n_dropped = _shrink_snapshot_to_fit(snapshot, _SUPABASE_JOBS_MAX_BYTES)
        if n_dropped:
            logger.warning("☁ Supabase save_jobs: dropados %d jobs antiguos para caber en %dB",
                           n_dropped, _SUPABASE_JOBS_MAX_BYTES)
        # ISO-8601 con sufijo "Z" para que Postgres lo parsee como UTC. Si
        # mandáramos "now()" como string, PostgREST lo trataría literalmente
        # ("now()" no es una llamada a función desde JSON).
        from datetime import datetime, timezone
        payload = {"data": snapshot,
                    "updated_at": datetime.now(timezone.utc).isoformat()}
        # B2: pedir representation para detectar PATCH a fila inexistente.
        # PostgREST con Prefer:return=minimal devuelve 204 incluso si NO hay
        # match — el upsert nunca dispararía. Usamos return=representation y
        # detectamos `[]` vacío (filtro no matcheó).
        headers = {**SUPABASE_HEADERS, "Prefer": "return=representation"}
        r = DB_HTTP_CLIENT.patch(SUPABASE_JOBS_URL, headers=headers, json=payload)
        if r.status_code == 200:
            # Si rows vacío: la fila id=1 no existe → upsert con POST.
            try:
                rows_after = r.json() if r.text else []
            except Exception:
                rows_after = None
            if rows_after == []:
                logger.warning("☁ relay_jobs id=1 no existe — creándola con POST upsert")
                create_url = f"{SUPABASE_URL}/rest/v1/relay_jobs"
                create_headers = {**SUPABASE_HEADERS,
                                  "Prefer": "resolution=merge-duplicates,return=minimal"}
                create_payload = {"id": 1, **payload}
                r2 = DB_HTTP_CLIENT.post(create_url, headers=create_headers,
                                          json=create_payload)
                if r2.status_code not in (200, 201, 204):
                    logger.error("Supabase save_jobs upsert HTTP %s: %s",
                                 r2.status_code, r2.text[:200])
                    return False
        elif r.status_code not in (200, 204):
            logger.error("Supabase save_jobs HTTP %s: %s", r.status_code, r.text[:200])
            return False
        _SUPABASE_LAST_OK = time.time()
        return True
    except Exception as exc:
        logger.error("Supabase save_jobs excepción: %s", exc)
        return False


def _supabase_load_jobs() -> int:
    """Carga JOBS desde relay_jobs (fila id=1). Mismo postproceso que el load
    desde disco: jobs en pending/awaiting_review se marcan como error
    porque sus hilos murieron en el restart."""
    global _SUPABASE_LAST_OK
    try:
        url = f"{SUPABASE_URL}/rest/v1/relay_jobs?id=eq.1&select=data"
        r = DB_HTTP_CLIENT.get(url, headers=SUPABASE_HEADERS)
        if r.status_code != 200:
            logger.error("Supabase load_jobs HTTP %s: %s", r.status_code, r.text[:200])
            return 0
        rows = r.json()
        if not rows: return 0
        data = rows[0].get("data") or {}
        if not isinstance(data, dict) or not data:
            return 0
        rescued, terminated, skipped = 0, 0, 0
        terminated_ids: list = []
        with LOCK:
            for jid, j in data.items():
                if not isinstance(j, dict):
                    skipped += 1
                    continue
                _coerce_job_floats(j)  # B4
                if j.get("status") in ("pending", "awaiting_review"):
                    j["status"]  = "error"
                    j["error"]   = "Relay reiniciado durante el procesamiento"
                    j["finished"] = time.time()
                    terminated += 1
                    terminated_ids.append(jid)
                j["img"]   = None
                j["video"] = None
                JOBS[jid] = j
                rescued += 1
        if terminated:
            logger.error(
                "☁ %d job(s) en curso marcados como error tras restart desde Supabase (ids: %s)",
                terminated, ", ".join(jid[:8] for jid in terminated_ids[:10]),
            )
        _SUPABASE_LAST_OK = time.time()
        return rescued
    except Exception as exc:
        logger.error("Supabase load_jobs excepción: %s", exc)
        return 0


# ─── JOBS en Appwrite (fallback secundario si Supabase cae) ──────────────────
# Mismo patrón que el config: colección "jobs", documento "relay_jobs", atributo
# string "data" con el snapshot serializado en JSON. La columna admite hasta
# 1 MB — el mismo umbral que el SUPABASE_JOBS_MAX_BYTES, así que si la versión
# slim entra en Supabase también entra aquí.
APPWRITE_JOBS_COLLECTION_ID = os.environ.get("APPWRITE_JOBS_COLLECTION_ID", "jobs")
APPWRITE_JOBS_DOCUMENT_ID   = os.environ.get("APPWRITE_JOBS_DOCUMENT_ID",   "relay_jobs")


def _appwrite_jobs_doc_url() -> str:
    return (f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}"
            f"/collections/{APPWRITE_JOBS_COLLECTION_ID}/documents/{APPWRITE_JOBS_DOCUMENT_ID}")


# Cap de la cadena `data` en Appwrite. La columna admite 1 MB; si supera ese
# tamaño el PATCH falla con 400. Aplicamos shrink antes de mandar.
_APPWRITE_JOBS_MAX_BYTES = 950_000


def _appwrite_save_jobs() -> bool:
    """Vuelca JOBS (versión slim) al documento relay_jobs en Appwrite. Best-effort.
    El snapshot se serializa como string (Appwrite no tiene JSONB nativo, su
    columna 'data' es un string).

    B1: shrink si supera _APPWRITE_JOBS_MAX_BYTES (column size = 1MB).
    B3: si el documento no existe (404), lo crea con POST en lugar de fallar
    silenciosamente cada flush."""
    global _APPWRITE_LAST_OK
    if not _appwrite_enabled():
        return False
    try:
        with LOCK:
            snapshot = {jid: _strip_job_for_supabase(j) for jid, j in JOBS.items()}
        snapshot, n_dropped = _shrink_snapshot_to_fit(snapshot, _APPWRITE_JOBS_MAX_BYTES)
        if n_dropped:
            logger.warning("🔷 Appwrite save_jobs: dropados %d jobs antiguos para caber en %dB",
                           n_dropped, _APPWRITE_JOBS_MAX_BYTES)
        payload = {"data": {"data": json.dumps(snapshot, ensure_ascii=False)}}
        r = DB_HTTP_CLIENT.patch(_appwrite_jobs_doc_url(),
                                  headers=_appwrite_headers(), json=payload)
        if r.status_code == 404:
            # B3: documento no existe → crearlo con POST.
            logger.warning("🔷 Appwrite doc %s no existe — creándolo",
                           APPWRITE_JOBS_DOCUMENT_ID)
            create_url = (f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}"
                          f"/collections/{APPWRITE_JOBS_COLLECTION_ID}/documents")
            create_payload = {
                "documentId": APPWRITE_JOBS_DOCUMENT_ID,
                "data": {"data": json.dumps(snapshot, ensure_ascii=False)},
            }
            r2 = DB_HTTP_CLIENT.post(create_url, headers=_appwrite_headers(),
                                      json=create_payload)
            if r2.status_code not in (200, 201):
                logger.error("Appwrite save_jobs create HTTP %s: %s",
                             r2.status_code, r2.text[:200])
                return False
        elif r.status_code not in (200, 204):
            logger.error("Appwrite save_jobs HTTP %s: %s", r.status_code, r.text[:200])
            return False
        _APPWRITE_LAST_OK = time.time()
        return True
    except Exception as exc:
        logger.error("Appwrite save_jobs excepción: %s", exc)
        return False


def _appwrite_load_jobs() -> int:
    """Carga JOBS desde el documento relay_jobs en Appwrite. Idéntico postproceso
    que _supabase_load_jobs: pending/awaiting_review pasan a error.

    B3: trata 404 como "no hay snapshot guardado" sin emitir ERROR (es estado
    válido en el primer arranque del proyecto)."""
    global _APPWRITE_LAST_OK
    if not _appwrite_enabled():
        return 0
    try:
        r = DB_HTTP_CLIENT.get(_appwrite_jobs_doc_url(), headers=_appwrite_headers())
        if r.status_code == 404:
            logger.info("🔷 Appwrite doc %s no existe (primer arranque)",
                        APPWRITE_JOBS_DOCUMENT_ID)
            return 0
        if r.status_code != 200:
            logger.error("Appwrite load_jobs HTTP %s: %s", r.status_code, r.text[:200])
            return 0
        doc = r.json()
        raw = doc.get("data")
        if not raw: return 0
        # Appwrite guarda el snapshot como string; deserializamos a dict.
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict) or not data:
            return 0
        rescued, terminated, skipped = 0, 0, 0
        terminated_ids: list = []
        with LOCK:
            for jid, j in data.items():
                if not isinstance(j, dict):
                    skipped += 1
                    continue
                _coerce_job_floats(j)  # B4
                if j.get("status") in ("pending", "awaiting_review"):
                    j["status"]  = "error"
                    j["error"]   = "Relay reiniciado durante el procesamiento"
                    j["finished"] = time.time()
                    terminated += 1
                    terminated_ids.append(jid)
                j["img"]   = None
                j["video"] = None
                JOBS[jid] = j
                rescued += 1
        if terminated:
            logger.error(
                "🔷 %d job(s) en curso marcados como error tras restart desde Appwrite (ids: %s)",
                terminated, ", ".join(jid[:8] for jid in terminated_ids[:10]),
            )
        _APPWRITE_LAST_OK = time.time()
        return rescued
    except Exception as exc:
        logger.error("Appwrite load_jobs excepción: %s", exc)
        return 0


# ─── ERROR LOG en Supabase + Appwrite (sobrevive cold start) ─────────────────
# FIX: el _ERROR_LOG solo persistía en disco local (relay_errors.json), que es
# EFÍMERO en Render free tier. Tras cada cold start o re-deploy el deque
# quedaba vacío aunque los jobs ERROR sí sobrevivieran en Supabase. El usuario
# veía "1 ERROR" en el panel y "0 logs" — incoherente.
#
# Ahora replicamos a las mismas DBs cloud que JOBS:
#   - Supabase: row `id=2` de la tabla `relay_jobs` (reutilizamos schema).
#   - Appwrite: documento `relay_errors` en la colección `jobs`.
# Al arrancar, si el disco no tiene errores, intentamos las DBs.
SUPABASE_ERRORS_URL = f"{SUPABASE_URL}/rest/v1/relay_jobs?id=eq.2"
APPWRITE_ERRORS_DOCUMENT_ID = os.environ.get("APPWRITE_ERRORS_DOCUMENT_ID",
                                              "relay_errors")


def _supabase_save_errors() -> bool:
    """Vuelca _ERROR_LOG a la row id=2 de relay_jobs. Best-effort: si falla,
    el disco local sigue siendo el principal y el siguiente flush reintenta."""
    try:
        with _ERROR_LOG_LOCK:
            errs = list(_ERROR_LOG)
        # Cap defensivo: si por algún bug se acumularan miles, el PATCH a
        # Supabase con >1MB falla. ERROR_LOG_MAX ya limita a 200 pero por
        # si acaso truncamos cualquier dato extra grande.
        slim = []
        for e in errs[-ERROR_LOG_MAX:]:
            if isinstance(e, dict):
                ee = dict(e)
                # Recortar msg muy largos defensivamente
                m = ee.get("msg")
                if isinstance(m, str) and len(m) > 1000:
                    ee["msg"] = m[:1000]
                slim.append(ee)
        from datetime import datetime, timezone
        payload = {"data": slim,
                    "updated_at": datetime.now(timezone.utc).isoformat()}
        headers = {**SUPABASE_HEADERS, "Prefer": "return=representation"}
        r = DB_HTTP_CLIENT.patch(SUPABASE_ERRORS_URL, headers=headers, json=payload)
        if r.status_code == 200:
            try:
                rows_after = r.json() if r.text else []
            except Exception:
                rows_after = None
            if rows_after == []:
                # La row id=2 no existe — crear con POST upsert.
                create_url = f"{SUPABASE_URL}/rest/v1/relay_jobs"
                create_headers = {**SUPABASE_HEADERS,
                                  "Prefer": "resolution=merge-duplicates,return=minimal"}
                create_payload = {"id": 2, **payload}
                r2 = DB_HTTP_CLIENT.post(create_url, headers=create_headers,
                                          json=create_payload)
                if r2.status_code not in (200, 201, 204):
                    return False
        elif r.status_code not in (200, 204):
            return False
        return True
    except Exception:
        return False  # no logueamos para no entrar en loop infinito


def _supabase_load_errors() -> int:
    """Carga el _ERROR_LOG desde la row id=2 de relay_jobs."""
    try:
        url = f"{SUPABASE_URL}/rest/v1/relay_jobs?id=eq.2&select=data"
        r = DB_HTTP_CLIENT.get(url, headers=SUPABASE_HEADERS)
        if r.status_code != 200:
            return 0
        rows = r.json()
        if not rows: return 0
        data = rows[0].get("data") or []
        if not isinstance(data, list):
            return 0
        loaded = 0
        with _ERROR_LOG_LOCK:
            for e in data[-ERROR_LOG_MAX:]:
                if isinstance(e, dict):
                    _ERROR_LOG.append(e)
                    loaded += 1
        return loaded
    except Exception:
        return 0


def _appwrite_errors_doc_url() -> str:
    return (f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}"
            f"/collections/{APPWRITE_JOBS_COLLECTION_ID}/documents/{APPWRITE_ERRORS_DOCUMENT_ID}")


def _appwrite_save_errors() -> bool:
    """Vuelca _ERROR_LOG al documento relay_errors de Appwrite."""
    if not _appwrite_enabled():
        return False
    try:
        with _ERROR_LOG_LOCK:
            errs = list(_ERROR_LOG)
        slim = []
        for e in errs[-ERROR_LOG_MAX:]:
            if isinstance(e, dict):
                ee = dict(e)
                m = ee.get("msg")
                if isinstance(m, str) and len(m) > 1000:
                    ee["msg"] = m[:1000]
                slim.append(ee)
        # Appwrite guarda como string JSON.
        payload = {"data": {"data": json.dumps(slim, ensure_ascii=False)}}
        r = DB_HTTP_CLIENT.patch(_appwrite_errors_doc_url(),
                                  headers=_appwrite_headers(), json=payload)
        if r.status_code in (200, 204):
            return True
        # Si no existe, POST con upsert.
        if r.status_code == 404:
            create_url = (f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}"
                          f"/collections/{APPWRITE_JOBS_COLLECTION_ID}/documents")
            create_payload = {
                "documentId": APPWRITE_ERRORS_DOCUMENT_ID,
                "data": {"data": json.dumps(slim, ensure_ascii=False)},
            }
            r2 = DB_HTTP_CLIENT.post(create_url,
                                      headers=_appwrite_headers(),
                                      json=create_payload)
            return r2.status_code in (200, 201)
        return False
    except Exception:
        return False


def _appwrite_load_errors() -> int:
    """Carga el _ERROR_LOG desde Appwrite."""
    if not _appwrite_enabled():
        return 0
    try:
        r = DB_HTTP_CLIENT.get(_appwrite_errors_doc_url(),
                                headers=_appwrite_headers())
        if r.status_code != 200:
            return 0
        doc = r.json()
        raw = doc.get("data")
        if not raw: return 0
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, list):
            return 0
        loaded = 0
        with _ERROR_LOG_LOCK:
            for e in data[-ERROR_LOG_MAX:]:
                if isinstance(e, dict):
                    _ERROR_LOG.append(e)
                    loaded += 1
        return loaded
    except Exception:
        return 0


# Cargar jobs persistidos AL ARRANCAR. Orden: disco primero (rápido y completo
# si está fresco), Supabase si el FS efímero del dyno perdió relay_jobs.json,
# Appwrite como tercer escalón si Supabase también está caído.
_loaded_jobs = _load_jobs_from_file()
if _loaded_jobs > 0:
    logger.info("📁 %d jobs cargados desde %s", _loaded_jobs, JOBS_FILE)
else:
    _loaded_jobs_supa = _supabase_load_jobs()
    if _loaded_jobs_supa > 0:
        logger.info("☁ %d jobs cargados desde Supabase (FS efímero — fallback)", _loaded_jobs_supa)
    else:
        _loaded_jobs_appw = _appwrite_load_jobs()
        if _loaded_jobs_appw > 0:
            logger.info("🔷 %d jobs cargados desde Appwrite (Supabase y FS no disponibles)", _loaded_jobs_appw)

# Cargar log de errores anteriores: disco → Supabase → Appwrite (mismo orden
# que JOBS). En Render free tier el disco es efímero, así que tras cada cold
# start los errores se recuperan desde las DBs cloud — esto es lo que arregla
# el bug "panel muestra 1 ERROR pero 0 logs".
_loaded_errs = _load_errors_from_file()
if _loaded_errs > 0:
    logger.info("📁 %d errores cargados desde %s", _loaded_errs, ERROR_LOG_FILE)
else:
    _loaded_errs_supa = _supabase_load_errors()
    if _loaded_errs_supa > 0:
        logger.info("☁ %d errores cargados desde Supabase (FS efímero — fallback)",
                    _loaded_errs_supa)
    else:
        _loaded_errs_appw = _appwrite_load_errors()
        if _loaded_errs_appw > 0:
            logger.info("🔷 %d errores cargados desde Appwrite (Supabase y FS no disponibles)",
                        _loaded_errs_appw)

def _errlog_flusher_loop():
    """Vuelca el log de errores a disco + Supabase + Appwrite con debounce.
    Disco es write-through inmediato; Supabase es la fuente de verdad entre
    re-deploys del FS efímero de Render free tier; Appwrite es el tercer
    espejo. Cada DB se llama independientemente — que una falle no impide
    que la otra escriba.

    Sin la persistencia a las DBs, el panel mostraba "1 ERROR + 0 logs" tras
    cada cold start porque _ERROR_LOG vivía solo en RAM + disco efímero."""
    _ERROR_LOG_DIRTY.wait()
    _ERROR_LOG_DIRTY.clear()
    time.sleep(2.0)
    _ERROR_LOG_DIRTY.clear()
    _save_errors_to_file()
    # Best-effort × 2 DBs en paralelo (igual patrón que flusher de jobs).
    def _safe_supa_err():
        try: _supabase_save_errors()
        except Exception: pass
    def _safe_appw_err():
        try: _appwrite_save_errors()
        except Exception: pass
    t_supa = threading.Thread(target=_safe_supa_err, daemon=True, name="flush-err-supa")
    t_appw = threading.Thread(target=_safe_appw_err, daemon=True, name="flush-err-appw")
    t_supa.start(); t_appw.start()
    t_supa.join(timeout=15.0)
    t_appw.join(timeout=15.0)


class NoApiKeyError(RuntimeError):
    """Marcador específico: el proveedor NO tiene key configurada (ni primaria ni
    respaldo). Distinto de un fallo real del proveedor — el job NO debe marcarse
    como error solo porque una IA opcional carezca de credenciales."""
    pass


def _try_with_backup(primary_key: str, backup_key: str, fn) -> dict:
    """Ejecuta fn(api_key). Si la primaria falla y existe la de respaldo, reintenta con esta.
    Devuelve el resultado de la primera que funcione."""
    primary_key = (primary_key or "").strip()
    backup_key  = (backup_key  or "").strip()
    if not primary_key and not backup_key:
        raise NoApiKeyError("Sin API key configurada (ni primaria ni respaldo)")
    if primary_key:
        try:
            return fn(primary_key)
        except Exception as exc_primary:
            if not backup_key:
                raise
            logger.warning("Primary key falló (%s) → intentando backup", type(exc_primary).__name__)
            try:
                return fn(backup_key)
            except Exception as exc_backup:
                raise RuntimeError(f"Primaria: {exc_primary} | Backup: {exc_backup}")
    # Solo hay backup
    return fn(backup_key)


def _http_post_with_retry(url: str, json_payload: dict, headers: Optional[dict] = None,
                          retries: int = 2, backoff_s: float = 1.5,
                          timeout: Optional[float] = None) -> "httpx.Response":
    """Wrap de HTTP_CLIENT.post con reintentos automáticos en errores transitorios:
    - 429 (rate limit), 502/503/504 (sobrecarga del proveedor o gateway)
    - httpx.TimeoutException (RTT > timeout configurado)
    - httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError (red parpadea)
    - cualquier httpx.HTTPError (categoría general)

    Espera `backoff_s * (i+1)` entre reintentos. Tras agotarlos:
    - Si el último intento fue Response → devuelve la Response (caller decide).
    - Si el último intento fue excepción → re-raise la última excepción.

    Si se pasa `timeout` (en segundos), se aplica por-request sobreescribiendo
    el read-timeout del cliente compartido (90s por defecto). Útil para GPT-5.5
    con web_search agéntico que puede tardar hasta 120s.

    Antes esta función solo reintentaba HTTP 429/5xx. Si httpx lanzaba (caso típico
    cuando la red del relay parpadea o el proveedor cierra la conexión), la excepción
    se propagaba al primer intento, agotando la key primaria sin reintento real.
    """
    short_url = url.split("?")[0][-50:]
    last_response: Optional["httpx.Response"] = None
    last_exception: Optional[BaseException] = None
    # Si nos pasaron timeout custom, construimos un Timeout completo (connect+
    # read+write+pool) escalado al límite pedido. read=timeout es lo crítico.
    req_timeout = None
    if timeout is not None:
        req_timeout = httpx.Timeout(
            connect=min(10.0, timeout),
            read=timeout,
            write=min(20.0, timeout),
            pool=min(5.0,  timeout),
        )
    for attempt in range(retries + 1):
        try:
            if req_timeout is not None:
                last_response = HTTP_CLIENT.post(url, json=json_payload,
                                                  headers=headers or {},
                                                  timeout=req_timeout)
            else:
                last_response = HTTP_CLIENT.post(url, json=json_payload, headers=headers or {})
            last_exception = None
            # Éxito → devolver inmediatamente
            if last_response.is_success:
                return last_response
            # No reintentamos errores 4xx que no sean 429 (son fallos del cliente)
            if last_response.status_code not in (429, 502, 503, 504):
                return last_response
            if attempt < retries:
                wait = backoff_s * (attempt + 1)
                logger.warning("HTTP %d en %s → reintentando en %.1fs (intento %d/%d)",
                               last_response.status_code, short_url, wait, attempt + 1, retries)
                time.sleep(wait)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError,
                httpx.RemoteProtocolError, httpx.WriteError) as e:
            last_exception = e
            last_response = None
            if attempt < retries:
                wait = backoff_s * (attempt + 1)
                logger.warning("httpx %s en %s → reintentando en %.1fs (intento %d/%d): %s",
                               type(e).__name__, short_url, wait, attempt + 1, retries, e)
                time.sleep(wait)
        except httpx.HTTPError as e:
            # Categoría más general de httpx — también reintentable.
            last_exception = e
            last_response = None
            if attempt < retries:
                wait = backoff_s * (attempt + 1)
                logger.warning("httpx HTTPError en %s → reintentando en %.1fs (intento %d/%d): %s",
                               short_url, wait, attempt + 1, retries, e)
                time.sleep(wait)
    # Tras agotar reintentos
    if last_response is not None:
        return last_response
    # Si solo hubo excepciones, re-raise la última con contexto.
    raise RuntimeError(
        f"HTTP_POST tras {retries + 1} intentos falló en {short_url}: "
        f"{type(last_exception).__name__ if last_exception else 'unknown'}: {last_exception}"
    ) from last_exception


API_KEY_CLIENTE = os.environ.get("API_KEY_CLIENTE", "")
EDITOR_KEY      = os.environ.get("EDITOR_KEY",      "")

# JOBS y LOCK están declarados arriba (sección persistencia de JOBS) porque
# _load_jobs_from_file() los necesita al arrancar el módulo.
_START_TIME = time.time()

# Campos pesados (base64 grande) que se EXCLUYEN de listados/polling.
# Se sirven sólo bajo demanda vía /api/image y /api/video.
_HEAVY_FIELDS: tuple = ("img", "video", "extracted_frames", "context_image",
                         "fused_image_b64", "fused_image_local_b64",
                         "context_image_provided_b64")

# Campos por-provider que NO viajan en /api/jobs (listado): son textos largos
# (razonamiento raw de cada IA, transcripción OCR completa) que inflan el
# polling × 5-10. El detalle/expediente del job los obtiene on-demand vía
# /api/partial/{job_id} cuando el usuario expande "Ver/Editar".
_HEAVY_RESPONSE_FIELDS: tuple = ("raw",)
_HEAVY_OCR_FIELDS:      tuple = ("text",)


def _strip_heavy_for_list(job: dict) -> dict:
    """Devuelve una copia ligera de un job para listados/polling: sin imágenes,
    sin video, sin raws de IA, sin transcripciones OCR completas. Conserva todos
    los metadatos (status, answer, ms, model, error, edited, fusion, etc.) para
    que el panel pueda renderizar la lista. El detalle completo se obtiene
    vía /api/partial/{job_id}.

    NUNCA debe lanzar (lo llama el polling cada 1.5s). Si el job tiene una
    forma rara, devuelve lo que se pueda y deja el resto sin filtrar.
    """
    try:
        entry = {k: v for k, v in job.items() if k not in _HEAVY_FIELDS}
        # Filtrar responses[*].raw — pero antes precomputamos el char-count en
        # raw_chars para que el panel pueda mostrar "✓ N chars" sin tener que
        # enviar los 5000+ chars del raw en cada poll de /api/jobs.
        resps = entry.get("responses")
        if isinstance(resps, dict):
            slim = {}
            for prov, r in resps.items():
                if isinstance(r, dict):
                    slim_r = {k: v for k, v in r.items() if k not in _HEAVY_RESPONSE_FIELDS}
                    raw_full = r.get("raw")
                    if isinstance(raw_full, str):
                        slim_r["raw_chars"] = len(raw_full)
                    slim[prov] = slim_r
                else:
                    slim[prov] = r
            entry["responses"] = slim
        # Filtrar ocr_results[*].text — mismo patrón: precomputamos `chars` (len
        # del text original) y `preview` (primeros 80 chars) para que la fila
        # del panel pueda renderizar "✅ N chars" + preview sin necesidad del
        # text completo. El text completo sigue llegando vía /api/partial/{id}
        # cuando el usuario expande "Ver".
        ocrs = entry.get("ocr_results")
        if isinstance(ocrs, dict):
            slim_ocr = {}
            for prov, r in ocrs.items():
                if isinstance(r, dict):
                    slim_r = {k: v for k, v in r.items() if k not in _HEAVY_OCR_FIELDS}
                    txt_full = r.get("text")
                    if isinstance(txt_full, str):
                        slim_r["chars"]   = len(txt_full)
                        slim_r["preview"] = txt_full[:80]
                    slim_ocr[prov] = slim_r
                else:
                    slim_ocr[prov] = r
            entry["ocr_results"] = slim_ocr
        # ocr_fused_text también puede ser largo; lo enviamos truncado para mostrar
        # un preview en el listado sin pagar el peso completo. Guardamos el len
        # original en ocr_fused_text_full_chars para que el panel muestre el
        # contador real (no "200" del truncado).
        fused = entry.get("ocr_fused_text")
        if isinstance(fused, str):
            entry["ocr_fused_text_full_chars"] = len(fused)
            if len(fused) > 200:
                entry["ocr_fused_text"] = fused[:200]
                entry["_ocr_fused_truncated"] = True
        # tavily_block (el XML <internet_context> inyectado al prompt) puede pesar
        # 3-15 KB cuando hay muchas preguntas. Mismo patrón que ocr_fused_text:
        # en el listado mandamos solo los chars y la cabecera, el bloque completo
        # se sirve vía /api/partial/{id}.
        tav = entry.get("tavily_block")
        if isinstance(tav, str):
            entry["tavily_block_full_chars"] = len(tav)
            if len(tav) > 200:
                entry["tavily_block"] = tav[:200]
                entry["_tavily_block_truncated"] = True
        return entry
    except Exception:
        # Defensa: si algo falla aquí, NO rompemos el listado completo —
        # devolvemos al menos el job sin filtrar campos por-provider.
        return {k: v for k, v in job.items() if k not in _HEAVY_FIELDS}

# Cola de correcciones (cuando el cómplice se da cuenta de que envió mal una respuesta
# anterior, la cola guarda la corrección hasta que el móvil la consume en su próximo poll).
CORRECTIONS: list = []
CORRECTIONS_LOCK = threading.Lock()
CORRECTIONS_FILE = Path(os.environ.get("CORRECTIONS_FILE", "relay_corrections.json"))


def _save_corrections_to_file():
    """B15: deepcopy DENTRO del lock para que json.dumps no vea mutaciones
    concurrentes en sub-dicts (`c["delivered"] = True` desde otro hilo entre
    `list(CORRECTIONS)` y `_safe_write_json`)."""
    import copy
    try:
        with CORRECTIONS_LOCK:
            data = copy.deepcopy(CORRECTIONS)
        _safe_write_json(CORRECTIONS_FILE, {"items": data})
    except Exception as exc:
        logger.error("save_corrections: %s", exc)


def _load_corrections_from_file():
    data = _safe_read_json(CORRECTIONS_FILE)
    if not data or not isinstance(data.get("items"), list):
        return 0
    with CORRECTIONS_LOCK:
        CORRECTIONS.clear()
        CORRECTIONS.extend([c for c in data["items"] if isinstance(c, dict)])
        return len(CORRECTIONS)


# Cargar correcciones pendientes (después de que sus helpers estén definidos)
_loaded_corr = _load_corrections_from_file()
if _loaded_corr > 0:
    logger.info("📁 %d correcciones cargadas desde %s", _loaded_corr, CORRECTIONS_FILE)


# ─── Límites duros (evitan acumulación de basura) ────────────────────────────
JOB_TTL          = int(os.environ.get("JOB_TTL",          "7200"))   # 2h — TTL absoluto. Subido de 1h→2h: a ~6min/ciclo real, 15 jobs abarcan 90min; con 1h solo sobrevivían ~10 y "últimos 15" no se cumplía. Solo afecta METADATOS en RAM (la media sigue capada a MEDIA_KEEP_RECENT=15 por recencia; JOBS_MAX=150 sigue de tope absoluto).
JOBS_MAX         = int(os.environ.get("JOBS_MAX",         "150"))    # cap absoluto del dict
IMG_PRUNE_AFTER  = int(os.environ.get("IMG_PRUNE_AFTER",  "1800"))   # 30min tras done → libera img/video
VIDEO_PRUNE_AFTER = int(os.environ.get("VIDEO_PRUNE_AFTER","1800"))  # 30min — videos pesan más, mismo timeout
# Cap DURO de media en RAM: conservamos los blobs (video/img/frames/fused/context)
# SOLO de los N jobs más recientes. Independiente del tiempo → la RAM de media
# queda acotada pase lo que pase (cadencia de ciclos, bitrate). Es lo que permite
# "ver siempre los últimos N vídeos y fotos" sin acumular el resto. 512MB Render.
# En Render Standard 2GB hay margen para retener los blobs (video, frames,
# fused) de los 15 últimos jobs. Total ~75MB en steady state — holgado en 2GB.
# Si bajas a free tier, cambia esto a 5 (export MEDIA_KEEP_RECENT=5).
MEDIA_KEEP_RECENT = int(os.environ.get("MEDIA_KEEP_RECENT", "15"))
STUCK_TIMEOUT    = int(os.environ.get("STUCK_TIMEOUT",    "900"))    # 15min → techo global. C13: era 540s (9min) pero AI_HARD_TIMEOUT=360s × 2 retries + OCR 90s + Tavily 8s = 818s, así que un job con retry legítimo lo mataba. 900s da margen seguro.
WATCHDOG_PERIOD  = int(os.environ.get("WATCHDOG_PERIOD",  "30"))     # cada 30s revisa estado global
# Guard anti-alucinación OCR: si MÁS de N OCRs corren pero NO detectan texto
# (empty/NO_LEGIBLE), el folio es casi seguro ilegible y las pocas OCR que sí
# "leyeron" algo probablemente lo ALUCINARON → marcamos el job error en vez de
# vibrar respuestas inventadas. Las OCR con error de red/API o sin key NO cuentan
# aquí (no procesaron la imagen → no son señal de legibilidad). Hay 9 OCRs; >4
# sin texto = la mayoría no pudo leerlo. Configurable por env.
OCR_NO_TEXT_ERROR_THRESHOLD = int(os.environ.get("OCR_NO_TEXT_ERROR_THRESHOLD", "4"))

# Estado de los daemons (para diagnóstico)
DAEMON_HEALTH = {
    "gc":        {"last_ok": 0.0, "errors": 0},
    "flusher":   {"last_ok": 0.0, "errors": 0},
    "watchdog":  {"last_ok": 0.0, "errors": 0},
}

# httpx Client compartido → reusa conexiones, evita socket exhaustion.
# En Render Standard 2GB+ hay holgura para 40 conexiones simultáneas (cubre
# 9 OCRs + 6 analyzers en paralelo + posibles reintentos sin pool timeout).
# Tavily tiene su pool dedicado aparte.
HTTP_CLIENT = httpx.Client(
    timeout=httpx.Timeout(connect=10.0, read=90.0, write=20.0, pool=8.0),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=40,
                         keepalive_expiry=30.0),
)

# Cliente DEDICADO para Tavily — pool propio para que N búsquedas en paralelo
# (1 por pregunta del examen, puede ser 10-20) NO agoten el pool del HTTP_CLIENT
# compartido que usan los analyzers (OCR + razonamiento). Sin este aislamiento,
# 20 hilos Tavily simultáneos saturaban max_connections=20 y los analyzers caían
# con pool timeout. Pool propio: 25 conexiones máximas, exclusivo para tavily.com.
TAVILY_HTTP_CLIENT = httpx.Client(
    timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=3.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=25,
                         keepalive_expiry=20.0),
)

app = FastAPI(title="Bolsillo IA Relay")


# ─── Global exception handler ─────────────────────────────────────────────────
# FIX FIABILIDAD: cualquier excepción NO manejada por un endpoint de FastAPI
# era convertida en 500 por uvicorn pero solo se logueaba en `uvicorn.error`,
# que puede no propagar al root logger según versión. Resultado: el panel
# mostraba "HTTP 502" pero "0 logs" — el operador no veía la causa raíz.
#
# Ahora capturamos toda excepción que escape de un handler y la logueamos
# como ERROR al deque _ERROR_LOG con contexto (método + path + tipo + msg).
#
# IMPORTANTE: NO capturamos HTTPException — esa es la vía oficial de FastAPI
# para devolver 4xx con detail. Si la capturáramos, un `raise HTTPException(404)`
# se convertiría en 500 con detail genérico, rompiendo todos los endpoints
# que usan 404 (job no encontrado, auth fallida, etc.).
from fastapi import Request as _Request
from fastapi.responses import JSONResponse as _JSONResponse
from starlette.exceptions import HTTPException as _StarletteHTTPException


@app.exception_handler(Exception)
async def _global_exception_handler(request: _Request, exc: Exception):
    """Loguea CUALQUIER excepción no manejada a _ERROR_LOG (visible en /api/errors).
    Sin esto, errores raros (KeyError, AttributeError, NPE) de handlers se
    convertían en 500/502 sin trazo en el panel.

    EXCLUSIÓN CRÍTICA: HTTPException (y subclases) NO se procesan aquí — se
    relanzan para que el handler default de FastAPI las procese y devuelva
    el código real (404, 401, 422, etc.) con el `detail` apropiado. Sin esta
    exclusión, todos los 4xx legítimos se convertirían en 500 genéricos."""
    if isinstance(exc, (HTTPException, _StarletteHTTPException)):
        # Re-elevar para que el handler default de FastAPI procese y devuelva
        # el código y detail correctos. NOTA: en FastAPI moderna esto suele
        # no llegar aquí (HTTPException tiene precedencia de tipo), pero
        # protegemos por si una versión futura cambia el orden.
        raise exc
    try:
        method = request.method
        path = request.url.path
        client = request.client.host if request.client else "?"
        logger.error(
            "💥 HTTP handler EXC %s %s · client=%s · %s: %s",
            method, path, client, type(exc).__name__, exc,
        )
        # Stack trace al stdout para Render logs (no al deque para no inflar).
        traceback.print_exc(file=sys.stdout)
    except Exception:
        # Si el logging falla, NO debe impedir devolver el 500 al cliente.
        pass
    # Devolvemos 500 con detail genérico (no exponemos detalles internos al
    # cliente; los detalles van solo al log persistido).
    return _JSONResponse(
        status_code=500,
        content={"detail": "internal server error",
                  "type": type(exc).__name__},
    )


# ─── Defensa: límite de tamaño de body ───────────────────────────────────────
# Una imagen JPEG de 8 MP en base64 ≈ 2-4 MB. Un MP4 de 3-5s a 720p ≈ 2-5 MB
# (≈3-7 MB en base64). Damos margen razonable para video sin exponerse a DoS:
# rechazamos cualquier cosa por encima de 50 MB.
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(50 * 1024 * 1024)))  # 50 MB


@app.middleware("http")
async def limit_body_size(request, call_next):
    # 1) Si el cliente envía Content-Length, rechazamos rápido sin leer el body.
    cl = request.headers.get("content-length")
    if cl:
        try:
            if int(cl) > MAX_BODY_BYTES:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Body demasiado grande (>{MAX_BODY_BYTES // (1024*1024)} MB)"},
                )
        except ValueError:
            pass
    # 2) B16: si NO hay Content-Length (chunked transfer-encoding), envolvemos
    #    el body stream para cortarlo en cuanto supere MAX_BODY_BYTES. Sin esto
    #    un cliente puede mandar `Transfer-Encoding: chunked` con GB y agotar RAM.
    if not cl and request.method in ("POST", "PUT", "PATCH"):
        # Solo rechazamos si efectivamente excede. Para no romper el contrato,
        # leemos el body con un cap explícito.
        try:
            body = b""
            async for chunk in request.stream():
                body += chunk
                if len(body) > MAX_BODY_BYTES:
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Body sin Content-Length excede {MAX_BODY_BYTES // (1024*1024)} MB"},
                    )
            # Replay del body para downstream — Starlette re-lee del receive
            # cuando el handler lee request.body(). Esto requiere un trick:
            # parcheamos request._receive con un closure que devuelve el cuerpo
            # buffered. Sin esto, el handler ve body vacío.
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            request._receive = receive
        except Exception:
            # Si falla la lectura, dejamos que el handler lo gestione.
            pass
    return await call_next(request)


# A8 + C7: middleware de cabeceras de seguridad. No bloquea nada por sí mismo
# pero limita el blast radius de XSS/key-leakage:
#   - Referrer-Policy: no-referrer → la query string ?key=... NO viaja al
#     siguiente dominio si el operador clica un link externo desde el panel.
#   - X-Content-Type-Options: nosniff → MIME sniffing apagado (defensa anti
#     content-type confusion).
#   - X-Frame-Options: DENY → no embebible en iframe, previene clickjacking.
#   - Content-Security-Policy: solo en /panel y derivados. El JS y CSS son
#     inline, así que necesitamos 'unsafe-inline'; al menos limitamos script-src
#     a 'self' para que ningún <script src=evil.com> cargue.
@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    path = request.url.path
    if path == "/panel" or path.startswith("/api/"):
        # CSP estricto para panel y API responses. Permitimos 'unsafe-inline'
        # porque el JS del panel está embebido en el HTML (refactor mayor para
        # quitarlo); el riesgo se mitiga con script-src 'self' + escapado.
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
            "media-src 'self' blob:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'"
        )
    # FIX FIABILIDAD: loguear cualquier respuesta 5xx (server errors). Captura
    # respuestas 500/502/503/504 que vienen de cualquier sitio (HTTPException
    # manual, errores de middleware downstream, etc.) y que NO pasaron por el
    # exception_handler global. Visible en VER ERRORES.
    try:
        if response.status_code >= 500:
            logger.error(
                "💥 HTTP %d en %s %s (response.status_code)",
                response.status_code, request.method, request.url.path,
            )
    except Exception:
        pass
    return response


# ─── Modelos ──────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    prompt: str
    # system es opcional: el cliente puede omitirlo (modo VIDEO suele no enviarlo
    # porque el prompt principal ya contiene las instrucciones). Default "" para
    # que las funciones de IA no exploten al concatenarlo.
    system: str = ""
    # Imagen (modo legacy de ráfaga → 1 frame seleccionado por BurstStackingProcessor)
    image_b64: Optional[str] = None
    image_mime: str = "image/jpeg"
    # Video (modo nuevo → OCR fase 1 con Qwen3-VL + Gemini Video, después análisis texto)
    video_b64: Optional[str] = None
    video_mime: str = "video/mp4"
    # Imagen de contexto (diagrama del caso práctico) — opcional. Cuando llega,
    # se manda como SEGUNDA imagen al analyzer ANTES de la foto/video "buena",
    # respetando la convención del prompt "IMAGEN 1: diagrama, IMAGEN 2: hoja".
    # Se reusa en TODAS las solicitudes posteriores hasta que:
    #   • llegue otro context_image_b64 distinto (reemplazo)
    #   • se llame /reset o se pulse "Quitar contexto" en el panel
    #   • expire el TTL (STICKY_CONTEXT_TTL_S)
    # AHORA hay DOS vías (no excluyentes): el cliente Android puede guardarlo en
    # local (ctx_diagram.jpg) y reenviarlo en cada /ask (CONTEXTO_VISUAL:SI →
    # saveContextImage en HeadlessUvcService.kt) Y/O el RELAY lo recuerda
    # server-side (_STICKY_CONTEXT): si la /ask no trae imagen-contexto, el relay
    # inyecta la última marcada con la casilla 🖼️ del panel. Así el diagrama
    # llega a las IAs aunque el móvil no lo reenvíe.
    context_image_b64: Optional[str] = None
    context_image_mime: str = "image/jpeg"
    # Imagen fusionada (stacking local + opcional Topaz Wonder 3). NO viene
    # del cliente — el relay la genera internamente en _fuse_pipeline_bg y
    # se la asigna a `req` para que las OCR funcs page-based (Mistral,
    # DeepSeek, GLM) la prefieran sobre el frame crudo del Laplacian.
    # Declarada aquí para que Pydantic permita la asignación sin error.
    fused_image_b64: Optional[str] = None
    fused_image_mime: Optional[str] = None
    # Clamp a [0, 200] en el propio modelo: cierra DE RAÍZ el vector OOM de
    # `'X' * (expected - len(...))` en _publish_partial (~3425) y PATCH
    # /api/response (~8909), donde `expected` sale de job["expected_questions"]
    # SIN cota. Un valor basura (bug del cliente → p.ej. 2_000_000_000) ahí
    # construiría una cadena de varios GB → MemoryError → worker caído. Ningún
    # examen real supera 200 preguntas (misma invariante que align_answer/fusionar).
    expected_questions: Optional[int] = Field(default=None, ge=0, le=200)
    review_timeout_seconds: int = 90
    # Telemetría SIM (opcional): el cliente Android la rellena con la SIM elegida
    # por SimCoverageSelector + su dBm en el momento del envío, para que el panel
    # muestre con qué cobertura se mandó la foto. Si el cliente no la envía
    # (versión antigua o debugSinAvion), los campos quedan None y el panel los oculta.
    sim_operator: Optional[str] = None
    sim_dbm: Optional[int] = None
    sim_slot: Optional[int] = None
    sim_summary: Optional[str] = None

    @field_validator('image_b64', 'video_b64', 'context_image_b64')
    @classmethod
    def _check_b64(cls, v):
        # Validación temprana: si el cliente manda base64 inválido, lo rechazamos
        # AQUÍ con 422 en vez de gastar llamadas a los providers (que devolverían
        # 400 críticos y dispararían la cascada de fallback inútilmente).
        # Solo verificamos los primeros 4KB — basta para detectar texto corrupto
        # sin pagar el coste de decodificar 50MB enteros.
        if v is None or v == "":
            return v
        try:
            base64.b64decode(v[:4096], validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("base64 inválido")
        return v

class EditRequest(BaseModel):
    # max_length=200 evita que un cliente buggy mande una "answer" de 10MB y
    # haga al regex chew CPU. Las respuestas reales son N letras ABCDX donde
    # N rara vez pasa de 50; 200 da margen 4× sin riesgo.
    answer: Optional[str] = Field(default=None, max_length=200)
    has_image: Optional[bool] = None

class CorrectionRequest(BaseModel):
    # Misma razón que EditRequest.answer (defensa contra payloads gigantes).
    answer: str = Field(..., max_length=200)  # letras corregidas (ABCDX)
    page:   Optional[int] = Field(default=None, ge=1, le=999)  # nº pregunta


class ConfigUpdateRequest(BaseModel):
    # API keys (string vacío = limpiar, None = no tocar)
    anthropic_key:        Optional[str] = None
    openai_key:           Optional[str] = None
    gemini_key:           Optional[str] = None
    deepseek_key:         Optional[str] = None
    mistral_key:          Optional[str] = None
    kimi_key:             Optional[str] = None
    qwen_key:             Optional[str] = None
    mimo_key:             Optional[str] = None
    anthropic_key_backup: Optional[str] = None
    openai_key_backup:    Optional[str] = None
    gemini_key_backup:    Optional[str] = None
    deepseek_key_backup:  Optional[str] = None
    mistral_key_backup:   Optional[str] = None
    kimi_key_backup:      Optional[str] = None
    qwen_key_backup:      Optional[str] = None
    mimo_key_backup:      Optional[str] = None
    # Modelos (editables; los defaults del código son el fallback).
    # Nota: kimi NO tiene modelo "chat-only" — sólo OCR de video (kimi_video_model).
    anthropic_model:      Optional[str] = None
    openai_model:         Optional[str] = None
    gemini_model:         Optional[str] = None
    deepseek_model:       Optional[str] = None
    mistral_model:        Optional[str] = None
    mimo_model:           Optional[str] = None
    qwen_video_model:     Optional[str] = None
    gemini_video_model:   Optional[str] = None
    kimi_video_model:     Optional[str] = None
    mimo_video_model:     Optional[str] = None
    claude_video_ocr_model: Optional[str] = None
    openai_video_ocr_model: Optional[str] = None
    # NVIDIA Nemotron como ANALYZER de texto (fase 2), no como OCR.
    nvidia_model:         Optional[str] = None
    nvidia_key:           Optional[str] = None
    nvidia_key_backup:    Optional[str] = None
    # Z.AI / Zhipu (GLM-OCR) — OCR-image puro
    z_ai_key:             Optional[str] = None
    z_ai_key_backup:      Optional[str] = None
    # GPT-5.5 tuning agéntico (Responses API · web_search GA)
    openai_reasoning_effort: Optional[str] = None     # "low" | "medium" | "high"
    openai_max_tool_calls:   Optional[int] = None     # 1..20
    openai_allowed_domains:  Optional[list] = None    # lista de strings (max 100)
    # Tavily (paso intermedio OCR → razonamiento)
    tavily_key:              Optional[str] = None
    tavily_key_backup:       Optional[str] = None
    tavily_enabled:          Optional[bool] = None    # True/False
    tavily_max_results:      Optional[int] = None     # 1..10
    tavily_search_depth:     Optional[str] = None     # "basic" | "advanced"
    tavily_http_timeout_s:   Optional[float] = None   # 1..30
    tavily_total_deadline_s: Optional[float] = None   # 1..60
    # Pesos en la votación de fusionar() (None = no tocar, int >= 0)
    anthropic_weight:     Optional[int] = None
    openai_weight:        Optional[int] = None
    gemini_weight:        Optional[int] = None
    deepseek_weight:      Optional[int] = None
    mistral_weight:       Optional[int] = None
    nvidia_weight:        Optional[int] = None
    kimi_weight:          Optional[int] = None
    mimo_weight:          Optional[int] = None
    # Meta-Judge (árbitro post-fusión)
    meta_judge_enabled:           Optional[bool] = None
    meta_judge_model:             Optional[str]  = None
    meta_judge_reasoning_effort:  Optional[str]  = None     # low|medium|high
    meta_judge_timeout_s:         Optional[float] = None
    # Pesos OCR de video (fase 1 — independientes de los analyzers)
    mistral_ocr_weight:   Optional[int] = None
    gemini_ocr_weight:    Optional[int] = None
    qwen_ocr_weight:      Optional[int] = None
    glm_ocr_weight:       Optional[int] = None
    anthropic_ocr_weight: Optional[int] = None
    openai_ocr_weight:    Optional[int] = None
    deepseek_ocr_weight:  Optional[int] = None
    kimi_ocr_weight:      Optional[int] = None
    mimo_ocr_weight:      Optional[int] = None
    # Topaz Labs Image API (post-stacking enhancement opcional via Wonder 3).
    # Sin backup_key: Topaz no expone una key secundaria como los LLMs.
    topaz_key:            Optional[str]  = None
    topaz_enabled:        Optional[bool] = None
    topaz_model:          Optional[str]  = None       # "Wonder 3" | "Wonder 2"
    topaz_output_height:  Optional[int]  = None       # 0 = mantener
    topaz_timeout_s:      Optional[float] = None      # 5..180


# ─── Auth ────────────────────────────────────────────────────────────────────
# A7: hmac.compare_digest evita timing attacks. Comparar con `==` permite a un
# atacante remoto inferir bytes correctos uno a uno. Con compare_digest el
# tiempo es constante respecto al contenido.
import hmac as _hmac

def _key_eq(provided: Optional[str], expected: str) -> bool:
    """Comparación constant-time. Si `provided` es None/vacío, retorna False
    sin llamar a compare_digest (que requiere bytes/str ambos)."""
    if not provided or not expected:
        return False
    try:
        return _hmac.compare_digest(str(provided), str(expected))
    except Exception:
        return False


def check_auth(x_api_key: Optional[str]):
    if not _key_eq(x_api_key, API_KEY_CLIENTE):
        raise HTTPException(status_code=401, detail="API key inválida")


def check_editor_auth(x_api_key: Optional[str]):
    if not (_key_eq(x_api_key, API_KEY_CLIENTE) or _key_eq(x_api_key, EDITOR_KEY)):
        raise HTTPException(status_code=401, detail="API key inválida")


def _is_authorized_key(key: Optional[str]) -> bool:
    """Helper para endpoints que aceptan EDITOR_KEY o API_KEY_CLIENTE. Replaces
    `key in [API_KEY_CLIENTE, EDITOR_KEY]` que era vulnerable a timing attack."""
    return _key_eq(key, API_KEY_CLIENTE) or _key_eq(key, EDITOR_KEY)


# ─── Utilidades ───────────────────────────────────────────────────────────────
def extract_answer(text: str) -> str:
    """Extrae la respuesta final (letras A/B/C/D/X) de la salida de la IA.

    Estrategia (en orden):
      1. Buscar TODOS los bloques <FINAL>...</FINAL> (case-insensitive, multilinea)
         y usar el ÚLTIMO. Claude con thinking habilitado puede escribir varios:
         razonamiento intermedio + revisión + final. El último suele ser la
         respuesta definitiva (corrección tras reconsiderar).
      2. Si no hay <FINAL>, buscar patrones explícitos cerca del final:
         "Respuesta final:", "Final answer:", "Respuestas:" → letras tras ese marcador.
      3. Fallback duro: extraer letras del ÚLTIMO 25% del texto (más cerca de
         la respuesta final que del razonamiento inicial), capado a 100 letras
         para evitar "monton de letras" cuando la IA es verbosa.
      4. Si nada cuaja → "X" (sin respuesta legible).
    """
    if not text:
        return "X"
    # Defensa contra respuestas hostiles/malformadas: cap absoluto de 200 KB
    # antes de regex. Si un proveedor devuelve 10 MB de letras (caso real visto
    # con modelos en loop), `re.sub(r'[^ABCDX]', '', text)` cuesta segundos de
    # CPU y bloquea el hilo. Conservamos los primeros y últimos 100 KB porque
    # ahí están <FINAL> y los marcadores "respuesta final:" típicamente.
    if len(text) > 200_000:
        text = text[:100_000] + "\n…[recortado por defensa CPU]…\n" + text[-100_000:]
    # 1) Último bloque <FINAL>...</FINAL>
    finals = re.findall(r'<FINAL>(.*?)</FINAL>', text, re.I | re.S)
    if finals:
        letters = re.sub(r'[^ABCDXabcdx]', '', finals[-1]).upper()
        if letters:
            return letters
    # 2) Marcadores explícitos al final del texto
    tail = text[-1500:]   # solo el final, no todo el razonamiento
    for marker in (
        r'(?:respuesta\s+final|final\s+answer|respuestas?\s+final|answers?)\s*[:=\-]\s*([^\n]+)',
        r'(?:resultado|result)\s*[:=\-]\s*([^\n]+)',
    ):
        m = re.search(marker, tail, re.I)
        if m:
            letters = re.sub(r'[^ABCDXabcdx]', '', m.group(1)).upper()
            if letters:
                return letters[:100]    # cap defensivo
    # 3) Fallback: últimas letras del texto, capado a 100 para no devolver
    #    una sopa de letras del razonamiento entero.
    region = text[-(max(len(text) // 4, 500)):]
    letters = re.sub(r'[^ABCDXabcdx]', '', region).upper()
    if letters:
        return letters[-100:]   # máximo 100 letras (ningún examen real tiene más)
    return "X"


def align_answer(raw: str, expected: int) -> str:
    """Devuelve EXACTAMENTE `expected` letras (A/B/C/D/X)."""
    if expected <= 0:
        return extract_answer(raw)
    # Cap defensivo: `expected` viene de req.expected_questions, que NO tiene cota
    # en el modelo Pydantic. Un valor basura (p.ej. 2_000_000_000) haría que
    # `['X'] * expected` reventara la RAM (MemoryError) y pudiera tumbar el worker.
    # Ningún examen real tiene >200 preguntas (mismo cap que fusionar()).
    if expected > 200:
        expected = 200
    # Tomar el ÚLTIMO bloque <FINAL>...</FINAL> (Claude con thinking puede emitir
    # varios; el último es la decisión definitiva). Si no hay, usamos el texto
    # crudo y los detectores de patrón numerado/secuencial se encargan.
    finals = re.findall(r'<FINAL>(.*?)</FINAL>', raw, re.I | re.S)
    region = finals[-1] if finals else raw
    # Modo numerado: "1. A  2. B  4. D"
    numbered = re.findall(r'(?<!\d)(\d{1,3})\s*[.:\)\-]\s*([ABCDXabcdx])\b', region)
    if numbered:
        result = ['X'] * expected
        any_in_range = False
        for num_str, letter in numbered:
            idx = int(num_str) - 1
            if 0 <= idx < expected:
                result[idx] = letter.upper()
                any_in_range = True
        if any_in_range:
            return ''.join(result)
    # Modo secuencial: últimas N letras (las de antes son del razonamiento)
    letters = re.sub(r'[^ABCDXabcdx]', '', region).upper()
    if len(letters) >= expected:
        return letters[-expected:]
    return letters + 'X' * (expected - len(letters))


_PROVIDER_WEIGHT_KEY = {
    # Clave del provider tal como lo registra _PROVIDERS abajo.
    "claude":   "ANTHROPIC_WEIGHT",
    "gpt":      "OPENAI_WEIGHT",
    "gemini":   "GEMINI_WEIGHT",
    "deepseek": "DEEPSEEK_WEIGHT",
    "mistral":  "MISTRAL_WEIGHT",
    "nvidia":   "NVIDIA_WEIGHT",
    "minimax":  "MINIMAX_WEIGHT",
}


def _weight_for(provider: str) -> int:
    """Lee el peso configurado para un proveedor (clamp 0..10). Default 1."""
    key = _PROVIDER_WEIGHT_KEY.get(provider)
    if not key:
        return 1
    with CONFIG_LOCK:
        raw = DYNAMIC_CONFIG.get(key, 1)
    try:
        return max(0, min(10, int(raw)))
    except (TypeError, ValueError):
        return 1


def fusionar(job: dict) -> str:
    """Votación por posición PONDERADA por peso de cada IA + longitud modo
    + filtrado de outliers. Nunca lanza.

    El bug que arregla: antes la longitud final = max(len de cada respuesta),
    así que si UNA IA respondía con un párrafo entero (50 letras) en vez de
    "ABCD" (4 letras), la respuesta final tenía 50 letras → más respuestas
    que preguntas.

    Algoritmo:
      1. Determinar `target_len`:
         - Si el job conoce expected_questions > 0 → ese valor (oro).
         - Si no → moda ponderada: la longitud con más votos (peso) gana.
           En empate, la mediana de los candidatos empatados (robusto).
      2. Ajustar cada respuesta a target_len:
         - len == target → ok
         - len < target  → no se rellena con X (sesgaría); la IA simplemente
                           no vota en las posiciones que le faltan
         - target < len ≤ target*1.5  → exceso "típico": coger las últimas
                           target_len letras (el final suele ser la respuesta
                           tras razonamiento)
         - len > target*1.5 → outlier grave (parloteó): últimas target_len
                           letras pero con el peso penalizado a la mitad
      3. Voto posicional ponderado:
         - Solo cuentan A/B/C/D; 'X' = sin opinión, no vota
         - Tie-break: a igual peso total, gana la letra apoyada por la IA con
           mayor peso individual; si siguen empatadas, la primera alfabética
         - Posición sin ningún voto válido → 'X'

    Refs: weighted majority voting + outlier-aware aggregation (LLM ensembles).
    """
    try:
        from collections import Counter
        from statistics import median_low
        responses = job.get("responses") or {}
        expected_hint = 0
        try:
            expected_hint = int(job.get("expected_questions") or 0)
        except (TypeError, ValueError):
            expected_hint = 0

        # 1) Recolectar respuestas válidas con su peso
        # okays = [(provider, clean_answer, weight)]
        okays = []
        for provider, r in responses.items():
            if not isinstance(r, dict): continue
            if r.get("status") != "done": continue
            ans = r.get("answer")
            if not isinstance(ans, str) or not ans: continue
            w = _weight_for(provider)
            if w <= 0: continue
            # Sanitización defensiva: extract_answer ya filtra, pero por si
            # alguien metió la respuesta cruda nosotros también nos defendemos
            clean = ''.join(c for c in ans.upper() if c in 'ABCDX')
            if not clean: continue
            okays.append((provider, clean, w))
        if not okays:
            return ""

        # 2) Determinar target_len (longitud final del MP4 merged)
        SANITY_CAP = 200  # ningún examen real tiene >200 preguntas
        if expected_hint > 0:
            target_len = min(expected_hint, SANITY_CAP)
        else:
            len_votes: Counter = Counter()
            for _, ans, w in okays:
                len_votes[len(ans)] += w
            # Moda ponderada; en empate, mediana_low entre candidatos empatados
            ranked = len_votes.most_common()
            top_w = ranked[0][1]
            tied = sorted(L for L, w in ranked if w == top_w)
            target_len = min(median_low(tied), SANITY_CAP)
        if target_len <= 0:
            return ""

        # 3) Ajustar respuestas a target_len con detección de outliers
        OUTLIER_RATIO   = 1.5    # >50% más letras que el target = sospechoso
        OUTLIER_PENALTY = 0.5    # los outliers cuentan la mitad
        adjusted = []  # [(answer_padded_to_target, effective_weight, orig_weight)]
        for _, ans, w in okays:
            L = len(ans)
            if L == target_len:
                a, ew = ans, float(w)
            elif L < target_len:
                # Sin pad — simplemente no vota en las posiciones que le faltan.
                # (Padding con 'X' o repetir letras sesgaría posiciones tardías.)
                a, ew = ans, float(w)
            elif L <= int(target_len * OUTLIER_RATIO):
                # Exceso moderado: cortar tomando las ÚLTIMAS target_len letras
                # (en respuestas LLM las "buenas" suelen quedar al final tras
                # el chain-of-thought).
                a, ew = ans[-target_len:], float(w)
            else:
                # Outlier grave: probablemente parloteó. Mismo recorte pero
                # con peso penalizado para que no domine la votación.
                a, ew = ans[-target_len:], float(w) * OUTLIER_PENALTY
            adjusted.append((a, ew, w))

        # 4) Voto posicional ponderado con tie-break por peso individual
        result = []
        for i in range(target_len):
            votes: dict = {}            # letra → suma de pesos
            top_individual: dict = {}   # letra → max peso individual (tie-break)
            for ans, ew, orig_w in adjusted:
                if i >= len(ans):
                    continue
                c = ans[i]
                if c not in 'ABCD':     # 'X' = sin opinión, no cuenta
                    continue
                votes[c] = votes.get(c, 0.0) + ew
                if orig_w > top_individual.get(c, 0):
                    top_individual[c] = orig_w
            if not votes:
                result.append('X')
                continue
            # Sort: 1º total weight desc, 2º max indiv weight desc, 3º letra alfa asc
            best = max(votes.items(),
                       key=lambda kv: (kv[1], top_individual.get(kv[0], 0), -ord(kv[0])))
            result.append(best[0])
        local_result = ''.join(result)

        # 5) META-JUDGE OPCIONAL (árbitro final post-fusión).
        #    Si META_JUDGE_ENABLED, llamamos a un LLM externo que ve:
        #      - El texto OCR fusionado del examen (preguntas + opciones)
        #      - Las respuestas A/B/C/D individuales de cada analyzer
        #    Y produce el veredicto final con razonamiento profundo.
        #
        #    DISEÑO ROBUSTO: si la llamada falla por CUALQUIER motivo
        #    (red, rate limit, JSON malformado, timeout, modelo sin acceso,
        #    longitud incorrecta, contenido inválido), devolvemos `local_result`
        #    de la fusión determinística. Nunca un error del meta-judge tumba
        #    una respuesta — solo puede MEJORARLA.
        # Gate del meta-judge: fusionar() se ejecuta UNA VEZ POR CADA analyzer que
        # reporta (vía _publish_partial). Sin gate, el árbitro (GPT-5 ~60s) se
        # llamaba hasta N veces por job — N llamadas redundantes que multiplican
        # coste y consumo de rate-limit de OpenAI, y que retrasaban 60s la
        # transición a awaiting_review en el PRIMER reporte. Solo arbitramos
        # cuando el ensemble es significativo:
        #   - ya respondió la MAYORÍA de analyzers esperados (fusión estable + da
        #     tiempo a que el árbitro termine ANTES de la auto-aprobación), o
        #   - TODOS los analyzers llegaron a estado terminal (no llega nadie más
        #     → última oportunidad de arbitrar, aunque algunos hayan fallado).
        # Si no se cumple, devolvemos la fusión local determinística (idéntico al
        # fallback de siempre): merged_answer queda válido y se refina al madurar
        # el ensemble. Preserva la fiabilidad (siempre hay respuesta) sin gastar
        # un árbitro por cada IA.
        _n_ok = len(okays)
        _providers_cfg = job.get("_providers") or []
        _n_total = len(_providers_cfg) if _providers_cfg else _n_ok
        _n_terminal = sum(1 for r in responses.values()
                          if isinstance(r, dict)
                          and r.get("status") in ("done", "error", "no_key"))
        _majority = max(2, (_n_total // 2) + 1)
        # GATE TEMPORAL — el meta-judge es una llamada HTTP de ~60s (hasta ~122s
        # con su retry). SOLO debe correr cuando el job YA está en awaiting_review
        # con el timer de auto-aprobado armado. Si corriera en la MISMA llamada a
        # _publish_partial que arma el timer —caso cascada: la única IA OK reporta
        # la última, así que _n_terminal>=_n_total con _n_ok=1— bloquearía la
        # transición a awaiting_review esos ~60-122s y el móvil seguiría viendo
        # "pending" durante su ventana de recogida (~cycleSec+15s) → caería a su
        # fallback aunque el relay esté sano (invariante sync relay↔móvil). El
        # snapshot trae _timer_started=False en la llamada que arma el timer y True
        # en las posteriores, así que el árbitro refina merged_answer DESPUÉS sin
        # retrasar el arranque. Un meta-judge sobre 1 sola respuesta (caso cascada)
        # aporta poco, así que saltarlo ahí no degrada la calidad de la fusión.
        _timer_started = bool(job.get("_timer_started"))
        _run_meta_judge = _timer_started and (
            (_n_ok >= _majority) or (_n_total > 0 and _n_terminal >= _n_total)
        )

        if _run_meta_judge and DYNAMIC_CONFIG.get("META_JUDGE_ENABLED"):
            try:
                exam_text = (job.get("ocr_fused_text") or "").strip()
                if exam_text and okays:
                    # Pasamos las respuestas crudas de cada analyzer (clean) + el peso
                    # para que el árbitro pueda ver quién dijo qué.
                    judge_input = [(p, ans, w) for (p, ans, w) in okays]
                    judge_result = meta_judge_fn(
                        exam_text=exam_text,
                        analyzer_answers=judge_input,
                        expected=target_len,
                    )
                    if judge_result and len(judge_result) == target_len:
                        if judge_result != local_result:
                            n_diff = sum(1 for a, b in zip(local_result, judge_result) if a != b)
                            logger.info(
                                "🧑‍⚖️ meta-judge cambió respuesta: local=%s → judge=%s (%d/%d letras distintas)",
                                local_result, judge_result, n_diff, target_len,
                            )
                        else:
                            logger.info("🧑‍⚖️ meta-judge confirmó fusión local: %s", local_result)
                        return judge_result
                    else:
                        logger.warning(
                            "🧑‍⚖️ meta-judge devolvió longitud incorrecta (%d != %d), usando fusión local",
                            len(judge_result or ""), target_len,
                        )
                else:
                    logger.info("🧑‍⚖️ meta-judge SKIP: sin OCR fusionado o sin analyzers OK")
            except Exception as exc:
                logger.warning("🧑‍⚖️ meta-judge falló, fallback a fusión local: %s", exc)
        return local_result
    except Exception as exc:
        logger.error("fusionar() falló: %s", exc)
        return ""


def meta_judge_fn(
    exam_text: str,
    analyzer_answers: list,
    expected: int,
) -> str:
    """Árbitro final que recibe el examen OCR fusionado + las respuestas A/B/C/D
    de cada analyzer y devuelve la respuesta consensuada.

    DISEÑO: NUNCA debe romper el flujo. Lanza RuntimeError ante cualquier
    problema (HTTP fail, JSON malformado, longitud incorrecta, etc.) y el
    caller (fusionar) hace fallback automático a la fusión local determinística.

    Args:
      exam_text: OCR fusionado del examen (preguntas + opciones).
      analyzer_answers: lista de tuples (provider, clean_answer, weight).
      expected: número de preguntas esperadas (K).

    Returns:
      String con EXACTAMENTE `expected` letras (A/B/C/D/X), o raise.

    Modelo: OPENAI por defecto (gpt-5 con reasoning_effort=high). Razones:
      - GPT-5 con reasoning=high es state-of-art en preguntas de razonamiento
        multi-paso según benchmarks 2026.
      - Reusa OPENAI_API_KEY que ya está configurada.
      - response_format=json_object garantiza output parseable.
    """
    if not exam_text or not exam_text.strip():
        raise RuntimeError("meta-judge: exam_text vacío")
    if expected <= 0:
        raise RuntimeError("meta-judge: expected debe ser > 0")
    if not analyzer_answers:
        raise RuntimeError("meta-judge: sin respuestas de analyzers")

    model = DYNAMIC_CONFIG.get("META_JUDGE_MODEL") or "gpt-5"
    effort = (DYNAMIC_CONFIG.get("META_JUDGE_REASONING_EFFORT") or "high").strip().lower()
    if effort not in ("low", "medium", "high"):
        effort = "high"
    try:
        timeout_s = float(DYNAMIC_CONFIG.get("META_JUDGE_TIMEOUT_S") or 60.0)
    except (TypeError, ValueError):
        timeout_s = 60.0
    timeout_s = max(10.0, min(180.0, timeout_s))

    api_key = DYNAMIC_CONFIG.get("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("meta-judge: OPENAI_API_KEY no configurada")

    # Cap defensivo del exam_text: con 40+ preguntas podría llegar a 20KB+.
    # GPT-5 acepta contexto grande pero pagas tokens — capamos a 30KB.
    if len(exam_text) > 30_000:
        exam_text = exam_text[:30_000] + "\n…[truncado por longitud]…"

    # Construir bloque de respuestas
    lines = []
    for prov, ans, w in analyzer_answers:
        # Recortar respuesta a expected*2 chars max (defensivo vs IAs verbosas)
        ans_short = ans[:max(expected * 2, 50)]
        lines.append(f"  - {prov} (peso={w}): {ans_short}")
    answers_block = "\n".join(lines)

    system = (
        "Eres un árbitro experto que decide la respuesta correcta de un examen "
        "tipo test después de analizar las opiniones de varias IAs analizadoras. "
        f"El examen tiene EXACTAMENTE {expected} preguntas. Tu tarea es devolver "
        f"un string de {expected} letras (A/B/C/D, o X si la pregunta es ilegible "
        "o no hay forma de saberlo).\n\n"
        "Reglas:\n"
        "- Si las IAs coinciden en una pregunta, mantén esa respuesta.\n"
        "- Si discrepan, razona basándote en el TEXTO DEL EXAMEN cuál es la "
        "correcta. Las IAs con peso más alto suelen ser más fiables pero NO "
        "siempre — confía en el razonamiento sobre el contenido.\n"
        "- Usa X solo si la pregunta no se entiende.\n"
        "- DEVUELVE EXACTAMENTE el JSON pedido, sin texto extra."
    )

    user = (
        f"<exam questions=\"{expected}\">\n{exam_text}\n</exam>\n\n"
        f"<analyzers_answers>\n{answers_block}\n</analyzers_answers>\n\n"
        f"Devuelve EXCLUSIVAMENTE este JSON (sin comentarios fuera):\n"
        f'{{"answers": "<{expected} letras A/B/C/D/X>"}}\n'
        f"Ejemplo si {expected}=5: {{\"answers\": \"ABDCX\"}}"
    )

    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type":  "application/json"}
    # Reasoning models (gpt-5*, o1*, o3*, o4*) usan max_completion_tokens y NO
    # aceptan temperature/top_p/seed (devuelven 400 "Unsupported parameter").
    # Modelos clásicos (gpt-4o, gpt-4) usan max_tokens + temperature/top_p/seed.
    model_lower = model.lower()
    is_reasoning = (model_lower.startswith("gpt-5") or model_lower.startswith("o1")
                    or model_lower.startswith("o3") or model_lower.startswith("o4"))
    payload: dict = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        # response_format=json_object → garantía de que content será JSON parseable
        "response_format": {"type": "json_object"},
    }
    if is_reasoning:
        payload["max_completion_tokens"] = 4096
        payload["reasoning_effort"]      = effort
    else:
        payload["max_tokens"]  = 4096
        payload["temperature"] = 0.0
        payload["top_p"]       = 0.1
        payload["seed"]        = 42

    r = _http_post_with_retry(
        "https://api.openai.com/v1/chat/completions",
        payload, headers=headers, retries=1, backoff_s=2.0,
        timeout=timeout_s,
    )
    if not r.is_success:
        raise RuntimeError(f"meta-judge HTTP {r.status_code}: {r.text[:200]}")
    try:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"meta-judge respuesta malformada: {e}")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("meta-judge: content vacío o no-string")

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"meta-judge JSON inválido: {e} · raw={content[:200]}")
    if not isinstance(parsed, dict):
        raise RuntimeError(f"meta-judge JSON no es dict: {type(parsed).__name__}")
    answers_raw = parsed.get("answers")
    if not isinstance(answers_raw, str):
        raise RuntimeError(f"meta-judge 'answers' no es string: {type(answers_raw).__name__}")

    # Sanitizar a A/B/C/D/X
    clean = ''.join(c for c in answers_raw.upper() if c in 'ABCDX')
    if len(clean) != expected:
        raise RuntimeError(
            f"meta-judge devolvió {len(clean)} letras, esperaba {expected} (raw={answers_raw[:80]!r})"
        )
    return clean


def gc_jobs():
    """Elimina jobs expirados, limita el tamaño total y libera imágenes viejas."""
    now = time.time()
    actions = {"stale": 0, "capped": 0, "img_pruned": 0, "timers_cancelled": 0}
    purged_ids: list = []
    with LOCK:
        # 1) TTL: borrar jobs muy viejos. B7: NO purgar jobs activos
        # (pending/awaiting_review) salvo que sean MUY viejos (2× TTL) — si el
        # watchdog cae y un job legítimamente activo cumple TTL=1h, perdemos
        # el trabajo. Damos 2h de margen antes del corte definitivo.
        stale = []
        for k, v in JOBS.items():
            age = now - _safe_float(v.get("created"))
            status = v.get("status")
            if status in ("pending", "awaiting_review"):
                if age > 2 * JOB_TTL:
                    stale.append(k)
            elif age > JOB_TTL:
                stale.append(k)
        for k in stale:
            JOBS.pop(k, None)
            purged_ids.append(k)
        actions["stale"] = len(stale)

        # 2) Cap absoluto: si quedan más de JOBS_MAX, eliminar los más antiguos
        #    DONE/ERROR primero (preservamos los activos)
        if len(JOBS) > JOBS_MAX:
            ordered_terminal = sorted(
                [(jid, j) for jid, j in JOBS.items() if j.get("status") in ("done", "error")],
                key=lambda kv: _safe_float(kv[1].get("created"))
            )
            excess = len(JOBS) - JOBS_MAX
            for jid, _ in ordered_terminal[:excess]:
                JOBS.pop(jid, None)
                purged_ids.append(jid)
                actions["capped"] += 1
            # Si AÚN excede (todos los jobs están activos = imposible normalmente),
            # eliminamos también los más antiguos (cualquier estado)
            if len(JOBS) > JOBS_MAX:
                ordered_all = sorted(JOBS.items(), key=lambda kv: _safe_float(kv[1].get("created")))
                for jid, _ in ordered_all[:len(JOBS) - JOBS_MAX]:
                    JOBS.pop(jid, None)
                    purged_ids.append(jid)
                    actions["capped"] += 1

        # 3) Retención de MEDIA por RECENCIA (cap de RAM DURO). Conservamos los
        #    blobs binarios pesados (img/video/frames/fused/context) SOLO de los
        #    MEDIA_KEEP_RECENT jobs MÁS RECIENTES; el resto los libera YA, sin
        #    esperar ningún TTL. Las "responses"/metadatos de TODOS los jobs se
        #    MANTIENEN (mientras el job exista) para el histórico del panel — solo
        #    se descarta el binario pesado.
        #
        #    Antes esto era por TIEMPO (30min): con ciclos de ~4min se acumulaban
        #    ~8 jobs × (video + 3 frames + fused×2 + contexto) en RAM, y el pico
        #    del stacking en float32 encima → OOM en los 512MB de Render. Por
        #    recencia la media queda acotada a N jobs PASE LO QUE PASE (cadencia
        #    de ciclos o bitrate), y a la vez garantiza "ver siempre los últimos
        #    N vídeos y fotos" desde que se inicia.
        _media_keys = (
            ("img",                         "img_pruned"),
            ("video",                       "video_pruned"),
            ("extracted_frames",            "frames_pruned"),
            ("fused_image_b64",             "fused_image_pruned"),
            ("fused_image_local_b64",       "fused_image_pruned"),
            ("context_image",               "context_image_pruned"),
            ("context_image_provided_b64",  "context_image_pruned"),
        )
        _recent_ids = {
            jid for jid, _ in sorted(
                JOBS.items(),
                key=lambda kv: _safe_float(kv[1].get("created")),
                reverse=True,
            )[:MEDIA_KEEP_RECENT]
        }
        for jid, j in JOBS.items():
            if jid in _recent_ids:
                continue   # los N más recientes conservan su media (los "últimos N")
            for _bk, _flag in _media_keys:
                if j.get(_bk) is not None:
                    j[_bk] = None
                    j[_flag] = True
                    actions.setdefault("media_pruned", 0)
                    actions["media_pruned"] += 1

    # B6: cancelar timers de auto-aprobar de jobs purgados. Sin esto, los
    # threading.Timer activos siguen disparando (auto_aprobar es idempotente:
    # detecta que el job ya no existe y retorna sin hacer nada), pero gastan
    # slot del scheduler y la ref queda colgando hasta que dispare.
    # NameError guard: en el primer GC tras arranque, `_cancel_auto_approve_timer`
    # puede no estar todavía definido si el daemon corrió antes de que el módulo
    # terminase de cargar (race de orden de definición). Skip silencioso ese caso.
    _cancel_fn = globals().get("_cancel_auto_approve_timer")
    if _cancel_fn is not None:
        for jid in purged_ids:
            try:
                if _cancel_fn(jid):
                    actions["timers_cancelled"] += 1
            except Exception:
                pass

    # Purga de CORRECTIONS huérfanas. Antes solo se limpiaban en /result (al
    # entregarlas al móvil), pero si el cliente Android se cae permanentemente
    # las correcciones nunca se entregan y crecen sin tope en RAM + disco.
    # Política: entregadas hace >1h ya las cogió el móvil; no entregadas hace
    # >24h = el cliente no va a volver, las descartamos.
    CORR_MAX_AGE_UNDELIVERED = 24 * 3600
    CORR_MAX_AGE_DELIVERED   = 3600
    corr_purged = 0
    with CORRECTIONS_LOCK:
        before = len(CORRECTIONS)
        CORRECTIONS[:] = [
            c for c in CORRECTIONS
            if (c.get("delivered") and now - c.get("delivered_at", 0) < CORR_MAX_AGE_DELIVERED)
            or (not c.get("delivered") and now - c.get("created", 0) < CORR_MAX_AGE_UNDELIVERED)
        ]
        corr_purged = before - len(CORRECTIONS)
    if corr_purged:
        actions["corr_purged"] = corr_purged
        try:
            _save_corrections_to_file()
        except Exception:
            pass

    if actions["stale"] or actions["capped"]:
        _mark_jobs_dirty()
    if any(actions.values()):
        logger.info("GC: stale=%d capped=%d img_pruned=%d timers=%d corr=%d · total_jobs=%d",
                    actions["stale"], actions["capped"], actions["img_pruned"],
                    actions["timers_cancelled"], actions.get("corr_purged", 0), len(JOBS))


def _resilient_daemon(name: str, period_s: float, fn):
    """Wrapper genérico para daemons: try/except infinito; nunca mueren.
    Reporta su salud en DAEMON_HEALTH para que /health lo muestre."""
    state = DAEMON_HEALTH.setdefault(name, {"last_ok": 0.0, "errors": 0})
    consecutive_errors = 0
    while True:
        try:
            fn()
            state["last_ok"] = time.time()
            consecutive_errors = 0
        except Exception as exc:
            state["errors"] += 1
            consecutive_errors += 1
            # IMPORTANTE: el backoff debe ser ≥ 1s incluso si period_s=0
            # (caso flusher/errlog que son event-driven). Sin esto, un fallo
            # repetido (disco lleno, FS read-only) producía hot-spin a 100% CPU.
            base = max(period_s, 1.0)
            backoff = min(base * (2 ** min(consecutive_errors, 5)), 60.0)
            logger.error("[%s] error #%d: %s — backoff %.0fs",
                         name, state["errors"], exc, backoff)
            traceback.print_exc(file=sys.stdout)
            time.sleep(backoff)
            continue
        time.sleep(period_s)


def _gc_loop():       gc_jobs()


# Rate-limit del flusher (B8): garantiza al menos N segundos entre flushes
# completos. Sin esto, con carga sostenida (3-5 IAs reportando + watchdog +
# auto_aprobar), `_JOBS_DIRTY.set()` se llamaba constantemente y el flusher
# encadenaba flushes back-to-back sin descanso, gastando todo el slot en I/O
# de Supabase/Appwrite mientras los logs gritaban "save_jobs HTTP 429".
_FLUSHER_MIN_INTERVAL_S = 3.0
_flusher_last_flush_t = 0.0


def _flusher_loop():
    """Espera por dirty flag, hace debounce y vuelca a disco + Supabase + Appwrite.
    Disco es write-through inmediato (rápido, sobrevive procesos pero no re-deploys
    en Render free); Supabase es la fuente de verdad entre re-deploys; Appwrite
    es el tercer espejo por si Supabase está caído (rate-limit, downtime, free
    tier suspendido). Cada DB se llama independientemente: que una falle no
    impide que la otra escriba.

    B8: rate-limit con `_FLUSHER_MIN_INTERVAL_S` entre flushes completos. Si el
    `_JOBS_DIRTY` se setea ráfaga, esperamos hasta agotar el intervalo mínimo
    antes de empezar el siguiente save → evita hot-spin bajo carga."""
    global _flusher_last_flush_t
    _JOBS_DIRTY.wait()
    _JOBS_DIRTY.clear()
    # Debounce: si llegan más dirty flags durante el sleep, las agrupamos.
    time.sleep(1.5)
    # B8: garantizar el intervalo mínimo desde el último flush
    elapsed = time.time() - _flusher_last_flush_t
    if elapsed < _FLUSHER_MIN_INTERVAL_S:
        time.sleep(_FLUSHER_MIN_INTERVAL_S - elapsed)
    _JOBS_DIRTY.clear()
    _flusher_last_flush_t = time.time()
    _save_jobs_to_file()
    # Best-effort × 2 DBs: si una falla, el job queda en la otra; al próximo
    # _mark_jobs_dirty se reintenta. No bloquear el flusher por errores de red.
    # C14: lanzar ambos en paralelo (threads) para no encadenar 5s+5s en el
    # peor caso. Esperamos al final con join() acotado para acumular ambos
    # resultados antes de dejar el flusher hacer otro flush.
    def _safe_supa():
        try: _supabase_save_jobs()
        except Exception as exc: logger.error("flusher: _supabase_save_jobs lanzó: %s", exc)

    def _safe_appw():
        try: _appwrite_save_jobs()
        except Exception as exc: logger.error("flusher: _appwrite_save_jobs lanzó: %s", exc)

    t_supa = threading.Thread(target=_safe_supa, daemon=True, name="flush-supa")
    t_appw = threading.Thread(target=_safe_appw, daemon=True, name="flush-appw")
    t_supa.start(); t_appw.start()
    # Join con timeout amplio: timeouts httpx son 10s read + 5s connect, así
    # que 20s es suficiente para que ambos completen o fallen.
    t_supa.join(timeout=20.0)
    t_appw.join(timeout=20.0)


def _watchdog_loop():
    """Detecta jobs colgados (hilos que murieron sin actualizar) y los marca error."""
    now = time.time()
    fixed = 0
    killed_details: list = []
    with LOCK:
        for jid, job in list(JOBS.items()):
            status = job.get("status")
            if status not in ("pending", "awaiting_review"):
                continue
            # _safe_float: si `created` viene corrupto (string/None de un snapshot
            # viejo de Supabase), NO propagamos TypeError — eso abortaba el ciclo
            # ENTERO del watchdog y dejaba sin rescatar a TODOS los jobs colgados.
            age = now - _safe_float(job.get("created"), now)
            # awaiting_review tiene su propio timer (auto_aprobar); solo intervenimos si
            # supera ampliamente su deadline (timer murió por algún motivo).
            if status == "awaiting_review":
                dl = _safe_float(job.get("review_deadline"), 0)
                if dl == 0:
                    # Sin deadline (el timer no se armó, o recargado de un snapshot
                    # viejo sin el campo): NO lo saltamos para siempre — caemos al
                    # respaldo por antigüedad para que el watchdog pueda rescatarlo.
                    if age < STUCK_TIMEOUT:
                        continue
                elif now - dl < 60:   # margen de 60s tras deadline
                    continue
            elif age < STUCK_TIMEOUT:
                continue
            job["status"]      = "error"
            job["error"]       = job.get("error") or f"Watchdog: job colgado >{int(age)}s"
            job["finished"]    = now
            job["watchdog"]    = True
            fixed += 1
            # Snapshot para log fuera del lock — incluye phase y providers
            # respondidos para diagnóstico ("¿qué se quedó colgado?").
            phase_at_kill = job.get("phase", "?")
            responses_status = {
                p: r.get("status", "?") for p, r in (job.get("responses") or {}).items()
                if isinstance(r, dict)
            }
            killed_details.append((jid, int(age), status, phase_at_kill, responses_status))
    if fixed:
        _mark_jobs_dirty()
        # Log AGREGADO (warning) + log POR JOB (error) — el por-job entra al
        # deque _ERROR_LOG y es visible en el panel. Antes solo había el
        # agregado en warning → el panel mostraba ERROR sin logs explicativos.
        logger.warning("🐶 Watchdog: %d jobs colgados marcados como error", fixed)
        for jid, age_s, prev_status, phase, resp_status in killed_details:
            try:
                logger.error(
                    "🐶 Watchdog kill: job=%s · prev_status=%s · age=%ds · "
                    "phase=%s · responses=%s",
                    jid[:8], prev_status, age_s, phase, resp_status,
                )
            except Exception:
                pass


# B20: track de la última escritura LOCAL de config a las DBs. Sin esto, el
# sync_loop puede pisar un cambio que el operador hizo hace 5s con la versión
# de hace 50s (la otra instancia). Si acabamos de escribir, skip-eamos el sync
# durante una ventana corta — preferimos consistencia eventual con favor a
# nuestra última escritura sobre last-writer-wins ciego.
_LAST_LOCAL_CONFIG_WRITE = 0.0
_SYNC_SKIP_AFTER_LOCAL_WRITE_S = 15.0


def _mark_local_config_write():
    """Llama esto justo después de un write exitoso a las DBs desde el panel
    o /api/config/import. El sync_loop respeta esta marca por N segundos."""
    global _LAST_LOCAL_CONFIG_WRITE
    _LAST_LOCAL_CONFIG_WRITE = time.time()


# Sync con Supabase cada 60 s. Si la otra instancia (Render↔Railway) cambia
# algo, lo recogemos sin reinicio. Frecuencia conservadora para no agotar el
# tier gratuito de Supabase. Cambios locales se publican en el momento por
# _save_config_to_file(). Si Supabase falla, se intenta Appwrite como fallback.
def _supabase_sync_loop():
    # B20: si acabamos de escribir local, no leamos: la otra instancia tiene
    # versión vieja (todavía no syncó con Supabase) y nos sobreescribiría con
    # algo que ya no es válido aquí. Damos margen para que Supabase asiente.
    if time.time() - _LAST_LOCAL_CONFIG_WRITE < _SYNC_SKIP_AFTER_LOCAL_WRITE_S:
        return
    # Si nuestra última escritura no llegó a Supabase pero sí a Appwrite,
    # Supabase tiene datos antiguos. Saltarlo evita sobreescribir cambios.
    if _SUPABASE_STALE and _appwrite_enabled():
        m = _appwrite_load_config()
        if m > 0:
            logger.debug("🔷 Appwrite sync (Supabase marked stale): %d claves refrescadas", m)
        return
    # Snapshot del timestamp antes de la llamada → permite distinguir si la HTTP
    # call tuvo éxito (LAST_OK actualizado) vs si falló (LAST_OK sin cambios).
    last_ok_before = _SUPABASE_LAST_OK
    n = _supabase_load_config()
    if n > 0:
        logger.debug("☁️ Supabase sync: %d claves refrescadas", n)
        return
    if _appwrite_enabled():
        m = _appwrite_load_config()
        if m > 0:
            # Distinguir el motivo: Supabase respondió 200 vacío vs. no respondió.
            supabase_responded = _SUPABASE_LAST_OK > last_ok_before
            if supabase_responded:
                logger.debug("🔷 Appwrite sync (Supabase vacío, sin datos aún): %d claves", m)
            else:
                logger.info("🔷 Appwrite sync fallback: %d claves refrescadas (Supabase no respondió)", m)


# Arrancar los daemons (resilientes — nunca mueren)
threading.Thread(target=_resilient_daemon, args=("gc",       60.0, _gc_loop),
                 daemon=True, name="gc-daemon").start()
threading.Thread(target=_resilient_daemon, args=("flusher",  0.0,  _flusher_loop),
                 daemon=True, name="flusher-daemon").start()
threading.Thread(target=_resilient_daemon, args=("watchdog", float(WATCHDOG_PERIOD), _watchdog_loop),
                 daemon=True, name="watchdog-daemon").start()
threading.Thread(target=_resilient_daemon, args=("errlog",   0.0,  _errlog_flusher_loop),
                 daemon=True, name="errlog-flusher").start()
threading.Thread(target=_resilient_daemon, args=("supabase_sync", 60.0, _supabase_sync_loop),
                 daemon=True, name="supabase-sync-daemon").start()


# ─── Auto-aprobación ─────────────────────────────────────────────────────────
# Refs en memoria a los threading.Timer activos. Sin estas refs, Python podría
# recolectar el Timer object antes de su disparo, dejando el job huérfano.
# A4: protegido por _AUTO_APPROVE_LOCK propio (no usamos LOCK general porque
# en este dict no hay re-entrancia con el job dict). Reduce contención.
_AUTO_APPROVE_TIMERS: dict = {}
_AUTO_APPROVE_LOCK = threading.Lock()


def _cancel_auto_approve_timer(job_id: str) -> bool:
    """Cancela y elimina el timer de auto-aprobar para `job_id`. Idempotente:
    si no había timer registrado, retorna False sin error. Llamado desde
    auto_aprobar (fired), gc_jobs (job purgado), /reset, /correction."""
    with _AUTO_APPROVE_LOCK:
        t = _AUTO_APPROVE_TIMERS.pop(job_id, None)
    if t is not None:
        try: t.cancel()
        except Exception: pass
        return True
    return False


def auto_aprobar(job_id: str):
    """Auto-aprueba un job tras review_timeout_seconds. Corre en threading.Timer
    aparte (no en _resilient_daemon), así que envolvemos TODO en try/except —
    si lanza, el job se queda colgado en awaiting_review hasta que el watchdog
    lo rescate.

    También marca _cancelled=True para que los hilos de IAs lentas (deepseek
    con thinking, claude con extended thinking) terminen antes y liberen
    conexiones HTTP en cuanto llegue su próximo checkpoint."""
    try:
        # Liberar la ref del Timer (ya disparó). Bajo el lock propio del dict
        # para evitar la race A4 (dos hilos creando timers simultáneos pisándose).
        with _AUTO_APPROVE_LOCK:
            _AUTO_APPROVE_TIMERS.pop(job_id, None)
        # A3: transición Y publicación de fase dentro de la misma sección crítica.
        # Antes el `_set_job_phase` se llamaba fuera del LOCK general (esa func
        # adquiere LOCK de nuevo), lo que abría una ventana donde otro hilo veía
        # status=done pero phase distinta a "done". Ahora hacemos las dos cosas
        # bajo el mismo LOCK y publicamos al panel después.
        prev_phase = None
        transitioned = False
        with LOCK:
            job = JOBS.get(job_id)
            if not job or job.get("status") != "awaiting_review":
                return
            job["status"]        = "done"
            job["auto_approved"] = True
            job["_cancelled"]    = True   # señal a IAs en vuelo para abortar
            prev_phase = job.get("phase")
            job["phase"] = "done"
            hist = job.setdefault("phase_history", [])
            hist.append({"phase": "done", "t": time.time(), "auto_approved": True})
            if len(hist) > _PHASE_HISTORY_MAX:
                del hist[:-_PHASE_HISTORY_MAX]
            transitioned = True
        if transitioned:
            _mark_jobs_dirty()
            logger.info("⏱ job=%s phase: %s → done (auto-aprobado)", job_id[:8], prev_phase)
    except Exception as exc:
        logger.error("💥 auto_aprobar[%s] falló: %s", job_id[:8], exc)
        traceback.print_exc(file=sys.stdout)


# ─── Publicación incremental ──────────────────────────────────────────────────
def _publish_partial(job_id: str, provider: str, entry: dict, expected: int):
    """Actualiza el resultado de UNA IA y recalcula la fusión. Thread-safe.
    NUNCA debe lanzar excepción aunque algo dentro falle.

    DISEÑO (post-auditoría de fiabilidad):
      1. Sección crítica corta dentro de LOCK: escribir entry, copiar job snapshot.
      2. Cómputo de fusionar() FUERA de LOCK (era hasta ~50ms con prompts grandes,
         bloqueaba el polling del panel mientras 5 IAs reportaban a la vez).
      3. Re-acquire LOCK para escribir el merged + decidir transición.
      4. threading.Timer.start() FUERA de LOCK (creación de thread puede tardar
         milisegundos bajo presión de OS y NO debe bloquear otros hilos del job).
    """
    try:
        # ── (1) Sección crítica CORTA: escribir entry + capturar snapshot ──
        _publish_late_only = False
        with LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            responses = job.setdefault("responses", {})
            # Si el job ya es terminal (revisor envió, watchdog marcó error, otras
            # IAs ya transicionaron), persistimos la respuesta tardía con flag
            # `late=True` para que el panel histórico vea el status final real
            # (done/error) en lugar de dejar el provider colgado en "processing"
            # forever (estado que se asignó en call_ai_task antes del HTTP call).
            # NO recalculamos merge ni transición — la decisión ya está tomada.
            if job.get("status") in ("done", "error"):
                late_entry = dict(entry)
                late_entry["late"] = True
                responses[provider] = late_entry
                _publish_late_only = True
            else:
                responses[provider] = entry
                # Snapshot suficiente para fusionar() (que solo lee responses + algunos
                # metadatos). Copia superficial: las dict-values son los mismos dicts
                # de entry, pero como las re-write atómicamente en _publish_partial
                # no hay race de lectura concurrente sobre los mismos campos.
                job_snapshot = {
                    "responses":          dict(responses),
                    "expected_questions": job.get("expected_questions"),
                    "_providers":         list(job.get("_providers") or []),
                    # Gate temporal del meta-judge (ver fusionar): si el timer de
                    # auto-aprobado aún NO está armado, NO corremos el árbitro HTTP
                    # — correría en esta misma llamada y retrasaría la transición a
                    # awaiting_review, rompiendo la ventana del móvil.
                    "_timer_started":     bool(job.get("_timer_started")),
                }
                already_touched = bool(job.get("_touched"))
        if _publish_late_only:
            _mark_jobs_dirty()
            return

        # ── (2) Cómputo de fusión SIN lock: pure CPU sobre el snapshot ──
        try:
            merged = fusionar(job_snapshot)
            if expected > 0 and merged:
                if len(merged) < expected:
                    merged += 'X' * (expected - len(merged))
                elif len(merged) > expected:
                    merged = merged[:expected]
        except Exception as exc:
            logger.error("Fallo calculando merged en job %s: %s", job_id[:8], exc)
            merged = ""

        # ── (3) Re-acquire LOCK para escribir el resultado + decidir transición ──
        timer_to_start = None    # (fire_in, callback, args)  — disparado fuera del lock
        phase_transition: Optional[str] = None
        log_after_lock = None    # tupla (level, fmt, args) si hay que loguear fuera del lock
        with LOCK:
            job = JOBS.get(job_id)
            if not job or job.get("status") in ("done", "error"):
                return
            # A5: leer `responses` AHORA, no usar la captura del primer lock.
            # Entre los dos locks otro hilo puede haber añadido/cambiado entries
            # → la decisión "todas fallaron" usaría datos stale. Re-leer aquí.
            responses = job.get("responses") or {}
            if not merged:
                merged = job.get("merged_answer", "")
            # B13: si el job ya está "touched" (revisor editó), NO sobreescribimos
            # merged_answer — el operador ya validó una versión y un retry tardío
            # de IA no debe reescribir el histórico bajo sus pies.
            if not already_touched and not job.get("_touched"):
                job["merged_answer"] = merged
                job["answer"]        = merged
            elif not job.get("merged_answer"):
                # Si nunca se escribió antes (caso raro), aceptamos el primer valor
                # útil aunque _touched=True (no había nada que pisar).
                job["merged_answer"] = merged

            if entry.get("ok") and not job.get("_timer_started"):
                job["_timer_started"] = True
                job["status"]         = "awaiting_review"
                tout = job.get("review_timeout_seconds", 90)
                # FIX SYNC CLIENTE-RELAY (2026-05): Antes el deadline se medía desde
                # cuando llegaba la 1ª IA → en video, con fase OCR de 30-60s, el
                # auto-aprobado caía DESPUÉS de cycleSec, así que el móvil polleaba
                # 15s y se rendía antes de tener respuesta.
                #
                # Ahora el deadline se ancla al MIN entre:
                #   (a) created + tout         → contrato absoluto con el cliente.
                #   (b) now + MIN_REVIEW_WIN   → mínimo de revisión humana si la 1ª
                #                                IA llegó tan tarde que ya estamos
                #                                pasados de (a).
                # Resultado: si la 1ª IA llega temprano, deadline = (a) (cliente
                # encuentra "done" cuando despierta). Si llega tarde, deadline =
                # 1stAI + MIN_REVIEW_WIN (al menos da unos segundos al admin).
                # Ventana mínima de revisión humana GARANTIZADA desde la 1ª IA,
                # configurable por env MIN_REVIEW_WINDOW_S (default 90s). Antes eran
                # 5s fijos: con vídeo el OCR tarda ~90-100s, así que `created+tout`
                # ya había vencido al llegar la 1ª IA y al revisor solo le quedaban
                # 5s para corregir las cards. Ahora siempre tiene ≥ esta ventana.
                try:
                    MIN_REVIEW_WIN = float(os.environ.get("MIN_REVIEW_WINDOW_S", "90") or 90)
                except (ValueError, TypeError):
                    MIN_REVIEW_WIN = 90.0
                MIN_REVIEW_WIN = max(5.0, min(300.0, MIN_REVIEW_WIN))
                created = job.get("created", time.time())
                now = time.time()
                abs_deadline = created + tout
                min_deadline_after_first = now + MIN_REVIEW_WIN
                deadline = max(abs_deadline, min_deadline_after_first)
                job["review_deadline"] = deadline
                # El timer se arma con el delta desde ahora — no con tout entero.
                # Si abs_deadline ya pasó (1ª IA tardona) el timer fire en ~5s.
                fire_in = max(0.5, deadline - now)
                timer_to_start = (fire_in, job_id)   # arrancar FUERA del lock
                log_after_lock = (
                    "info",
                    "Job %s → awaiting_review (1ª IA: %s). Auto-aprob en %.1fs "
                    "(deadline=created+%ds=%.1fs abs, 1stAI+%.0fs=%.1fs min → fire=%.1f)",
                    (job_id[:8], provider, fire_in, tout, abs_deadline - created,
                     MIN_REVIEW_WIN, min_deadline_after_first - created, fire_in),
                )
                phase_transition = "awaiting_review"
                # A3: aplicar fase INLINE bajo el mismo LOCK que el status.
                _set_job_phase_locked(job, phase_transition)
            elif not job.get("_timer_started"):
                # No hubo ok=True. Si TODAS las IAs ya reportaron resultado FINAL
                # y ninguna sirvió, marcar el job como error inmediatamente para no
                # esperar al watchdog (STUCK_TIMEOUT=180s).
                # IMPORTANTE: contar SOLO providers que llegaron a un estado terminal
                # (done/error/no_key). NO contar las que aún están "waiting"/"pending"/
                # "processing" — sino una IA rápida que falle (ej. Gemini 503) marcaría
                # el job en error antes de dar tiempo a Claude/GPT/etc. a responder.
                # status="no_key" no cuenta como "fallo" — simplemente no participa.
                _terminal = ("done", "error", "no_key")
                total_providers = len(job.get("_providers") or _PROVIDERS)
                reported = sum(1 for r in responses.values()
                               if r.get("status") in _terminal)
                ok_count       = sum(1 for r in responses.values() if r.get("ok"))
                no_key_count   = sum(1 for r in responses.values() if r.get("status") == "no_key")
                error_count    = sum(1 for r in responses.values()
                                     if r.get("status") == "error")
                # Reporte completo cuando todas las providers han llegado a terminal
                if reported >= total_providers and ok_count == 0:
                    job["status"] = "error"
                    job["finished"] = time.time()
                    if no_key_count == total_providers:
                        job["error"] = ("Ninguna IA tiene API key configurada — "
                                        "ve al panel /panel y rellena al menos una key principal")
                        logger.error("Job %s → error (sin keys configuradas en NINGÚN proveedor)",
                                     job_id[:8])
                    else:
                        errs = [r.get("error","?")[:80] for r in responses.values()
                                if r.get("status") == "error"]
                        msg = "Todas las IAs con key fallaron · " + " | ".join(errs)
                        if no_key_count > 0:
                            msg += f" · ({no_key_count} sin key)"
                        job["error"] = msg
                        logger.error("Job %s → error (errores=%d, sin_key=%d)",
                                     job_id[:8], error_count, no_key_count)
                    phase_transition = "error_analysis"
                    # A3: aplicar fase INLINE bajo el mismo LOCK
                    _set_job_phase_locked(job, phase_transition)

        # ── (4) Arrancar Timer FUERA del lock + guardar ref para evitar GC ──
        # threading.Timer.start() crea un thread, lo cual puede tardar bajo presión
        # de OS. Hacerlo fuera del LOCK evita bloquear otros hilos del job que
        # estén esperando. La ref se guarda en _AUTO_APPROVE_TIMERS para que el
        # GC de Python no recolecte el Timer mid-flight (sería catastrófico:
        # auto_aprobar nunca se llamaría y el job se quedaría colgado hasta el
        # watchdog en STUCK_TIMEOUT).
        if timer_to_start is not None:
            try:
                fire_in, jid = timer_to_start
                t = threading.Timer(fire_in, auto_aprobar, args=[jid])
                t.daemon = True
                # A4: la escritura va bajo _AUTO_APPROVE_LOCK. Además, si ya
                # había un timer (race con otro publish_partial reportando casi
                # simultáneamente), lo cancelamos antes de pisarlo — evita dos
                # auto_aprobar disparándose.
                with _AUTO_APPROVE_LOCK:
                    old = _AUTO_APPROVE_TIMERS.get(jid)
                    if old is not None:
                        try: old.cancel()
                        except Exception: pass
                    _AUTO_APPROVE_TIMERS[jid] = t
                t.start()
            except Exception as exc:
                logger.error("No se pudo arrancar timer auto-aprobar %s: %s",
                             job_id[:8], exc)
                # Fallback: el watchdog cogerá el job en STUCK_TIMEOUT si nadie lo
                # aprueba antes — no quedamos sin red de seguridad.
        if log_after_lock:
            lvl, fmt, args = log_after_lock
            getattr(logger, lvl)(fmt, *args)

        _mark_jobs_dirty()
        # A3: la fase ya se actualizó INLINE dentro del LOCK (con
        # `_set_job_phase_locked`). Aquí solo logueamos para tracing.
        if phase_transition:
            logger.info("⏱ job=%s phase: → %s", job_id[:8], phase_transition)
    except Exception as exc:
        logger.error("💥 _publish_partial[%s/%s]: %s", job_id[:8], provider, exc)
        traceback.print_exc(file=sys.stdout)


# ─── Llamadas a IA (cada una en su hilo) ─────────────────────────────────────
# Timeout global por defecto. Algunos providers son consistentemente más lentos
# (DeepSeek con thinking, Claude con web_search) → tienen override en _PROVIDER_TIMEOUTS.
AI_HARD_TIMEOUT = float(os.environ.get("AI_HARD_TIMEOUT", "360"))  # 6 min tope absoluto

# Timeout UNIFICADO para todos los analyzers (360s). Decisión de diseño:
#   - Las IAs rápidas (gpt, gemini, mistral, mimo) terminan cuando terminan;
#     este 360s es solo el TECHO si se cuelgan, NO un mínimo a esperar.
#   - Las IAs lentas legítimamente (deepseek-R1 con thinking, claude con
#     extended thinking 8k + web_search) tienen aire de sobra (rara vez >300s).
#   - Coordinado: el job termina cuando todas hayan respondido o se cuelguen
#     simultáneamente, no a tiempos dispares que daban sensación caótica.
#   - El techo absoluto del job lo pone STUCK_TIMEOUT (watchdog global).
# Si una IA concreta necesita override (caso raro), via env var:
#   AI_TIMEOUT_GPT=180 AI_TIMEOUT_CLAUDE=240 ...
_PROVIDER_TIMEOUTS: dict = {
    "deepseek": float(os.environ.get("AI_TIMEOUT_DEEPSEEK", "360")),
    "claude":   float(os.environ.get("AI_TIMEOUT_CLAUDE",   "360")),
    "gpt":      float(os.environ.get("AI_TIMEOUT_GPT",      "360")),
    "gemini":   float(os.environ.get("AI_TIMEOUT_GEMINI",   "360")),
    "mistral":  float(os.environ.get("AI_TIMEOUT_MISTRAL",  "360")),
    "mimo":     float(os.environ.get("AI_TIMEOUT_MIMO",     "360")),
}


def _call_ai_with_timeout(fn, req: AskRequest, timeout_s: float,
                           cancel_check=None) -> dict:
    """Ejecuta fn(req) en un hilo aparte y lo cancela si supera timeout_s.
    Garantiza que un proveedor colgado NO bloquea al worker indefinidamente.

    B5: si `cancel_check` es callable y devuelve True, abortamos antes del
    timeout. El thread sigue corriendo en background (Python no permite kill)
    pero al menos liberamos al caller para que pueda terminar el job sin
    esperar al provider. El thread morirá cuando httpx termine su request o
    cuando el daemon termine en exit."""
    box: dict = {}
    def _runner():
        try: box["res"] = fn(req)
        except Exception as e: box["err"] = e
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    if cancel_check is None:
        t.join(timeout_s)
    else:
        # Poll cada 2s en lugar de un solo join largo, para detectar cancelación
        # rápido sin sumar latencia a operaciones normales.
        deadline = time.monotonic() + timeout_s
        poll_interval = 2.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            t.join(timeout=min(poll_interval, remaining))
            if not t.is_alive():
                break
            try:
                if cancel_check():
                    raise RuntimeError("cancelled (job ya en done/error)")
            except RuntimeError:
                raise
            except Exception:
                # Si el check lanza, ignoramos y seguimos esperando.
                pass
    if t.is_alive():
        raise TimeoutError(f"AI worker no respondió en {timeout_s:.0f}s")
    if "err" in box:
        raise box["err"]
    return box["res"]


def call_ai_task(job_id: str, provider: str, fn, req: AskRequest):
    # monotonic: inmune a ajustes del reloj del sistema (NTP/DST). time.time()
    # daría latencias negativas si el reloj retrocede durante el job.
    t0 = time.monotonic()
    expected = req.expected_questions or 0
    try:
        with LOCK:
            job = JOBS.get(job_id)
            if not job:
                logger.info("Job %s ya no existe → abortando %s", job_id[:8], provider)
                return
            # Si el job ya está terminal o cancelado (auto-aprobado, /result enviado,
            # watchdog), no arrancamos la llamada HTTP — ahorramos conexión + tiempo.
            if job.get("status") in ("done", "error") or job.get("_cancelled"):
                logger.info("⏭ %s [%s] cancelado antes de arrancar (status=%s)",
                            provider, job_id[:8], job.get("status"))
                return
            if provider in job.get("responses", {}):
                job["responses"][provider]["status"] = "processing"

        # Timeout específico por provider (DeepSeek/Claude más lentos por thinking)
        timeout_s = _PROVIDER_TIMEOUTS.get(provider, AI_HARD_TIMEOUT)
        # B5: cancel_check para abortar si el job fue auto-aprobado o cerrado
        # por el cómplice mientras esta IA está en vuelo. No mata el thread
        # (Python no permite), pero libera al worker para terminar el job.
        def _cancel_check():
            with LOCK:
                jj = JOBS.get(job_id)
                if not jj:
                    return True
                return bool(jj.get("_cancelled")) or jj.get("status") in ("done", "error")
        result = _call_ai_with_timeout(fn, req, timeout_s, cancel_check=_cancel_check)
        raw    = result.get("raw", "") or ""
        try:
            answer = align_answer(raw, expected) if expected > 0 else result.get("answer", "")
        except Exception as exc:
            logger.error("align_answer falló: %s → usando answer original", exc)
            answer = result.get("answer", "")
        ms = int((time.monotonic() - t0) * 1000)
        # Cap defensivo del raw EN MEMORIA: si una IA aluciona o entra en loop
        # de generación, su raw puede ser MB+. JOBS_MAX=150 × 6 analyzers × 50MB
        # = OOM catastrófico. 100KB cubre cualquier razonamiento legítimo
        # (un raw normal son 5-20 KB) y deja diagnóstico completo.
        _RAW_MAX_CHARS = 100_000
        if isinstance(raw, str) and len(raw) > _RAW_MAX_CHARS:
            raw_truncated_from = len(raw)
            raw = raw[:_RAW_MAX_CHARS]
        else:
            raw_truncated_from = 0
        entry = {
            "status": "done", "ok": True,
            "answer": answer, "raw": raw,
            "provider": provider, "model": result.get("model", ""), "ms": ms,
        }
        if raw_truncated_from:
            entry["raw_truncated_from"] = raw_truncated_from
        _publish_partial(job_id, provider, entry, expected)
    except NoApiKeyError as e:
        # IA sin key configurada → NO es un fallo del proveedor; simplemente
        # no participa. Marcamos status="no_key" para que el dashboard lo
        # diferencie y la lógica de "todas fallaron" no la cuente como caída.
        ms = int((time.monotonic() - t0) * 1000)
        logger.info("⏭ %s [%s] sin key configurada → no participa", provider, job_id[:8])
        try:
            _publish_partial(job_id, provider, {
                "status": "no_key", "ok": False,
                "error": "Sin API key configurada", "provider": provider, "ms": ms,
            }, expected)
        except Exception as exc2:
            logger.error("💥 Fallo publicando no_key de %s: %s", provider, exc2)
    except Exception as e:
        ms = int((time.monotonic() - t0) * 1000)
        err_str = f"{type(e).__name__}: {e}"
        logger.error("Error %s [%s] en %dms: %s", provider, job_id[:8], ms, err_str)
        try:
            _publish_partial(job_id, provider, {
                "status": "error", "ok": False,
                "error": err_str[:200], "provider": provider, "ms": ms,
            }, expected)
        except Exception as exc2:
            logger.error("💥 Doble fallo publicando error de %s: %s", provider, exc2)


# ─── Funciones de IA ──────────────────────────────────────────────────────────
def _anthropic_image_block(b64: str, mime: str) -> dict:
    """Construye el content-block 'image' para la API de Anthropic Messages."""
    return {"type": "image", "source": {"type": "base64",
                                          "media_type": mime, "data": b64}}


def _gpt_responses_image_block(b64: str, mime: str) -> dict:
    """Bloque 'input_image' para OpenAI Responses API (data URL inline)."""
    return {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"}


def _gpt_chat_image_block(b64: str, mime: str) -> dict:
    """Bloque 'image_url' para OpenAI chat/completions (fallback legacy)."""
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _gemini_image_part(b64: str, mime: str) -> dict:
    """Part 'inline_data' para Gemini generateContent."""
    return {"inline_data": {"mime_type": mime, "data": b64}}


def gpt_fn(req: AskRequest) -> dict:
    """GPT-5.5 con la **Responses API agéntica** completa (modo investigador).

    Features activas (configurables vía panel · DYNAMIC_CONFIG):
      • Tool `web_search` GA (con fallback automático a `web_search_preview` si
        la cuenta no tiene acceso al GA todavía). Loop agéntico: el modelo decide
        cuándo abrir páginas, buscar dentro, y re-buscar si la primera fuente no
        responde la pregunta.
      • `reasoning.effort` = low/medium/high (medium por defecto). Sube a high
        para Deep Research; baja a low para velocidad si confías en el OCR.
      • `filters.allowed_domains` (máx 100): si configuras "boe.es",
        "tribunalconstitucional.es", etc., la búsqueda se ciñe a esos dominios.
        Vacío = búsqueda libre en internet.
      • `max_tool_calls`: tope al loop agéntico (default 4) para no exceder 2 min.
      • Timeout HTTP estricto de 120s (consigna del usuario: "que no se esté más
        de 2 min"). El hilo entero también está capado a 120s vía
        _PROVIDER_TIMEOUTS["gpt"]=120.
      • Fallback automático a /chat/completions si /responses no soporta el modelo.
    """
    model = DYNAMIC_CONFIG.get("OPENAI_MODEL") or OPENAI_MODEL
    effort = (DYNAMIC_CONFIG.get("OPENAI_REASONING_EFFORT")
              or OPENAI_REASONING_EFFORT).strip().lower()
    if effort not in ("low", "medium", "high"):
        effort = "medium"
    try:
        max_tool_calls = int(DYNAMIC_CONFIG.get("OPENAI_MAX_TOOL_CALLS") or OPENAI_MAX_TOOL_CALLS)
    except (TypeError, ValueError):
        max_tool_calls = OPENAI_MAX_TOOL_CALLS
    max_tool_calls = max(1, min(20, max_tool_calls))
    allowed_domains = DYNAMIC_CONFIG.get("OPENAI_ALLOWED_DOMAINS") or []
    # Normalizar a list[str] limpia (por si vino como CSV string desde DB)
    if isinstance(allowed_domains, str):
        allowed_domains = [d.strip() for d in allowed_domains.replace("\n", ",").split(",") if d.strip()]
    allowed_domains = [d for d in allowed_domains if isinstance(d, str) and d.strip()][:100]
    # max_output_tokens depende del effort: con high necesita más margen para
    # cadena de pensamiento + tool results + respuesta final.
    max_out = {"low": 4096, "medium": 8000, "high": 12000}[effort]

    # Construir la tool web_search con filters si hay dominios autorizados.
    web_tool: dict = {"type": "web_search"}
    if allowed_domains:
        web_tool["filters"] = {"allowed_domains": allowed_domains}

    def _call(api_key: str) -> dict:
        headers = {"Authorization": f"Bearer {api_key}"}
        user_content: list = []
        # Orden: IMAGEN 1 (diagrama de contexto, si existe) → IMAGEN 2 (la "buena").
        # Coincide con el prompt SYSTEM_PROMPT_CON_CONTEXTO del móvil.
        if req.context_image_b64:
            user_content.append(_gpt_responses_image_block(
                req.context_image_b64, req.context_image_mime))
        if req.image_b64:
            user_content.append(_gpt_responses_image_block(
                req.image_b64, req.image_mime))
        user_content.append({"type": "input_text", "text": req.prompt})
        payload = {
            "model":      model,
            "input":      [
                {"role": "system", "content": req.system},
                {"role": "user",   "content": user_content},
            ],
            "reasoning":  {"effort": effort},
            "tools":      [web_tool],
            "max_tool_calls":    max_tool_calls,
            "max_output_tokens": max_out,
            # NO mandamos `temperature` en /responses: gpt-5.x con reasoning la
            # RECHAZA con 400 ("Unsupported parameter: temperature"). El
            # determinismo lo aporta el propio reasoning + effort.
        }
        # Llamada con TIMEOUT ESTRICTO 120s — consigna usuario.
        r = _http_post_with_retry("https://api.openai.com/v1/responses",
                                   payload, headers=headers, retries=2, backoff_s=2.0,
                                   timeout=OPENAI_WEB_SEARCH_HTTP_TIMEOUT_S)
        # Fallback A: la cuenta no tiene `web_search` GA todavía → reintento con
        # `web_search_preview` (lo que funcionaba antes).
        if not r.is_success and r.status_code in (400, 404, 422):
            body_lc = (r.text or "").lower()
            if "web_search" in body_lc and ("preview" in body_lc or "not available" in body_lc
                                            or "unsupported" in body_lc or "invalid" in body_lc):
                logger.info("GPT /responses no acepta web_search GA → fallback a preview")
                payload["tools"] = [{"type": "web_search_preview"}]
                r = _http_post_with_retry("https://api.openai.com/v1/responses",
                                           payload, headers=headers, retries=1, backoff_s=2.0,
                                           timeout=OPENAI_WEB_SEARCH_HTTP_TIMEOUT_S)
        # Fallback B: /responses no soporta el modelo o la cuenta no tiene acceso.
        # Caemos a /chat/completions (sin agentic search; respuesta básica).
        if not r.is_success:
            # Reasoning models (gpt-5*, o1*, o3*, o4*) usan max_completion_tokens
            # en /chat/completions y NO aceptan temperature/top_p/seed/penalties
            # (devuelven 400 "Unsupported parameter"). Modelos clásicos (gpt-4o,
            # gpt-4, gpt-3.5) siguen con max_tokens y aceptan los demás.
            model_lc = model.lower()
            is_reasoning = (model_lc.startswith("gpt-5") or model_lc.startswith("o1")
                            or model_lc.startswith("o3") or model_lc.startswith("o4"))
            user_content_chat: list = []
            if req.context_image_b64:
                user_content_chat.append(_gpt_chat_image_block(
                    req.context_image_b64, req.context_image_mime))
            if req.image_b64:
                user_content_chat.append(_gpt_chat_image_block(
                    req.image_b64, req.image_mime))
            user_content_chat.append({"type": "text", "text": req.prompt})
            payload_legacy: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": req.system},
                    {"role": "user",   "content": user_content_chat},
                ],
            }
            if is_reasoning:
                payload_legacy["max_completion_tokens"] = 4096
                # reasoning_effort opcional (lo respeta /chat/completions para gpt-5/o*)
                payload_legacy["reasoning_effort"] = effort
            else:
                payload_legacy["max_tokens"]  = 4096
                payload_legacy["temperature"] = 0.0
                payload_legacy["top_p"]       = 0.1
                payload_legacy["seed"]        = 42
            r = _http_post_with_retry("https://api.openai.com/v1/chat/completions",
                                       payload_legacy, headers=headers, retries=2, backoff_s=2.0,
                                       timeout=OPENAI_WEB_SEARCH_HTTP_TIMEOUT_S)
            if not r.is_success:
                raise RuntimeError(f"GPT {r.status_code}: {r.text[:200]}")
            try:
                txt = r.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError) as e:
                raise RuntimeError(f"GPT chat respuesta malformada: {e}")
            return {"answer": extract_answer(txt), "raw": txt, "model": model}
        # /responses: extraer texto del output (puede venir como output_text
        # directo o como output[*].content[*] con type=output_text).
        try:
            data = r.json()
            txt = data.get("output_text", "")
            if not txt:
                for item in data.get("output", []):
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            txt += c.get("text", "")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"GPT respuesta malformada: {e}")
        return {"answer": extract_answer(txt), "raw": txt, "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["OPENAI_API_KEY"],
                            DYNAMIC_CONFIG["OPENAI_API_KEY_BACKUP"], _call)


def claude_fn(req: AskRequest) -> dict:
    """Claude Opus 4.7 con web search nativo + adaptive thinking.

    Cambios vs Opus 4.6:
      - `thinking={"type": "enabled", "budget_tokens": N}` está DEPRECADO en 4.7
        y devuelve 400 ("extended thinking removed"). Sustituido por
        `thinking={"type": "adaptive"}` + `output_config={"effort": "high"}`
        (el modelo decide cuánto pensar según la pregunta).
      - `temperature` / `top_p` / `top_k` están DEPRECADOS en 4.7 — cualquier
        valor non-default devuelve 400 ("temperature is deprecated for this
        model"). Hay que OMITIRLOS del todo en el call.
      - web_search_20250305 (beta) sigue funcionando con dynamic pre-filtering.
      - max_tokens 16000: margen para thinking + tool results + respuesta final.
    """
    import anthropic
    model = DYNAMIC_CONFIG.get("CLAUDE_MODEL") or CLAUDE_MODEL
    # Detección por modelo: Opus 4.7+ rechaza temperature/extended-thinking.
    # Para modelos viejos (Opus 4.6, Sonnet 4.5, Haiku 3.5) seguimos usando el
    # esquema antiguo (extended thinking + temperature en fallback).
    model_lc = model.lower()
    is_new_thinking = (
        "opus-4-7"  in model_lc or "opus-4-8"  in model_lc or "opus-5"    in model_lc
        or "sonnet-4-7" in model_lc or "sonnet-5" in model_lc
        or "haiku-4-6"  in model_lc or "haiku-5"  in model_lc
    )
    def _call(api_key: str) -> dict:
        # timeout explícito → evita que el SDK use su default de 600s y deje
        # threads colgados cuando la red se va. _call_ai_with_timeout sigue
        # actuando como segunda capa de defensa.
        client  = anthropic.Anthropic(api_key=api_key, timeout=90.0)
        content: list = []
        # Orden: IMAGEN 1 (diagrama de contexto, si existe) → IMAGEN 2 (la "buena").
        if req.context_image_b64:
            content.append(_anthropic_image_block(
                req.context_image_b64, req.context_image_mime))
        if req.image_b64:
            content.append(_anthropic_image_block(
                req.image_b64, req.image_mime))
        content.append({"type": "text", "text": req.prompt})
        try:
            create_kwargs: dict = dict(
                model=model,
                max_tokens=16000,
                system=req.system,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
                messages=[{"role": "user", "content": content}],
                betas=["web-search-2025-03-05"],
            )
            if is_new_thinking:
                # Opus 4.7+: adaptive thinking + output_config con effort.
                # Temperature/top_p/top_k OMITIDOS — el modelo los rechaza.
                create_kwargs["thinking"]      = {"type": "adaptive"}
                create_kwargs["output_config"] = {"effort": "high"}
            else:
                # Modelos viejos (Opus 4.6 / Sonnet 4.5): extended thinking enabled.
                create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}
                # temperature OMITIDA: con thinking enabled Anthropic la fuerza a 1.0
            # client.BETA.messages.create (no el estable): `betas=[...]` (web_search)
            # NO es kwarg del namespace estable → con anthropic>=0.96 lanzaría TypeError
            # y caería al fallback básico (sin thinking ni web_search). El namespace
            # beta acepta betas + thinking + output_config en la misma llamada.
            resp = client.beta.messages.create(**create_kwargs)
        except (TypeError, AttributeError):
            # SDK más viejo: sin thinking/web_search → llamada básica. Para
            # modelos viejos podemos pedir determinismo (temperature=0); para
            # Opus 4.7+ NO se puede ni siquiera en el fallback.
            basic_kwargs: dict = dict(
                model=model, max_tokens=4096,
                system=req.system,
                messages=[{"role": "user", "content": content}],
            )
            if not is_new_thinking:
                basic_kwargs["temperature"] = 0.0
            resp = client.messages.create(**basic_kwargs)
        # Extraer texto de la respuesta (múltiples content blocks: text + thinking + tool_use)
        txt = ""
        for block in resp.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                txt += getattr(block, "text", "") or ""
        if not txt and resp.content:
            txt = getattr(resp.content[0], "text", "") or ""
        return {"answer": extract_answer(txt), "raw": txt, "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["ANTHROPIC_API_KEY"],
                            DYNAMIC_CONFIG["ANTHROPIC_API_KEY_BACKUP"], _call)


def _extract_gemini_text(data: dict) -> tuple[str, str]:
    """Extrae el texto de una respuesta de generateContent. Devuelve (txt, motivo).
    Si no hay texto, motivo explica por qué: SAFETY_BLOCK, MAX_TOKENS, RECITATION,
    FINISH_<reason>, EMPTY_CANDIDATES, etc. Esto permite mejorar el log y reintentar
    inteligentemente sin Google Search si fue eso lo que disparó el filtro."""
    txt = ""
    candidates = data.get("candidates") or []
    # Prompt-level block (input rechazado por safety)
    pf = data.get("promptFeedback") or {}
    if pf.get("blockReason"):
        return "", f"PROMPT_BLOCKED_{pf['blockReason']}"
    if not candidates:
        return "", "EMPTY_CANDIDATES"
    for cand in candidates:
        for p in (cand.get("content") or {}).get("parts") or []:
            if "text" in p:
                txt += p["text"]
    if txt:
        return txt, "OK"
    # Sin texto pero hay candidates → revisar finishReason
    fr = (candidates[0] or {}).get("finishReason", "UNKNOWN")
    return "", f"FINISH_{fr}"


def _gemini_thinking_config(model: str):
    """`thinkingConfig` adecuado al modelo. Gemini 3.x usa `thinkingLevel`;
    Gemini 2.5/1.5 NO lo soportan (devuelven 400 'Thinking level is not supported
    for this model') → devolvemos None y lo omitimos, dejando que la API use su
    razonamiento por defecto. _is_gemini_thinking_400 es la red de seguridad para
    cualquier modelo que igualmente lo rechace."""
    m = (model or "").lower()
    if "gemini-3" in m:
        return {"thinkingLevel": "low"}
    return None


def _is_gemini_thinking_400(body: str) -> bool:
    """True si un 400 de Gemini se debe a thinkingConfig/topK (modelo que no los
    soporta) → el caller reintenta sin esos campos. 'thinking' cubre tanto
    'thinkingConfig' como el mensaje 'Thinking level is not supported'."""
    b = (body or "").lower()
    return any(t in b for t in ("thinking", "topk", "top_k", "unknown name"))


def gemini_fn(req: AskRequest) -> dict:
    """Gemini 2.5 Pro con grounding (Google Search) + razonamiento integrado.
    - 2M tokens de contexto
    - google_search tool: búsqueda nativa para preguntas que requieran datos actuales
    """
    model = DYNAMIC_CONFIG.get("GEMINI_MODEL") or GEMINI_MODEL

    def _build_payload(include_tools: bool, max_tokens: int, with_thinking: bool = True) -> dict:
        parts: list = []
        # Orden: IMAGEN 1 (diagrama de contexto, si existe) → IMAGEN 2 (la "buena").
        if req.context_image_b64:
            parts.append(_gemini_image_part(
                req.context_image_b64, req.context_image_mime))
        if req.image_b64:
            parts.append(_gemini_image_part(req.image_b64, req.image_mime))
        parts.append({"text": req.prompt})
        gen_cfg: dict = {
            "temperature":     1.0,   # Google desaconseja <1 (loops/degradación)
            "topP":            0.95,
            "maxOutputTokens": max_tokens,
        }
        if with_thinking:
            # thinkingConfig SOLO para modelos que lo soportan (Gemini 3.x →
            # thinkingLevel). gemini-2.5-pro lo rechaza con 400, así que el helper
            # devuelve None y lo omitimos (era el bug: 400 en cada análisis).
            _tc = _gemini_thinking_config(model)
            if _tc:
                gen_cfg["thinkingConfig"] = _tc
        p: dict = {
            "contents":          [{"role": "user", "parts": parts}],
            "systemInstruction": {"parts": [{"text": req.system}]},
            "generationConfig":  gen_cfg,
        }
        if include_tools:
            p["tools"] = [{"google_search": {}}]
        return p

    def _call(api_key: str) -> dict:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={api_key}")
        # maxOutputTokens alto: Gemini usa pensamiento interno que cuenta contra
        # este límite; con poco margen acaba en MAX_TOKENS sin parts de texto.
        use_thinking = True
        payload = _build_payload(include_tools=True, max_tokens=16384, with_thinking=use_thinking)
        r = _http_post_with_retry(url, payload, retries=2, backoff_s=2.0,
                                  timeout=150.0)  # Gemini analyzer: cap HTTP individual
        # Fallback THINKING: si el modelo rechaza thinkingConfig con 400 (p.ej.
        # gemini-2.5-pro recibiendo thinkingLevel) → reintentar sin thinking.
        if (not r.is_success) and r.status_code == 400 and _is_gemini_thinking_400(r.text):
            logger.warning("Gemini analyzer: model=%s rechazó thinkingConfig — reintento sin thinking", model)
            use_thinking = False
            payload = _build_payload(include_tools=True, max_tokens=16384, with_thinking=use_thinking)
            r = _http_post_with_retry(url, payload, retries=1, backoff_s=2.0, timeout=150.0)
        if not r.is_success:
            # Fallback TOOLS: reintentar sin google_search por si el modelo no lo soporta.
            payload = _build_payload(include_tools=False, max_tokens=16384, with_thinking=use_thinking)
            r = _http_post_with_retry(url, payload, retries=1, backoff_s=2.0,
                                      timeout=150.0)
            if not r.is_success:
                raise RuntimeError(f"Gemini {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
        except ValueError as e:
            raise RuntimeError(f"Gemini JSON inválido: {e}")
        txt, motivo = _extract_gemini_text(data)
        # B19: solo reintentamos si hay evidencia clara de que el reintento
        # AYUDARÁ. Motivos como SAFETY o RECITATION son rechazos firmes del
        # filtro y reintentar gasta una llamada extra sin ganancia. Reducimos
        # la lista a los motivos donde quitar tools o subir budget genuinamente
        # cambia el resultado: MAX_TOKENS (más budget) y FINISH_OTHER/UNKNOWN
        # (errores transitorios). EMPTY_CANDIDATES también vale si el primer
        # call usó tools (a veces google_search consume todo el budget).
        used_tools = ("tools" in (payload or {}))
        retry_motives = {"FINISH_MAX_TOKENS", "FINISH_OTHER", "FINISH_UNKNOWN"}
        if used_tools:
            retry_motives.add("EMPTY_CANDIDATES")
        if not txt and motivo in retry_motives:
            logger.warning("Gemini sin texto (motivo=%s) → reintentando sin tools y +budget",
                           motivo)
            payload = _build_payload(include_tools=False, max_tokens=16384, with_thinking=use_thinking)
            r = _http_post_with_retry(url, payload, retries=1, backoff_s=2.0,
                                      timeout=150.0)
            if r.is_success:
                try:
                    data = r.json()
                    txt, motivo = _extract_gemini_text(data)
                except ValueError:
                    pass
        if not txt:
            # Diagnóstico claro: incluir motivo en el RuntimeError para que el panel
            # muestre exactamente qué pasó (SAFETY block vs MAX_TOKENS vs etc.).
            raise RuntimeError(f"Gemini sin texto ({motivo}). Body: {r.text[:200]}")
        return {"answer": extract_answer(txt), "raw": txt, "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["GEMINI_API_KEY"],
                            DYNAMIC_CONFIG["GEMINI_API_KEY_BACKUP"], _call)


def deepseek_fn(req: AskRequest) -> dict:
    """DeepSeek v4-pro vía API Anthropic-compatible (solo texto).
    En modo VIDEO el prompt ya viene enriquecido con el OCR fusionado, así que
    DeepSeek puede analizarlo sin necesitar la imagen. En modo IMAGEN no se
    incluye en _PROVIDERS (el router lo filtra) porque DeepSeek no soporta visión.
    """
    import anthropic
    model = DYNAMIC_CONFIG.get("DEEPSEEK_MODEL") or DEEPSEEK_MODEL
    def _call(api_key: str) -> dict:
        # timeout explícito (mismo motivo que en claude_fn).
        client = anthropic.Anthropic(
            api_key=api_key,
            base_url="https://api.deepseek.com/anthropic",
            timeout=90.0,
        )
        # deepseek-v4-pro es la IA de RAZONAMIENTO de texto: dejamos el thinking
        # ACTIVADO (importante que razone) pero ACOTADO a 8192 tokens, y subimos
        # max_tokens a 24000 para que, tras razonar, quede margen de SOBRA para el
        # bloque <FINAL>. (Antes con max_tokens=4096 el thinking se comía todo el
        # presupuesto → el bloque `text` salía vacío → extract_answer = "X".)
        base_kwargs = dict(
            model=model,
            max_tokens=24000,
            system=req.system,
            messages=[{"role": "user", "content": req.prompt}],
        )
        try:
            resp = client.messages.create(
                **base_kwargs,
                # temperature/top_p OMITIDOS: en modo thinking se ignoran (DeepSeek)
                # y el SDK Anthropic fuerza temperature=1 con thinking activado.
                thinking={"type": "enabled", "budget_tokens": 8192},
            )
        except (TypeError, AttributeError):
            # El SDK anthropic instalado no acepta el kwarg `thinking` (versión
            # vieja) → reintentar SIN thinking en vez de tumbar la IA. Mismo
            # patrón que claude_fn. (Era el bug: deepseek caía con TypeError en
            # 49ms y quedaba muerto en cada job.)
            resp = client.messages.create(**base_kwargs)
        txt = ""
        for block in resp.content:
            if getattr(block, "type", "") == "text":
                txt += getattr(block, "text", "") or ""
        if not txt and resp.content:
            txt = getattr(resp.content[0], "text", "") or ""
        return {"answer": extract_answer(txt), "raw": txt, "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["DEEPSEEK_API_KEY"],
                            DYNAMIC_CONFIG["DEEPSEEK_API_KEY_BACKUP"], _call)


def mistral_fn(req: AskRequest) -> dict:
    """Mistral Large vía API OpenAI-compatible (analyzer de texto).
    En modo VIDEO el prompt viene enriquecido con OCR fusionado.
    En modo IMAGEN se filtra (Mistral OCR no procesa imagen base64 vía
    chat/completions de la forma simple que usamos).

    NOTA sampling: con `temperature=0` Mistral activa GREEDY SAMPLING y EXIGE
    `top_p=1.0` (no 0.1) — error 3054 "top_p must be 1 when using greedy
    sampling" en caso contrario. El determinismo se mantiene igual: greedy
    sampling SIEMPRE elige el token más probable independientemente del top_p
    (top_p=1 con greedy es equivalente a "sin filtro de núcleo")."""
    model = DYNAMIC_CONFIG.get("MISTRAL_MODEL") or MISTRAL_MODEL
    def _call(api_key: str) -> dict:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user",   "content": req.prompt},
            ],
            # temperature=0 → greedy sampling → top_p DEBE ser 1.0 (constraint
            # del API Mistral). El determinismo no se pierde — greedy ignora
            # top_p de facto, sólo necesita que esté declarado a 1.
            "temperature":  0.0,
            "top_p":        1.0,
            "random_seed":  42,    # Mistral lo llama `random_seed` (no `seed`)
            "max_tokens":   4096,
        }
        r = _http_post_with_retry(
            "https://api.mistral.ai/v1/chat/completions",
            payload, headers=headers, retries=2, backoff_s=2.0,
            timeout=120.0,  # Mistral analyzer: HTTP cap individual para liberar conexión
        )
        if not r.is_success:
            raise RuntimeError(f"Mistral {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            if isinstance(txt, list):
                txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
            if not txt:
                raise KeyError("sin contenido en respuesta")
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Mistral respuesta malformada: {e}")
        return {"answer": extract_answer(txt), "raw": txt, "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["MISTRAL_API_KEY"],
                            DYNAMIC_CONFIG["MISTRAL_API_KEY_BACKUP"], _call)


# ─── OCR de VIDEO (fase 1 cuando el cliente manda video_b64) ─────────────────
# Estas funciones NO votan en la fusión final. Su trabajo es extraer el TEXTO
# del video (preguntas + respuestas visibles). El texto se concatena y se pasa
# como prompt a los modelos analíticos en la fase 2.

def qwen_video_ocr_fn(req: AskRequest) -> dict:
    """Qwen3-VL-Plus via DashScope OpenAI-compatible. Acepta video como data URL
    base64 inline. Hasta 20 min de video, second-level temporal accuracy."""
    if not req.video_b64:
        raise RuntimeError("qwen_video_ocr_fn requiere video_b64")
    model = DYNAMIC_CONFIG.get("QWEN_VIDEO_MODEL") or QWEN_VIDEO_MODEL
    ocr_prompt = OCR_VIDEO_PROMPT
    def _call(api_key: str) -> dict:
        headers = {"Authorization": f"Bearer {api_key}"}
        data_url = f"data:{req.video_mime};base64,{req.video_b64}"
        # Si hay diagrama de contexto, lo metemos como primera imagen con un
        # texto-etiqueta que distingue su rol del vídeo principal.
        msg_content: list = []
        if req.context_image_b64:
            msg_content.append({"type": "text",
                                 "text": "Diagrama de contexto (referencia del caso práctico):"})
            msg_content.append(_gpt_chat_image_block(
                req.context_image_b64, req.context_image_mime))
            msg_content.append({"type": "text",
                                 "text": "Vídeo de la hoja de preguntas a transcribir:"})
        msg_content.append({"type": "video_url", "video_url": {"url": data_url}, "fps": 2})
        msg_content.append({"type": "text", "text": ocr_prompt})
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": msg_content}],
            # Determinismo máximo: temp + top_p + seed. Max_tokens 8192 (era
            # 4096) — exámenes densos con 20+ preguntas truncaban a mitad.
            "max_tokens":  8192,
            "temperature": 0.0,
            "top_p":       0.1,
            "seed":        42,
        }
        r = _http_post_with_retry(
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
            payload, headers=headers, retries=2, backoff_s=2.0,
            timeout=80.0,  # Qwen OCR: cap HTTP individual; el thread tiene 360s
        )
        if not r.is_success:
            raise RuntimeError(f"Qwen OCR {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            if isinstance(txt, list):
                # Algunos formatos devuelven content como array de partes
                txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Qwen OCR respuesta malformada: {e}")
        if not (txt or "").strip():
            try:
                choices = data.get("choices", [])
                first = choices[0] if choices else {}
                logger.warning(
                    "🔎 Qwen OCR vacío · model=%s · choices=%d · "
                    "finish_reason=%s · usage=%s · response_keys=%s",
                    model, len(choices), first.get("finish_reason"),
                    data.get("usage"), list(data.keys()),
                )
            except Exception as diag_exc:
                logger.warning("Qwen OCR vacío · diag falló: %s", diag_exc)
        # NO extraemos A/B/C/D: este es solo OCR, devuelve texto crudo.
        return {"answer": "", "raw": txt or "", "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["QWEN_API_KEY"],
                            DYNAMIC_CONFIG["QWEN_API_KEY_BACKUP"], _call)


def gemini_video_ocr_fn(req: AskRequest) -> dict:
    """Gemini 2.5 Pro vía API nativa, modo video understanding (inline_data)."""
    if not req.video_b64:
        raise RuntimeError("gemini_video_ocr_fn requiere video_b64")
    model = DYNAMIC_CONFIG.get("GEMINI_VIDEO_MODEL") or GEMINI_VIDEO_MODEL
    ocr_prompt = OCR_VIDEO_PROMPT
    def _call(api_key: str) -> dict:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={api_key}")
        # Si el job trae diagrama de contexto, lo metemos como PRIMERA part del
        # mismo turn user. Gemini procesa todas las inline_data del turn como
        # contexto visual sin distinción de orden semántico, así que añadimos
        # un texto-etiqueta antes de cada imagen para que el modelo sepa qué es
        # qué.
        parts: list = []
        if req.context_image_b64:
            parts.append({"text": "Diagrama de contexto (referencia del caso práctico):"})
            parts.append(_gemini_image_part(
                req.context_image_b64, req.context_image_mime))
            parts.append({"text": "Vídeo de la hoja de preguntas a transcribir:"})
        parts.append({"inline_data": {"mime_type": req.video_mime, "data": req.video_b64}})
        parts.append({"text": ocr_prompt})

        def _build_gen_config(strict: bool = True) -> dict:
            """Construye generationConfig. En modo strict incluye topK/thinking
            (config moderna). Si esos campos disparan 400 en el modelo activo,
            el caller recae a un config minimal compatible con la API legacy."""
            cfg: dict = {
                "temperature":     1.0,   # Gemini 3.x desaconseja temp<1 (loops)
                "topP":            0.95,
                "maxOutputTokens": 24576,
            }
            if strict:
                # thinkingConfig SOLO si el modelo lo soporta (Gemini 3.x →
                # thinkingLevel). gemini-2.5-pro lo rechaza con 400; el helper
                # devuelve None y lo omitimos. El fallback strict=False es la red
                # de seguridad para cualquier modelo que igualmente lo rechace.
                _tc = _gemini_thinking_config(model)
                if _tc:
                    cfg["thinkingConfig"] = _tc
            return cfg

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": _build_gen_config(strict=True),
        }
        # Reintentos automáticos en 503 (Gemini sobrecargado por demanda) y 429.
        r = _http_post_with_retry(url, payload, retries=2, backoff_s=2.0,
                                  timeout=80.0)  # Gemini OCR: cap HTTP individual
        # FALLBACK gemini config: si el modelo activo no soporta thinkingConfig/
        # topK (caso visto en prod con gemini-2.5-pro versión vieja, o cuando el
        # usuario configura un modelo Gemini 1.5 que NO conoce esos campos), la
        # API devuelve 400 con un mensaje tipo "Unknown name 'thinkingConfig'"
        # o "Unknown name 'topK'". Detectamos esos marcadores y reintentamos
        # SIN esos campos. Sin esto, todo OCR de Gemini fallaba en cuanto el
        # operador rotaba a un modelo viejo.
        if (not r.is_success) and r.status_code == 400:
            body_lc = (r.text or "").lower()
            triggers = ("thinking", "thinkingconfig", "thinking_config", "thinking_budget",
                         "thinkinglevel", "budget", "topk", "top_k", "unknown name")
            if any(t in body_lc for t in triggers):
                logger.warning(
                    "Gemini Video OCR rechazó config moderna (model=%s, body=%s) — reintentando sin thinkingConfig/topK",
                    model, (r.text or "")[:300],
                )
                payload["generationConfig"] = _build_gen_config(strict=False)
                r = _http_post_with_retry(url, payload, retries=1, backoff_s=2.0,
                                           timeout=80.0)
        if not r.is_success:
            # Diagnóstico ampliado: 200 chars no llegaba para entender muchas
            # de las causas reales de 400 de Gemini ("Request payload size
            # exceeds the limit", "The video file is corrupt", "Invalid mime
            # type", "Model not found", quota exceeded, etc.). Subimos a 800.
            body_preview = (r.text or "")[:800]
            try:
                PersistentErrorLog_logger = logger
                PersistentErrorLog_logger.error(
                    "Gemini Video OCR HTTP %d (model=%s, video_kb=%d): %s",
                    r.status_code, model,
                    (len(req.video_b64) * 3 // 4 // 1024) if req.video_b64 else 0,
                    body_preview,
                )
            except Exception:
                pass
            raise RuntimeError(f"Gemini Video OCR {r.status_code}: {body_preview}")
        try:
            data = r.json()
            txt = ""
            for cand in data.get("candidates", []):
                for p in cand.get("content", {}).get("parts", []):
                    if "text" in p:
                        txt += p["text"]
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Gemini Video OCR respuesta malformada: {e}")
        # Diagnóstico: si devuelve texto vacío, loguear toda la metadata útil
        # para entender por qué (safety filter, MAX_TOKENS, sin candidatos, etc.).
        if not (txt or "").strip():
            try:
                cands = data.get("candidates", [])
                first = cands[0] if cands else {}
                logger.warning(
                    "🔎 Gemini OCR vacío · model=%s · candidates=%d · "
                    "finishReason=%s · safetyRatings=%s · promptFeedback=%s · "
                    "usage=%s · response_keys=%s",
                    model, len(cands),
                    first.get("finishReason"),
                    first.get("safetyRatings"),
                    data.get("promptFeedback"),
                    data.get("usageMetadata"),
                    list(data.keys()),
                )
            except Exception as diag_exc:
                logger.warning("Gemini OCR vacío · diag falló: %s", diag_exc)
        return {"answer": "", "raw": txt or "", "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["GEMINI_API_KEY"],
                            DYNAMIC_CONFIG["GEMINI_API_KEY_BACKUP"], _call)


def kimi_video_ocr_fn(req: AskRequest) -> dict:
    """Kimi K2.6 (Moonshot AI) vía API OpenAI-compatible. Acepta video como
    data URL base64 inline. Visión via MoonViT 400M encoder.

    NOTA: Kimi K2.5/K2.6 SOLO acepta `temperature=1.0` (error 400 "invalid
    temperature: only 1 is allowed for this model" con cualquier otro valor).
    Tampoco se puede usar `top_p`/`seed` para forzar determinismo. Greedy/
    sampling sigue funcionando porque Moonshot lo aplica internamente cuando
    el prompt es de tipo OCR (no creativo)."""
    if not req.video_b64:
        raise RuntimeError("kimi_video_ocr_fn requiere video_b64")
    model = DYNAMIC_CONFIG.get("KIMI_VIDEO_MODEL") or KIMI_VIDEO_MODEL
    ocr_prompt = OCR_VIDEO_PROMPT
    def _call(api_key: str) -> dict:
        headers = {"Authorization": f"Bearer {api_key}"}
        data_url = f"data:{req.video_mime};base64,{req.video_b64}"
        msg_content: list = []
        if req.context_image_b64:
            msg_content.append({"type": "text",
                                 "text": "Diagrama de contexto (referencia del caso práctico):"})
            msg_content.append(_gpt_chat_image_block(
                req.context_image_b64, req.context_image_mime))
            msg_content.append({"type": "text",
                                 "text": "Vídeo de la hoja de preguntas a transcribir:"})
        msg_content.append({"type": "video_url", "video_url": {"url": data_url}})
        msg_content.append({"type": "text", "text": ocr_prompt})
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": msg_content}],
            "max_tokens":  16384,  # subido: K2.6 va en thinking mode y el razonamiento gasta tokens → con 8192 salía vacío (finish_reason=length)
            "temperature": 1.0,    # Kimi K2.5/K2.6 sólo acepta 1.0 (error 400 en otros)
        }
        r = _http_post_with_retry(
            "https://api.moonshot.ai/v1/chat/completions",
            payload, headers=headers, retries=2, backoff_s=2.0,
            timeout=80.0,  # Kimi OCR: cap HTTP individual
        )
        if not r.is_success:
            raise RuntimeError(f"Kimi OCR {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            if isinstance(txt, list):
                txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Kimi OCR respuesta malformada: {e}")
        if not (txt or "").strip():
            try:
                choices = data.get("choices", [])
                first = choices[0] if choices else {}
                logger.warning(
                    "🔎 Kimi OCR vacío · model=%s · choices=%d · "
                    "finish_reason=%s · usage=%s · response_keys=%s",
                    model, len(choices), first.get("finish_reason"),
                    data.get("usage"), list(data.keys()),
                )
            except Exception as diag_exc:
                logger.warning("Kimi OCR vacío · diag falló: %s", diag_exc)
        return {"answer": "", "raw": txt or "", "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["KIMI_API_KEY"],
                            DYNAMIC_CONFIG["KIMI_API_KEY_BACKUP"], _call)


def mimo_video_ocr_fn(req: AskRequest) -> dict:
    """MiMo-V2.5 (Xiaomi) vía API OpenAI-compatible. Acepta video como
    data URL base64 inline. Capability full-modal (texto+imagen+audio+video).
    Diferencias clave vs los otros 3 proveedores:
      • Header de auth es `api-key: <KEY>` (NO `Authorization: Bearer`).
      • Doc usa `max_completion_tokens` en lugar de `max_tokens` — ambos
        funcionan en práctica porque MiMo es OpenAI-compat, pero ponemos el
        canónico de su docs para ir sobre seguro.
      • Endpoint base: https://api.xiaomimimo.com/v1/chat/completions
    """
    if not req.video_b64:
        raise RuntimeError("mimo_video_ocr_fn requiere video_b64")
    model = DYNAMIC_CONFIG.get("MIMO_VIDEO_MODEL") or MIMO_VIDEO_MODEL
    ocr_prompt = OCR_VIDEO_PROMPT
    def _call(api_key: str) -> dict:
        # MiMo usa header `api-key:` no Bearer (confirmado en docs Xiaomi).
        headers = {"api-key": api_key}
        data_url = f"data:{req.video_mime};base64,{req.video_b64}"
        msg_content: list = []
        if req.context_image_b64:
            msg_content.append({"type": "text",
                                 "text": "Diagrama de contexto (referencia del caso práctico):"})
            msg_content.append(_gpt_chat_image_block(
                req.context_image_b64, req.context_image_mime))
            msg_content.append({"type": "text",
                                 "text": "Vídeo de la hoja de preguntas a transcribir:"})
        msg_content.append({"type": "video_url", "video_url": {"url": data_url}})
        msg_content.append({"type": "text", "text": ocr_prompt})
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": msg_content,
            }],
            # 16384 en vez de 8192: como otros vendors con thinking interno
            # (Gemini/Kimi), 8192 quedaba justo en exámenes densos y devolvía
            # content vacío con finish_reason=length. El doble da margen sin
            # afectar latencia notable (MiMo solo "facturará" lo que consuma).
            "max_completion_tokens": 16384,
            "temperature": 0.0,
            "top_p":       0.1,
            "seed":        42,
        }
        r = _http_post_with_retry(
            "https://api.xiaomimimo.com/v1/chat/completions",
            payload, headers=headers, retries=2, backoff_s=2.0,
            timeout=80.0,  # MiMo OCR: cap HTTP individual
        )
        if not r.is_success:
            raise RuntimeError(f"MiMo OCR {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            if isinstance(txt, list):
                # Algunos formatos devuelven content como array de partes
                txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"MiMo OCR respuesta malformada: {e}")
        if not (txt or "").strip():
            try:
                choices = data.get("choices", [])
                first = choices[0] if choices else {}
                logger.warning(
                    "🔎 MiMo OCR vacío · model=%s · choices=%d · "
                    "finish_reason=%s · content_filter=%s · usage=%s · response_keys=%s",
                    model, len(choices),
                    first.get("finish_reason"),
                    first.get("content_filter_results"),
                    data.get("usage"),
                    list(data.keys()),
                )
            except Exception as diag_exc:
                logger.warning("MiMo OCR vacío · diag falló: %s", diag_exc)
        return {"answer": "", "raw": txt or "", "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["MIMO_API_KEY"],
                            DYNAMIC_CONFIG["MIMO_API_KEY_BACKUP"], _call)


def _extract_top_k_frames_jpeg_b64(
    video_b64: str, video_mime: str,
    top_k: int = 3, jpeg_quality: int = 85,
    samples_per_segment: int = 10,
) -> List[Tuple[str, str]]:
    """Extrae los `top_k` frames más nítidos del video como JPEGs, con
    diversidad temporal: divide el video en `top_k` segmentos y elige el
    frame con mayor Laplacian variance (sharpness) de cada uno.

    Por qué top_k frames y no solo el del medio:
      - Claude/GPT no soportan video nativo → en el viejo helper se mandaba
        SOLO el frame del medio. Si la cámara se movía justo ahí (papel
        girando, mano temblando), ese frame estaba borroso y el OCR fallaba
        aunque el resto del clip estuviera bien.
      - 3 frames de tramos distintos del video dan al modelo varias poses
        del mismo papel → si una está borrosa, las otras 2 le permiten
        reconstruir las preguntas. Las APIs vision de Anthropic y OpenAI
        aceptan múltiples bloques `image`/`image_url` por mensaje.

    Por qué Laplacian variance:
      - Estándar de facto para sharpness scoring (OpenCV docs / PyImageSearch).
        Sobel+Variance y Tenengrad dan resultados similares pero Laplacian
        es 1 pasada con kernel 3×3 → más rápido y menos código.
      - Calculado sobre el frame reducido a 320px max-side: misma decisión
        relativa (qué frame es más nítido) con ~10× menos CPU.

    Diversidad temporal:
      - Si simplemente ordenamos TODOS los frames por sharpness y tomamos
        los top 3, salen casi-consecutivos (los frames próximos a uno
        nítido también lo son). Eso desperdicia 2 slots.
      - Dividir el video en `top_k` ventanas temporales y elegir UNO de
        cada una garantiza que cubrimos toda la duración del clip.

    Returns lista vacía si cv2 no está o falla. NO lanza nunca.
    """
    if not video_b64 or top_k <= 0:
        return []
    tmp_path = None
    cap = None
    try:
        import cv2
        import numpy as np  # noqa: F401  (cv2 lo importa internamente)
        import tempfile
        ext = ".mp4" if "mp4" in (video_mime or "") else ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            tmp_path = f.name
            f.write(base64.b64decode(video_b64, validate=False))
        cap = cv2.VideoCapture(tmp_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n_frames <= 0:
            cap.release()
            return []

        # Atajo: si piden 1 frame o el video tiene MUY pocos frames, fallback
        # al frame del medio (comportamiento clásico del helper antiguo).
        if top_k == 1 or n_frames <= top_k:
            target = max(0, (n_frames // 2) - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return []
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                return []
            return [(base64.b64encode(jpg.tobytes()).decode("ascii"), "image/jpeg")]

        # Dividir el video en top_k ventanas temporales y elegir UN frame
        # nítido de cada una (diversidad temporal + calidad).
        seg_size = n_frames // top_k
        selected: List[Tuple[str, str]] = []
        for k in range(top_k):
            seg_start = k * seg_size
            seg_end = (k + 1) * seg_size if k < top_k - 1 else n_frames
            step = max(1, (seg_end - seg_start) // max(1, samples_per_segment))
            # Criterio de selección LEXICOGRÁFICO (has_sheet, sharp): se prefiere
            # un frame que muestre el folio COMPLETO (4 esquinas detectables →
            # enderezable) sobre uno más nítido pero sin folio. A igualdad de
            # has_sheet, gana el más nítido. Si ningún frame del segmento muestra
            # el folio, se queda con el más nítido (comportamiento clásico).
            best_key = (False, -1.0)   # (has_sheet, sharp)
            best_frame = None
            idx = seg_start
            while idx < seg_end:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, fr = cap.read()
                if ok and fr is not None:
                    h, w = fr.shape[:2]
                    # Reducir a 320 max-side para evaluar sharpness rápido
                    # (Laplacian es O(píxeles) → bajar resolución 10× = 10× speed).
                    scale = 320.0 / float(max(w, h, 1))
                    if scale < 1.0:
                        fr_small = cv2.resize(
                            fr, (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        fr_small = fr
                    try:
                        gray = cv2.cvtColor(fr_small, cv2.COLOR_BGR2GRAY)
                        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    except Exception:
                        sharp = 0.0
                    # ¿El frame muestra el folio completo? (detección a 640px sobre
                    # el frame original). Se antepone a la nitidez: un folio entero
                    # algo menos nítido vale más que medio folio muy nítido.
                    has_sheet = _frame_shows_sheet(fr)
                    key = (has_sheet, sharp)
                    if key > best_key:
                        best_key = key
                        # Guardamos el frame ORIGINAL (resolución completa) para
                        # codificar el JPEG sin pérdidas de la reducción a 320px.
                        best_frame = fr
                idx += step
            if best_frame is not None:
                ok, jpg = cv2.imencode(".jpg", best_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if ok:
                    selected.append((base64.b64encode(jpg.tobytes()).decode("ascii"), "image/jpeg"))
        cap.release()
        return selected
    except ImportError:
        # cv2 no instalado — no es fatal, el caller decide cómo reaccionar.
        return []
    except Exception as exc:
        # ERROR: sin frames pre-extraídos el panel no muestra previews del
        # video y los OCR page-based (Mistral/DeepSeek/GLM) fallan al
        # buscar imagen. Pipeline VIDEO degradado.
        logger.error("_extract_top_k_frames_jpeg_b64 falló: %s", exc)
        return []
    finally:
        # Liberar cap SIEMPRE (no solo en el camino feliz): en excepción el
        # release() inline no se ejecutaba → fuga de file handle del decoder.
        # Idempotente: si ya se llamó inline, el segundo release es no-op.
        if cap is not None:
            try: cap.release()
            except Exception: pass
        if tmp_path is not None:
            try: os.unlink(tmp_path)
            except OSError: pass


def _stack_top_frames_jpeg_b64(
    video_b64: str,
    video_mime: str,
    n_keep: int = 15,
    samples_total: int = 60,
    jpeg_quality: int = 92,
    max_side: int = 2048,
) -> Optional[Tuple[str, str]]:
    """Genera UNA imagen fusionada de altísima calidad a partir del vídeo.

    Pipeline (port directo de AdvancedBurstProcessor.kt del cliente Android):
      1. **Lucky imaging**: muestrea `samples_total` frames espaciados del clip,
         calcula sharpness = var(Laplacian(gray)) / mean(gray) (normalizada por
         luma → no penaliza tomas con menos luz), ordena descendente y se queda
         con los `n_keep` mejores.
      2. **Alineación sub-píxel densa**: para cada frame≠ref calcula Optical
         Flow Farneback contra el frame de referencia (el más nítido), luego
         remapea el frame a coordenadas del ref con cv2.remap+INTER_LANCZOS4.
         Operamos a 1/4 de resolución para que Farneback sea rápido (~150ms
         por par) y escalamos el flow a la resolución completa.
      3. **Fusión anti-ghosting**: por cada píxel, peso = 1/(|diff|+10) donde
         diff = |ref - alt_alineado|. Los píxeles que cambiaron mucho (mano,
         objetos en movimiento) pesan menos → el contenido estático del folio
         se refuerza por √N y el ruido temporal se cancela.
      4. **CLAHE** sobre canal L de Lab (no toca chroma): contrast limited
         adaptive histogram equalization clipLimit=2.0, tile 8×8.
      5. **Unsharp masking**: blur Gaussiano σ=1.5, addWeighted 1.5/-0.5 →
         realce de bordes (= texto).
      6. **Resize a max_side 2048** (si fuera mayor) y JPEG quality 92.

    Returns (jpeg_b64, "image/jpeg") en éxito, None en cualquier fallo.
    NUNCA lanza — devuelve None para que el caller decida fallback al frame
    crudo del Laplacian.

    Coste típico: 2.5-4 s para clip de 4 s a 720p en CPU Render (1 vCPU).
    Reduce ruido temporal ~√15 ≈ 3.9×, mejor SNR de texto sustancialmente.
    """
    if not video_b64:
        return None
    tmp_path = None
    cap = None
    try:
        import cv2
        import numpy as np
        import tempfile
        ext = ".mp4" if "mp4" in (video_mime or "") else ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            tmp_path = f.name
            f.write(base64.b64decode(video_b64, validate=False))
        cap = cv2.VideoCapture(tmp_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n_frames < 2:
            return None

        # ──── PASO 1: lucky imaging — sample + sharpness ────
        sample_idx = max(1, n_frames // max(1, samples_total))
        candidates: list = []  # [(has_sheet, sharpness, frame_bgr)]
        idx = 0
        while idx < n_frames and len(candidates) < samples_total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok and fr is not None:
                # Sharpness normalizada por luma (mismo patrón que el cliente
                # Android: lapVar / grayMean → estable bajo cambios de luz).
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                lap_var = float(lap.var())
                gray_mean = float(gray.mean())
                sharpness = lap_var / gray_mean if gray_mean > 1.0 else lap_var
                # has_sheet: ¿el frame muestra el folio completo? Se antepone a la
                # nitidez al elegir los n_keep que se fusionan (mismo criterio que
                # _extract_top_k_frames). Así la fusión parte de tomas con la hoja
                # entera → el dewarp posterior la endereza mejor y el ref de
                # alineación (top_frames[0]) es un folio completo nítido.
                has_sheet = _frame_shows_sheet(fr)
                candidates.append((has_sheet, sharpness, fr))
            idx += sample_idx
        if len(candidates) < 2:
            return None

        # Orden lexicográfico: primero los que muestran el folio completo, y
        # dentro de cada grupo, por nitidez descendente. Si <n_keep tienen folio,
        # se completan con los más nítidos sin folio (degradación natural).
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top_frames = [fr for _, _, fr in candidates[:n_keep]]
        # Liberar YA los frames candidatos NO seleccionados: `candidates` retiene
        # hasta `samples_total` frames BGR a resolución completa (~6MB c/u a 1080p).
        # Sin este del coexistían en RAM con los `aligned_floats` en float32 de la
        # fase siguiente → pico que reventaba los 512MB de Render. Tras esto solo
        # quedan los n_keep frames de `top_frames`.
        del candidates
        ref = top_frames[0]
        h, w = ref.shape[:2]

        # Cap de memoria por resolución: el stacking trabaja en float32 (4 bytes/
        # canal) → un frame 1080p son ~24MB y uno 4K ~95MB en float32. Reescalamos
        # los top_frames a `max_side` ANTES de la fase float32 (antes el resize era
        # el ÚLTIMO paso, tras stackear a resolución completa → pico enorme con
        # fuentes 4K). Solo actúa si la fuente excede max_side; 1080p no se toca.
        _long_side = max(h, w)
        if _long_side > max_side:
            _scale = max_side / float(_long_side)
            _new_wh = (max(1, int(round(w * _scale))), max(1, int(round(h * _scale))))
            top_frames = [cv2.resize(f, _new_wh, interpolation=cv2.INTER_AREA)
                          for f in top_frames]
            ref = top_frames[0]
            h, w = ref.shape[:2]

        # ──── PASO 2: alineación densa con Optical Flow Farneback ────
        # Trabajamos a 1/4 de resolución para Farneback (más rápido), luego
        # escalamos el flow a tamaño completo. Las coordenadas base son
        # constantes (cada píxel a su posición original).
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref_gray_small = cv2.resize(ref_gray, (w // 4, h // 4),
                                     interpolation=cv2.INTER_AREA)

        # mapX/mapY base: matriz de coordenadas (c, r) que usaremos como
        # punto de partida. Sumamos el flow para obtener las coords destino.
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                      np.arange(h, dtype=np.float32))

        aligned_floats = [ref.astype(np.float32)]
        for fr in top_frames[1:]:
            alt_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            alt_gray_small = cv2.resize(alt_gray, (w // 4, h // 4),
                                         interpolation=cv2.INTER_AREA)
            try:
                flow_small = cv2.calcOpticalFlowFarneback(
                    ref_gray_small, alt_gray_small, None,
                    0.5, 5, 15, 3, 7, 1.5, 0,
                )
            except cv2.error:
                # Si Farneback falla (caso raro: frame demasiado pequeño),
                # añadimos el frame sin alinear — la máscara anti-ghosting
                # ya se encargará de dejarlo con peso bajo.
                aligned_floats.append(fr.astype(np.float32))
                continue
            flow_large = cv2.resize(flow_small, (w, h),
                                     interpolation=cv2.INTER_LINEAR) * 4.0
            map_x = grid_x + flow_large[..., 0]
            map_y = grid_y + flow_large[..., 1]
            warped = cv2.remap(fr, map_x, map_y, cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REPLICATE)
            aligned_floats.append(warped.astype(np.float32))

        # ──── PASO 3: fusión anti-ghosting (pesos = 1/(diff+10)) ────
        ref_f = ref.astype(np.float32)
        accumulator = np.zeros_like(ref_f)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        for aligned_f in aligned_floats:
            diff = np.abs(ref_f - aligned_f)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 1.5)
            weight = 1.0 / (diff_gray + 10.0)
            weight_3c = cv2.merge([weight, weight, weight])
            accumulator += aligned_f * weight_3c
            weight_sum += weight
        weight_sum_3c = cv2.merge([weight_sum, weight_sum, weight_sum])
        # Evita división por cero (impossible con +10 floor, pero por si acaso)
        fused = np.divide(accumulator, weight_sum_3c, where=weight_sum_3c > 1e-6)
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        # Liberar YA los float32 grandes (aligned_floats + accumulator ≈ el PICO de
        # RAM del stacking, ~0.3-0.5 GB con 8 frames a 2400px). Los pasos 4-6 solo
        # usan `fused` (uint8) → sin esto, esos ~300MB seguían vivos durante CLAHE/
        # unsharp. En Render 2GB recorta el pico antes de los pasos finales.
        del aligned_floats, accumulator, weight_sum, weight_sum_3c, ref_f

        # Recorte de bordes (8px) — zonas donde el remap no tuvo info válida
        if h > 32 and w > 32:
            fused = fused[8:h - 8, 8:w - 8]

        # ──── PASO 4: CLAHE sobre canal L de Lab ────
        lab = cv2.cvtColor(fused, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L = clahe.apply(L)
        lab = cv2.merge([L, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

        # ──── PASO 5: unsharp masking ────
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0.0)

        # ──── PASO 6: resize si excede max_side + JPEG 92 ────
        sh, sw = sharpened.shape[:2]
        if max(sh, sw) > max_side:
            scale = max_side / float(max(sh, sw))
            new_w = max(1, int(sw * scale))
            new_h = max(1, int(sh * scale))
            sharpened = cv2.resize(sharpened, (new_w, new_h),
                                    interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", sharpened,
                                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            return None
        return (base64.b64encode(jpg.tobytes()).decode("ascii"), "image/jpeg")
    except ImportError:
        logger.warning("_stack_top_frames_jpeg_b64: cv2/numpy no instalados — "
                       "imposible hacer stacking server-side")
        return None
    except Exception as exc:
        # ERROR (no warning): sin stacking, los OCR page-based pierden la
        # imagen casi-perfecta y caen al frame top-1 crudo (peor calidad OCR).
        # El operador necesita verlo en VER ERRORES.
        logger.error("_stack_top_frames_jpeg_b64 falló: %s", exc)
        return None
    finally:
        if cap is not None:
            try: cap.release()
            except Exception: pass
        if tmp_path is not None:
            try: os.unlink(tmp_path)
            except OSError: pass


def _topaz_enhance_image(
    image_b64: str,
    image_mime: str = "image/jpeg",
) -> Optional[Tuple[str, str]]:
    """Llama a la Image API de Topaz Labs (modelo Wonder 3) para super-resolution
    + denoise sobre una imagen ya fusionada. Flujo async oficial:

        POST /image/v1/enhance/async  →  { "process_id": "xxx" }
        GET  /image/v1/status/{pid}   (poll cada 2s)  →  "Completed"
        GET  /image/v1/download/{pid} →  { "url": "..." }
        GET  download_url             →  bytes del JPEG mejorado

    Resilencia:
      - Timeout global TOPAZ_TIMEOUT_S (default 60s). Si excede, devuelve None.
      - HTTP 429: backoff exponencial (3 reintentos máx).
      - Si el modelo TOPAZ_MODEL no existe (HTTP 400 "model not found"),
        fallback automático a "Wonder 2" (el modelo anterior, equivalente
        funcional aunque con menos calidad de edge preservation).
      - Cualquier otro fallo → return None (el caller usa el input como
        fallback).

    Returns (jpeg_b64, mime) en éxito, None en error/timeout/no-key.
    """
    if not DYNAMIC_CONFIG.get("TOPAZ_ENABLED"):
        return None
    api_key = (DYNAMIC_CONFIG.get("TOPAZ_API_KEY") or "").strip()
    if not api_key:
        return None
    try:
        timeout_s = float(DYNAMIC_CONFIG.get("TOPAZ_TIMEOUT_S") or 60.0)
    except (TypeError, ValueError):
        timeout_s = 60.0
    deadline = time.monotonic() + timeout_s
    model = (DYNAMIC_CONFIG.get("TOPAZ_MODEL") or "Wonder 3").strip()
    try:
        output_height = int(DYNAMIC_CONFIG.get("TOPAZ_OUTPUT_HEIGHT") or 0)
    except (TypeError, ValueError):
        output_height = 0
    headers = {"X-API-KEY": api_key}

    try:
        img_bytes = base64.b64decode(image_b64, validate=False)
    except Exception as exc:
        # ERROR (validación): si llega base64 inválido a Topaz, hay un bug
        # upstream en el stacking o en el cache de fused_image. Visible en panel.
        logger.error("Topaz: input b64 inválido (%s)", exc)
        return None

    # ──── PASO 1: POST async submit ────
    # _http_post_with_retry no soporta multipart, así que usamos httpx directo.
    # Backoff exponencial manual en HTTP 429 (consigna API Topaz: respect
    # rate limits with exponential backoff).
    import httpx

    def _submit(model_name: str) -> Optional[str]:
        files = {"image": ("input.jpg", img_bytes, image_mime or "image/jpeg")}
        data = {"model": model_name, "output_format": "jpeg"}
        if output_height > 0:
            data["output_height"] = str(output_height)
        r = None
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                r = httpx.post(
                    "https://api.topazlabs.com/image/v1/enhance/async",
                    headers=headers, files=files, data=data,
                    timeout=min(30.0, max(5.0, remaining)),
                )
                if r.status_code == 429:
                    time.sleep(min(2.0 ** attempt, remaining))
                    continue
                break
            except httpx.RequestError as exc:
                if attempt == 2:
                    logger.warning("Topaz POST excepción: %s", exc)
                    return None
                time.sleep(min(2.0 ** attempt, max(0.0, remaining)))
        if r is None or not r.is_success:
            sc = r.status_code if r is not None else "no-response"
            body = (r.text or "")[:160] if r is not None else ""
            logger.warning("Topaz POST %s → HTTP %s: %s", model_name, sc, body)
            return None
        try:
            return (r.json() or {}).get("process_id")
        except ValueError:
            return None

    pid = _submit(model)
    if not pid and model != "Wonder 2":
        # Fallback automático: si "Wonder 3" no está disponible en esta cuenta
        # (rollout incompleto, depreciación), reintentamos con "Wonder 2".
        logger.info("Topaz: '%s' falló · fallback a 'Wonder 2'", model)
        pid = _submit("Wonder 2")
    if not pid:
        return None

    # ──── PASO 2: poll status hasta Completed / Failed / timeout ────
    poll_url = f"https://api.topazlabs.com/image/v1/status/{pid}"
    while time.monotonic() < deadline:
        try:
            sr = httpx.get(poll_url, headers=headers, timeout=10.0)
        except httpx.RequestError as exc:
            logger.warning("Topaz status request falló: %s", exc)
            time.sleep(2.0)
            continue
        if sr.status_code == 429:
            time.sleep(3.0)
            continue
        if not sr.is_success:
            logger.warning("Topaz status %s: %s", sr.status_code,
                           (sr.text or "")[:160])
            return None
        try:
            st = (sr.json() or {}).get("status", "")
        except ValueError:
            st = ""
        st_lc = st.lower()
        if "complet" in st_lc:
            break
        if "fail" in st_lc or "error" in st_lc:
            logger.warning("Topaz status=Failed (pid=%s)", pid[:8] if pid else "?")
            return None
        time.sleep(2.0)
    else:
        logger.warning("Topaz timeout %ss esperando resultado (pid=%s)",
                       timeout_s, pid[:8] if pid else "?")
        return None

    # ──── PASO 3: download URL → bytes ────
    try:
        dr = httpx.get(f"https://api.topazlabs.com/image/v1/download/{pid}",
                       headers=headers, timeout=15.0)
    except httpx.RequestError as exc:
        logger.warning("Topaz download request falló: %s", exc)
        return None
    if not dr.is_success:
        logger.warning("Topaz download HTTP %s", dr.status_code)
        return None
    try:
        download_url = (dr.json() or {}).get("url") or (dr.json() or {}).get("download_url")
    except ValueError:
        return None
    if not download_url:
        # Algunas variantes devuelven los bytes directos en lugar de URL.
        if dr.content and len(dr.content) > 1024:
            return (base64.b64encode(dr.content).decode("ascii"), "image/jpeg")
        return None
    try:
        fr = httpx.get(download_url, timeout=max(5.0, deadline - time.monotonic()))
    except httpx.RequestError as exc:
        logger.warning("Topaz GET fichero falló: %s", exc)
        return None
    if not fr.is_success or not fr.content:
        return None
    return (base64.b64encode(fr.content).decode("ascii"), "image/jpeg")


def _extract_video_frame_jpeg_b64(video_b64: str, video_mime: str) -> Optional[tuple]:
    """Backwards-compat: extrae 1 frame (el del medio) como JPEG (b64, mime).
    Mantenido por si algún caller externo todavía lo usa; los OCR de
    Anthropic/OpenAI usan ahora `_extract_top_k_frames_jpeg_b64` con top_k=3.
    """
    frames = _extract_top_k_frames_jpeg_b64(video_b64, video_mime, top_k=1, jpeg_quality=80)
    return frames[0] if frames else None


def anthropic_video_ocr_fn(req: AskRequest) -> dict:
    """Claude (Anthropic) como OCR de video usando capacidad de visión.

    Claude NO soporta video nativo (solo imágenes en formato jpeg/png/gif/webp).
    Estrategia:
      1) Si el cliente manda `image_b64` (frame único representativo), Claude
         lo usa directo (1 imagen).
      2) Si solo viene `video_b64`, extraemos los 3 frames más nítidos del
         video (top-K por Laplacian variance con diversidad temporal) y los
         mandamos juntos. Antes solo se mandaba el frame del medio: si la
         cámara se movía justo en ese instante, Claude veía un frame borroso
         y fallaba el OCR aunque el resto del clip estuviera nítido.
      3) Si cv2 no está instalado o falla, lanzamos error y este OCR queda
         fuera de la fusión (los otros proveedores siguen votando).

    Layout del mensaje (best practice Anthropic vision docs 2026):
      - Imágenes ANTES del texto → Claude construye el contexto visual antes
        de leer el prompt y puede referirlas como "Frame 1/2/3".
      - Cada imagen precedida de un text block "Frame N:" → evita que Claude
        confunda cuál imagen es cuál (recomendación oficial).

    Usa la MISMA api key que el analyzer Claude (ANTHROPIC_API_KEY).
    """
    if not req.image_b64 and not req.video_b64:
        raise RuntimeError("anthropic_video_ocr_fn requiere image_b64 o video_b64")
    frames: List[Tuple[str, str]] = []
    if req.image_b64:
        frames = [(req.image_b64, req.image_mime or "image/jpeg")]
    else:
        frames = _extract_top_k_frames_jpeg_b64(req.video_b64, req.video_mime, top_k=3)
        if not frames:
            raise RuntimeError(
                "Claude OCR requiere frames del video. El cliente puede mandar "
                "image_b64 (recomendado) o instala opencv-python-headless para que "
                "el servidor extraiga los frames más nítidos automáticamente."
            )
    import anthropic
    model = DYNAMIC_CONFIG.get("CLAUDE_VIDEO_OCR_MODEL") or CLAUDE_VIDEO_OCR_MODEL
    ocr_prompt = OCR_VIDEO_PROMPT
    # Opus 4.7+ deprecó temperature/top_p/top_k — devuelven 400. Detección por
    # nombre de modelo: si es 4.7+ los omitimos; si es 4.6 o anterior los pasamos.
    model_lc = model.lower()
    is_new_thinking = (
        "opus-4-7"  in model_lc or "opus-4-8"  in model_lc or "opus-5"    in model_lc
        or "sonnet-4-7" in model_lc or "sonnet-5" in model_lc
        or "haiku-4-6"  in model_lc or "haiku-5"  in model_lc
    )
    def _call(api_key: str) -> dict:
        client = anthropic.Anthropic(api_key=api_key, timeout=90.0)
        # Construir content con imágenes ANTES del texto y labels "Frame N:"
        # cuando hay más de una (best practice oficial Anthropic).
        # Si hay diagrama de contexto, va el PRIMERO con su propio label
        # "Diagrama de contexto:" para que Claude no lo confunda con un frame.
        content: list = []
        if req.context_image_b64:
            content.append({"type": "text", "text": "Diagrama de contexto (referencia del caso práctico):"})
            content.append(_anthropic_image_block(
                req.context_image_b64, req.context_image_mime))
        for i, (b64, mime) in enumerate(frames, start=1):
            if len(frames) > 1:
                content.append({"type": "text", "text": f"Frame {i}:"})
            content.append({"type": "image", "source": {"type": "base64",
                                                          "media_type": mime, "data": b64}})
        content.append({"type": "text", "text": ocr_prompt})
        create_kwargs: dict = dict(
            model=model,
            max_tokens=8192,   # 4096 truncaba exámenes densos
            messages=[{"role": "user", "content": content}],
        )
        if not is_new_thinking:
            # Modelos clásicos: determinismo via temperature/top_p
            create_kwargs["temperature"] = 0.0
            create_kwargs["top_p"]       = 0.1
        # Opus 4.7+: parámetros omitidos (el modelo es determinístico por diseño
        # con prompts de tipo OCR; no se puede ajustar sampling vía API).
        resp = client.messages.create(**create_kwargs)
        txt = ""
        for block in resp.content:
            if getattr(block, "type", "") == "text":
                txt += getattr(block, "text", "") or ""
        if not txt and resp.content:
            txt = getattr(resp.content[0], "text", "") or ""
        return {"answer": "", "raw": txt or "", "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["ANTHROPIC_API_KEY"],
                            DYNAMIC_CONFIG["ANTHROPIC_API_KEY_BACKUP"], _call)


def openai_video_ocr_fn(req: AskRequest) -> dict:
    """OpenAI (GPT-4o / GPT-5.5) como OCR de video usando visión sobre frames.

    OpenAI tampoco soporta video nativo — misma estrategia que Claude:
      1) Si el cliente manda `image_b64`, GPT lo usa directo.
      2) Si solo viene `video_b64`, mandamos los 3 frames más nítidos por
         Laplacian variance (diversidad temporal). Antes era SOLO el frame
         del medio → cualquier blur en ese instante hundía el OCR.

    Layout del mensaje (best practice OpenAI cookbook 2026):
      - Texto ANTES de las imágenes → el prompt prepara al modelo para qué
        buscar y mejora la extracción multi-imagen (es lo CONTRARIO de
        Claude, ambas best practices están en sus docs respectivas).
      - Labels "Frame N:" entre imágenes para evitar confusión.

    Reusa OPENAI_API_KEY del analyzer (no se duplica en la UI).
    """
    if not req.image_b64 and not req.video_b64:
        raise RuntimeError("openai_video_ocr_fn requiere image_b64 o video_b64")
    frames: List[Tuple[str, str]] = []
    if req.image_b64:
        frames = [(req.image_b64, req.image_mime or "image/jpeg")]
    else:
        frames = _extract_top_k_frames_jpeg_b64(req.video_b64, req.video_mime, top_k=3)
        if not frames:
            raise RuntimeError(
                "OpenAI OCR requiere frames del video. El cliente puede mandar "
                "image_b64 (recomendado) o instala opencv-python-headless para que "
                "el servidor extraiga los frames más nítidos automáticamente."
            )
    model = DYNAMIC_CONFIG.get("OPENAI_VIDEO_OCR_MODEL") or OPENAI_VIDEO_OCR_MODEL
    ocr_prompt = OCR_VIDEO_PROMPT
    def _call(api_key: str) -> dict:
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        # Texto primero (best practice OpenAI), luego diagrama de contexto si
        # existe, y por último cada frame con su label.
        content: list = [{"type": "text", "text": ocr_prompt}]
        if req.context_image_b64:
            content.append({"type": "text", "text": "Diagrama de contexto (referencia del caso práctico):"})
            content.append(_gpt_chat_image_block(
                req.context_image_b64, req.context_image_mime))
        for i, (b64, mime) in enumerate(frames, start=1):
            if len(frames) > 1:
                content.append({"type": "text", "text": f"Frame {i}:"})
            data_url = f"data:{mime};base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        payload = {
            "model":       model,
            "messages":    [{"role": "user", "content": content}],
            "max_tokens":  8192,
            "temperature": 0.0,
            "top_p":       0.1,
            "seed":        42,
        }
        r = _http_post_with_retry(
            "https://api.openai.com/v1/chat/completions",
            payload, headers=headers, retries=2, backoff_s=2.0,
            timeout=80.0,  # OpenAI OCR-video sobre frames: cap HTTP individual
        )
        if not r.is_success:
            raise RuntimeError(f"OpenAI OCR {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            if isinstance(txt, list):
                txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"OpenAI OCR respuesta malformada: {e}")
        return {"answer": "", "raw": txt or "", "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["OPENAI_API_KEY"],
                            DYNAMIC_CONFIG["OPENAI_API_KEY_BACKUP"], _call)


def nvidia_fn(req: AskRequest) -> dict:
    """NVIDIA Llama-3.3 Nemotron Super 49B — analyzer de RAZONAMIENTO de texto.

    Endpoint NIM Cloud: integrate.api.nvidia.com/v1/chat/completions (OpenAI-compat).
    Texto-only — en modo VIDEO el prompt viene enriquecido con OCR fusionado.
    En modo IMAGEN se filtra (Nemotron Super es texto puro, sin vision).

    El modelo soporta TOGGLE de reasoning vía system prompt:
      - "detailed thinking on"  → activa modo CoT profundo (recomendado para
        exámenes tipo test, donde queremos que razone sobre cada pregunta).
      - "detailed thinking off" → respuesta directa, más rápida.
    Lo prefijamos al `req.system` que ya recibe el modelo, igual que hacen los
    otros analyzers.

    Auth: Authorization: Bearer <NVIDIA_API_KEY>.
    """
    model = DYNAMIC_CONFIG.get("NVIDIA_MODEL") or NVIDIA_MODEL
    # Activar reasoning Nemotron explícitamente — sin esto el modelo responde
    # en modo no-thinking por defecto. El system prompt original del caller
    # se mantiene tras el toggle.
    system_full = "detailed thinking on\n\n" + (req.system or "")
    def _call(api_key: str) -> dict:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_full},
                {"role": "user",   "content": req.prompt},
            ],
            # Nemotron con "detailed thinking on": NVIDIA recomienda temp 0.6 /
            # top_p 0.95 (greedy temp=0 con reasoning ON → loops/degradación).
            "temperature":  0.6,
            "top_p":        0.95,
            "max_tokens":   4096,
            "stream":       False,
        }
        r = _http_post_with_retry(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            payload, headers=headers, retries=2, backoff_s=2.0,
            timeout=120.0,  # Nemotron Super 49B con reasoning on puede tardar
        )
        if not r.is_success:
            raise RuntimeError(f"NVIDIA {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            if isinstance(txt, list):
                txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
            if not txt:
                raise KeyError("sin contenido en respuesta")
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"NVIDIA respuesta malformada: {e}")
        return {"answer": extract_answer(txt), "raw": txt, "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["NVIDIA_API_KEY"],
                            DYNAMIC_CONFIG.get("NVIDIA_API_KEY_BACKUP", ""), _call)


def _detect_sheet_quad(img, detect_long_side: int = 1000,
                       min_area_frac: float = 0.25,
                       aspect_lo: float = 1.15, aspect_hi: float = 1.75):
    """Detecta el cuadrilátero de una HOJA (A4) en una imagen BGR (np.ndarray).

    Devuelve las 4 esquinas ordenadas (TL, TR, BR, BL) como np.float32 en
    coordenadas de `img`, o None si no hay un cuadrilátero plausible. Es la
    MISMA lógica que usa el dewarp → "se detecta folio" ⇔ "el dewarp lo podrá
    enderezar". Por eso se comparte en dos sitios:
      - _dewarp_document: para enderezar la imagen fusionada.
      - selección de frames (_extract_top_k_frames / _stack_top_frames): como
        criterio de prioridad "este frame muestra el folio completo", que se
        antepone a la nitidez al elegir qué frames van al OCR.

    `detect_long_side` controla la resolución de detección: 1000 para el dewarp
    final (preciso); menor (p.ej. 640) en el scoring de muchos frames (rápido).
    NUNCA lanza: devuelve None ante cualquier fallo.
    """
    try:
        import cv2
        import numpy as np
        if img is None:
            return None
        h0, w0 = img.shape[:2]
        if h0 < 200 or w0 < 200:
            return None
        # Detección sobre versión reducida (rápido).
        scale = float(detect_long_side) / float(max(h0, w0))
        small = (cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                 if scale < 1.0 else img)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        frame_area = float(small.shape[0] * small.shape[1])
        quad = None
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
            if cv2.contourArea(c) < min_area_frac * frame_area:   # la hoja llena ≥min_area_frac del frame
                break                                              # (orden desc → resto menores)
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                break
        if quad is None:
            return None
        if scale < 1.0:
            quad = quad / scale                           # → coords de la imagen original
        # Ordenar esquinas: TL, TR, BR, BL.
        s = quad.sum(axis=1)
        d = quad[:, 0] - quad[:, 1]
        ordered = np.array([quad[int(np.argmin(s))], quad[int(np.argmax(d))],
                            quad[int(np.argmax(s))], quad[int(np.argmin(d))]],
                           dtype=np.float32)
        out_w = int(max(np.linalg.norm(ordered[2] - ordered[3]),
                        np.linalg.norm(ordered[1] - ordered[0])))
        out_h = int(max(np.linalg.norm(ordered[1] - ordered[2]),
                        np.linalg.norm(ordered[0] - ordered[3])))
        if out_w < 100 or out_h < 100:
            return None
        # Sanity A4 (~1.41): rechaza cuadriláteros falsos (borde de mesa, etc.)
        # que distorsionarían en vez de enderezar.
        aspect = max(out_w, out_h) / float(min(out_w, out_h))
        if not (aspect_lo <= aspect <= aspect_hi):
            return None
        return ordered
    except Exception:
        return None


def _frame_shows_sheet(img, detect_long_side: int = 640) -> bool:
    """True si el frame BGR muestra un folio A4 completo (detectable/enderezable).
    Wrapper booleano de _detect_sheet_quad para el scoring de selección de
    frames: en la selección solo importa el sí/no (priorizar folios completos),
    no la precisión sub-píxel, así que detecta a menor resolución que el dewarp."""
    return _detect_sheet_quad(img, detect_long_side=detect_long_side) is not None


def _dewarp_document(jpeg_b64: str, mime: str = "image/jpeg",
                     out_long_side: int = 2400,
                     jpeg_quality: int = 92) -> Optional[Tuple[str, str]]:
    """Detecta el cuadrilátero de la HOJA (A4) en una imagen y CORRIGE la
    perspectiva (dewarp) para enderezarla. Pensado para cámara en el pecho con
    ángulo oblicuo: enderezar la hoja mejora mucho el OCR de imagen.

    Devuelve (jpeg_b64, "image/jpeg") si detecta una hoja plausible; None si no
    encuentra un cuadrilátero claro (el caller conserva la imagen original).
    NUNCA lanza — devuelve None ante cualquier fallo (fallback al método actual).
    """
    if not jpeg_b64:
        return None
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(base64.b64decode(jpeg_b64, validate=False),
                                         dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Detección de las 4 esquinas (MISMA lógica que el scoring de frames:
        # _detect_sheet_quad). El dewarp usa resolución de detección 1000 (default).
        ordered = _detect_sheet_quad(img)
        if ordered is None:
            return None
        out_w = int(max(np.linalg.norm(ordered[2] - ordered[3]),
                        np.linalg.norm(ordered[1] - ordered[0])))
        out_h = int(max(np.linalg.norm(ordered[1] - ordered[2]),
                        np.linalg.norm(ordered[0] - ordered[3])))
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                       dtype=np.float32)
        warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(ordered, dst),
                                     (out_w, out_h), flags=cv2.INTER_LANCZOS4)
        lh, lw = warped.shape[:2]
        if max(lh, lw) > out_long_side:
            r = out_long_side / float(max(lh, lw))
            warped = cv2.resize(warped, (int(lw * r), int(lh * r)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", warped,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            return None
        return (base64.b64encode(buf.tobytes()).decode("ascii"), "image/jpeg")
    except Exception as exc:
        logger.warning("[dewarp] falló (se conserva la imagen original): %s", exc)
        return None


def _get_image_for_page_ocr(
    req: AskRequest, wait_for_fused_s: Optional[float] = None,
) -> Tuple[str, str]:
    """Devuelve (b64, mime) de la mejor imagen disponible para OCR de página
    única (Mistral OCR / DeepSeek OCR / GLM OCR). Orden de preferencia:

      1. `req.image_b64`         (modo IMAGEN del cliente — sin video)
      2. `req.fused_image_b64`   (stacking + Topaz, cuando llegue)
      3. `_extract_top_k_frames_jpeg_b64(top_k=1)` clásico (fallback final)

    Espera DINÁMICA por la fusión:
      • Si TOPAZ_ENABLED + TOPAZ_API_KEY → wait = TOPAZ_TIMEOUT_S + 5s
        (típicamente 65s) — da tiempo a que Topaz termine y la versión
        SR llegue a `req.fused_image_b64`. Si Topaz cae/timeout, el
        background ya cacheó el stacking local antes (~4s), así que NO
        esperamos los 65s reales en el caso fallido — `req.fused_image_b64`
        ya está rellena con stacking_local y salimos enseguida.
      • Si Topaz desactivado → wait = 12s (solo stacking, da margen para
        videos largos donde la decodificación cv2 tarda más).

    Levanta RuntimeError si NADA está disponible (cv2 no instalado + sin
    image_b64 + sin frames).
    """
    if req.image_b64:
        return (req.image_b64, req.image_mime or "image/jpeg")

    # Calcular el wait máximo. Sólo lo usamos como TECHO: en cuanto
    # `req.fused_image_b64` esté listo, salimos sin esperar más.
    if wait_for_fused_s is None:
        topaz_active = bool(DYNAMIC_CONFIG.get("TOPAZ_ENABLED")
                             and (DYNAMIC_CONFIG.get("TOPAZ_API_KEY") or "").strip())
        if topaz_active:
            try:
                topaz_to = float(DYNAMIC_CONFIG.get("TOPAZ_TIMEOUT_S") or 60.0)
            except (TypeError, ValueError):
                topaz_to = 60.0
            wait_for_fused_s = topaz_to + 5.0
        else:
            wait_for_fused_s = 12.0

    if wait_for_fused_s > 0 and req.video_b64:
        deadline = time.monotonic() + wait_for_fused_s
        # Estrategia de salida temprana:
        # 1) En cuanto `fused_image_b64` cambie por SEGUNDA vez (Topaz pisó al
        #    stacking_local), salimos al toque — ya tenemos la versión SR.
        # 2) Si solo apareció una vez (stacking_local) y pasó `topaz_grace_s`,
        #    asumimos que Topaz cayó/timeout y salimos con esa.
        # 3) Si ni apareció (cv2 falló), esperamos hasta deadline y caemos al
        #    fallback Laplacian abajo.
        first_len: Optional[int] = None
        first_seen_at: Optional[float] = None
        topaz_grace_s = 30.0
        while time.monotonic() < deadline:
            cur_b64 = req.fused_image_b64
            if cur_b64:
                cur_len = len(cur_b64)
                if first_len is None:
                    first_len = cur_len
                    first_seen_at = time.monotonic()
                elif cur_len != first_len:
                    # Cambió de tamaño → Topaz pisó al stacking. Salir.
                    break
                elif time.monotonic() - first_seen_at >= topaz_grace_s:
                    # Mismo b64 tras el grace → Topaz no llegó. Salir con stacking.
                    break
            time.sleep(0.1)

    if req.fused_image_b64:
        return (req.fused_image_b64, req.fused_image_mime or "image/jpeg")

    # Fallback al comportamiento clásico: top-1 frame por Laplacian.
    if req.video_b64:
        frames = _extract_top_k_frames_jpeg_b64(req.video_b64, req.video_mime, top_k=1)
        if frames:
            return frames[0]
    raise RuntimeError(
        "OCR page-based requiere image_b64, video_b64 con cv2, o fused_image_b64."
    )


def mistral_image_ocr_fn(req: AskRequest) -> dict:
    """Mistral OCR 3 (`mistral-ocr-2512`) sobre el MEJOR frame extraído del video.

    Por qué este proveedor pese a tener ya 7 OCRs de video:
      - Mistral OCR es una API DEDICADA de OCR (no un LLM general con vision):
        99%+ accuracy en 11+ idiomas, mejor que frontier LLMs en OCR puro
        sobre documento estático según los benchmarks OmniDocBench/OCRBench
        2026.
      - Los 7 OCRs existentes son LLMs visión: comparten un sesgo común (todos
        salen de la misma generación de modelos multimodales y a veces alucinan
        las mismas preguntas). Una API de OCR pura añade DIVERSIDAD REAL en
        la fusión multi-OCR — el majority-vote es más robusto cuando los
        votantes son arquitecturalmente distintos.
      - Coste despreciable: $2/1000 páginas → ~$0.002 por job (1 frame).
      - Solo MANDAMOS EL MEJOR FRAME (top_k=1 por Laplacian variance + tramo
        central del clip): Mistral OCR es por documento; mandarle 3 frames
        serían 3 transcripciones que romperían el Needleman-Wunsch del
        fusionador (pensaría que hay 3× preguntas).

    Endpoint: POST https://api.mistral.ai/v1/ocr
    Body: { "model": "mistral-ocr-2512", "document": { "type": "image_url",
            "image_url": "data:image/jpeg;base64,..." } }
    Response: { "pages": [{"index", "markdown", "images", "dimensions"}], ... }
    """
    if not req.image_b64 and not req.video_b64:
        raise RuntimeError("mistral_image_ocr_fn requiere image_b64 o video_b64")
    # Prefiere la fusión multi-frame (stacking + Topaz Wonder 3) si está
    # disponible — calidad MUCHO mejor que el frame crudo del Laplacian.
    # Wait DINÁMICO (None → lógica Topaz-aware): 65s si Topaz activo, con salida
    # temprana en cuanto Topaz pisa al stacking (~15s típico) o grace de 30s si
    # Topaz cae · 12s si Topaz desactivado. Sin esto, Topaz no llegaba a tiempo
    # (~10-20s) y el OCR usaba siempre el stacking crudo. Si nada llega a tiempo,
    # cae al top-1 clásico por Laplacian. El wait está acotado por deadline:
    # nunca cuelga, y va oculto bajo los OCR multi-frame (Claude/GPT, 40-90s).
    img_b64, img_mime = _get_image_for_page_ocr(req)
    model = DYNAMIC_CONFIG.get("MISTRAL_OCR_MODEL") or MISTRAL_OCR_MODEL
    def _call(api_key: str) -> dict:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        data_url = f"data:{img_mime};base64,{img_b64}"
        payload = {
            "model": model,
            "document": {"type": "image_url", "image_url": data_url},
        }
        r = _http_post_with_retry(
            "https://api.mistral.ai/v1/ocr",
            payload, headers=headers, retries=2, backoff_s=2.0,
            timeout=80.0,  # Mistral OCR: cap HTTP individual (página única es rápido)
        )
        if not r.is_success:
            raise RuntimeError(f"Mistral OCR {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            pages = data.get("pages") or []
            # Concatenar markdown de TODAS las páginas (típicamente 1 para imagen
            # única, pero defensivo por si en el futuro Mistral pagina internamente).
            txt_parts = []
            for p in pages:
                if isinstance(p, dict):
                    md = p.get("markdown") or p.get("text") or ""
                    if md:
                        txt_parts.append(md)
            txt = "\n\n".join(txt_parts)
            if not txt:
                # Algunas variantes devuelven `text` a nivel raíz. Fallback defensivo.
                txt = data.get("text") or ""
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise RuntimeError(f"Mistral OCR respuesta malformada: {e}")
        return {"answer": "", "raw": txt or "", "model": model}
    return _try_with_backup(DYNAMIC_CONFIG["MISTRAL_API_KEY"],
                            DYNAMIC_CONFIG.get("MISTRAL_API_KEY_BACKUP", ""), _call)


def deepseek_image_ocr_fn(req: AskRequest) -> dict:
    """DeepSeek-OCR (`deepseek-ocr`) sobre el MEJOR frame extraído del video.

    DeepSeek-OCR es un modelo OCR-DEDICATED de DeepSeek (no es DeepSeek-V3 con
    vision genérica). Aporta diversidad arquitectural distinta a Mistral OCR y
    a los LLMs visión generales. Endpoint OpenAI-compatible — reutiliza
    DEEPSEEK_API_KEY que ya está configurada para el analyzer.

    Igual que Mistral OCR: top_k=1 best frame para no romper el parser del
    fusionador con preguntas duplicadas.
    """
    if not req.image_b64 and not req.video_b64:
        raise RuntimeError("deepseek_image_ocr_fn requiere image_b64 o video_b64")
    # Mismo helper que Mistral OCR — wait dinámico Topaz-aware, fallback a top-1 frame.
    img_b64, img_mime = _get_image_for_page_ocr(req)
    model = DYNAMIC_CONFIG.get("DEEPSEEK_OCR_MODEL") or DEEPSEEK_OCR_MODEL
    # DeepSeek-OCR vive en un proveedor TERCERO (SiliconFlow por defecto), NO en
    # api.deepseek.com. Base URL y key propios (DEEPSEEK_OCR_*).
    base = (DYNAMIC_CONFIG.get("DEEPSEEK_OCR_BASE_URL")
            or "https://api.novita.ai/openai/v1").rstrip("/")
    ocr_prompt = OCR_VIDEO_PROMPT
    def _call(api_key: str) -> dict:
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type":  "application/json"}
        data_url = f"data:{img_mime};base64,{img_b64}"
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": ocr_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            "max_tokens":  8192,
            "temperature": 0.0,
            "top_p":       0.1,
            "seed":        42,
        }
        r = _http_post_with_retry(
            f"{base}/chat/completions",
            payload, headers=headers, retries=2, backoff_s=2.0,
            timeout=80.0,
        )
        if not r.is_success:
            raise RuntimeError(f"DeepSeek OCR {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            if isinstance(txt, list):
                txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"DeepSeek OCR respuesta malformada: {e}")
        return {"answer": "", "raw": txt or "", "model": model}
    # Key: usa DEEPSEEK_OCR_API_KEY si la pones; si no, la de Novita (el OCR va por
    # defecto a Novita). Si ambas vacías → NoApiKeyError → se salta sin error fatal.
    _ocr_key = (DYNAMIC_CONFIG.get("DEEPSEEK_OCR_API_KEY")
                or DYNAMIC_CONFIG.get("NOVITA_API_KEY") or "")
    return _try_with_backup(_ocr_key,
                            DYNAMIC_CONFIG.get("NOVITA_API_KEY_BACKUP", ""), _call)


def glm_image_ocr_fn(req: AskRequest) -> dict:
    """GLM-OCR (Z.AI / Zhipu) — top OmniDocBench v1.5 (94.62) y +19 puntos vs
    PaddleOCR-VL en OCRBench. Modelo OCR-dedicated 0.9B con encoder CogViT.

    Endpoint específico: POST /api/paas/v4/layout_parsing
    Schema oficial (verificado en docs.z.ai/api-reference/tools/layout-parsing):
      Request body:
        - model: "glm-ocr"
        - file:  URL pública HTTPS O data URL base64 ("data:image/jpeg;base64,...")
      Response:
        - md_results:      Markdown completo del documento (campo principal)
        - layout_details:  Array por bloque con {content, type, bbox, ...}
                           — útil como fallback si md_results llega vacío
    Si Z_AI_API_KEY no está configurada, NoApiKeyError → no participa en el voto.

    Mando top_k=1 best frame (página única). Límite GLM-OCR: imagen ≤ 10MB,
    nuestros JPEGs típicos están en 200-500KB → muy por debajo.

    Idiomas soportados: chino, inglés, francés, ESPAÑOL, ruso, alemán, japonés,
    coreano. Por eso peso 1.3 en el fusionador (TIER 2 para texto en español).
    """
    if not req.image_b64 and not req.video_b64:
        raise RuntimeError("glm_image_ocr_fn requiere image_b64 o video_b64")
    # Mismo helper — wait dinámico Topaz-aware, fallback al top-1 clásico.
    img_b64, img_mime = _get_image_for_page_ocr(req)
    model = DYNAMIC_CONFIG.get("GLM_OCR_MODEL") or GLM_OCR_MODEL
    def _call(api_key: str) -> dict:
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type":  "application/json"}
        data_url = f"data:{img_mime};base64,{img_b64}"
        # Campo del request: "file" (NO "file_url" — antes lo tenía mal).
        # Acepta URL HTTPS o data URL base64 según docs oficiales.
        payload = {
            "model": model,
            "file":  data_url,
        }
        r = _http_post_with_retry(
            "https://api.z.ai/api/paas/v4/layout_parsing",
            payload, headers=headers, retries=2, backoff_s=2.0,
            timeout=80.0,
        )
        if not r.is_success:
            raise RuntimeError(f"GLM OCR {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            # 1) Campo principal según docs: `md_results` (markdown completo).
            txt = data.get("md_results") or ""
            # 2) Fallback: concatenar `layout_details[].content` por bloque
            #    (útil si Z.AI devuelve solo el array detallado sin md_results
            #    consolidado — p.ej. en imágenes con layout muy complejo).
            if not txt:
                details = data.get("layout_details") or []
                parts = []
                for d in details:
                    if isinstance(d, dict):
                        c = d.get("content") or ""
                        if c: parts.append(c)
                txt = "\n\n".join(parts)
            # 3) Fallbacks adicionales por si Z.AI cambia el schema sin avisar
            #    (vimos esto pasar con otros vendors); cuestan 0 si md_results
            #    ya pobló txt.
            if not txt:
                txt = data.get("text") or data.get("markdown") or ""
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise RuntimeError(f"GLM OCR respuesta malformada: {e}")
        return {"answer": "", "raw": txt or "", "model": model}
    return _try_with_backup(DYNAMIC_CONFIG.get("Z_AI_API_KEY", ""),
                            DYNAMIC_CONFIG.get("Z_AI_API_KEY_BACKUP", ""), _call)


# Lista de proveedores OCR (flujo VIDEO, fase 1).
# Se ejecutan EN PARALELO y sus respuestas se fusionan (Needleman-Wunsch +
# weighted majority vote) en una transcripción canónica para fase 2 (analyzers).
#
# CUATRO categorías arquitecturalmente distintas para maximizar diversidad
# (los benchmarks 2026 muestran que ensembles de OCRs heterogéneos reducen
# 30-50% el char error rate vs el mejor OCR individual — research apunta a
# que predicciones correctas convergen y errores divergen, "Consensus Entropy"):
#
#   1. Video nativo LLM (5): Qwen, Gemini, Kimi, MiMo, NVIDIA — MP4 entero
#      vía data URL inline. Sesgo común entre LLMs multimodales.
#   2. Multi-frame LLM (2): Anthropic, OpenAI — top-3 frames por Laplacian
#      variance con diversidad temporal. Mejor que el frame medio si la
#      cámara se mueve.
#   3. OCR puro de imagen DEDICATED (3): Mistral OCR 3, DeepSeek-OCR, GLM-OCR
#      — APIs especializadas en OCR documental (no LLMs generales). Reciben
#      el MEJOR frame único. Aportan diversidad arquitectural fuerte.
#
# En `_OCR_PROVIDER_WEIGHTS` (más abajo) los OCRs dedicated reciben peso
# extra en el voto: están específicamente entrenados para la tarea y
# tienen menos alucinación creativa que un LLM general.
_OCR_PROVIDERS = [
    ("qwen_ocr",      qwen_video_ocr_fn),
    ("gemini_ocr",    gemini_video_ocr_fn),
    ("kimi_ocr",      kimi_video_ocr_fn),
    # DESACTIVADO 2026-05-22: api.xiaomimimo.com hacía timeout (~251s, 3 intentos)
    # en cada job → solo gastaba tiempo/hilo y daba error, nunca devolvía OCR.
    # Reactivar (descomentar) si el endpoint de Xiaomi vuelve a responder.
    # ("mimo_ocr",      mimo_video_ocr_fn),
    ("anthropic_ocr", anthropic_video_ocr_fn),
    ("openai_ocr",    openai_video_ocr_fn),
    ("mistral_ocr",   mistral_image_ocr_fn),
    ("deepseek_ocr",  deepseek_image_ocr_fn),
    ("glm_ocr",       glm_image_ocr_fn),
]

# OCRs VIDEO-NATIVOS: procesan el clip MP4 entero, ven todas las preguntas
# aunque la cámara se mueva entre tomas. Estos son los ÚNICOS que cuentan para
# determinar cuántas preguntas tiene el examen (el cap del cluster_cap).
#
# Los otros OCRs (anthropic/openai/mistral/deepseek/glm) son page-based: solo
# ven UN frame extraído del video. Si la cámara estaba mal posicionada en ese
# frame, pueden ver una hoja CORTADA con menos preguntas (8 en lugar de 13).
# Si los page-based contaran para el cap, una imagen recortada hundiría el
# total esperado y descartaríamos preguntas reales.
#
# IMPORTANTE: los page-based SÍ contribuyen al VOTO del texto de cada pregunta
# y al voto del NUM ("1.", "2.", ...) — su precisión por-pregunta es alta, solo
# su CONTEO es poco fiable cuando ven un crop parcial. Esta separación afecta
# ÚNICAMENTE al target del cap, no al matching ni al voto de contenido.
_VIDEO_OCR_NAMES = frozenset({"qwen_ocr", "gemini_ocr", "kimi_ocr", "mimo_ocr"})

# El resto (page-based) — solo para diagnóstico
_PAGE_OCR_NAMES = frozenset({"anthropic_ocr", "openai_ocr", "mistral_ocr",
                              "deepseek_ocr", "glm_ocr"})


# Pesos por proveedor en el majority vote del fusionador. Ajustados para el
# caso de uso real del usuario: HOJAS DE EXAMEN TIPO TEST EN ESPAÑOL.
#
# Best practice 2026 (Consensus Entropy, weighted ensemble voting): los
# pesos deben reflejar accuracy real del proveedor en la TAREA y el
# IDIOMA específicos. Aplicar peso uniforme infraestima a los OCRs que
# dominan en latin/español y sobre-pesa los que están entrenados con
# foco en chino o escenarios distintos a documentos.
#
# Tiering basado en benchmarks 2026 (Mistral docs, OCR Arena, OmniDocBench
# v1.5, multilingual eval scores, vendor language support):
#
#   TIER 1 (1.5) — Top en español documentado:
#     · Mistral OCR 3 — vendor 99%+ accuracy en 11+ idiomas (latin family,
#       incluido español). API dedicada de OCR.
#     · Gemini 2.5 Pro — ELO 1665 en OCR Arena (#4 global). Top printed-
#       media composite 85%. Multilingual nativo Google.
#     · Qwen3-VL — Alibaba expandió OCR de 19→32 idiomas con español como
#       latin core. Upgrade explícito en multilingual robustness.
#
#   TIER 2 (1.3) — Fuerte en español:
#     · GLM-OCR (Z.AI) — SOTA OmniDocBench v1.5 (94.62%), top multilingual
#       eval (69.3 vs Paddle 54.8). Soporta español oficialmente.
#     · Claude Opus (anthropic_ocr) — entrenado fuerte en latin/español
#       documentado, multi-frame con top-3 frames.
#     · GPT-4o/GPT-5 (openai_ocr) — 95% handwriting benchmark, multi-frame
#       con top-3. Buen rendimiento en español aunque optimizado para EN.
#
#   TIER 3 (1.0) — Aceptable en español:
#     · DeepSeek-OCR — foco original chino-inglés; español funcional pero
#       no es su prioridad de training.
#     · NVIDIA Nemotron — general purpose multilingual sin foco específico.
#
#   TIER 4 (<1.0) — Débil/no-prioritarios en español:
#     · Kimi K2.6 (Moonshot) — foco primario chino, multimodal reciente.
#     · MiMo (Xiaomi) — documentado en el technical report como "modest
#       degradation on OCR tasks", optimizado para home scenarios.
#
# Estos pesos están coordinados con el threshold de mayoría (>=2 OCRs
# físicos): el peso solo desempata cuando hay tie de count. En ese caso,
# 2 tier-1 (3.0) tumban a 2 tier-3 (2.0) pero NO tumban a 3 tier-2 (3.9).
# Si en logs ves "vote_weighted_tiebreak" causando regresiones, ajusta
# aquí sin tocar resto del pipeline.
# Mapeo provider OCR → key en DYNAMIC_CONFIG. Los pesos son AHORA EDITABLES
# desde el panel /panel (antes estaban hardcoded como floats; el usuario no
# podía tunear OCRs específicos sin tocar código). Defaults tier-based en
# DYNAMIC_CONFIG arriba. Escala 0..10 (0 = ignorar este OCR en el voto).
_OCR_PROVIDER_WEIGHT_KEY: dict = {
    "mistral_ocr":   "MISTRAL_OCR_WEIGHT",
    "gemini_ocr":    "GEMINI_OCR_WEIGHT",
    "qwen_ocr":      "QWEN_OCR_WEIGHT",
    "glm_ocr":       "GLM_OCR_WEIGHT",
    "anthropic_ocr": "ANTHROPIC_OCR_WEIGHT",
    "openai_ocr":    "OPENAI_OCR_WEIGHT",
    "deepseek_ocr":  "DEEPSEEK_OCR_WEIGHT",
    "kimi_ocr":      "KIMI_OCR_WEIGHT",
    "mimo_ocr":      "MIMO_OCR_WEIGHT",
}


def _ocr_provider_weight(provider: str) -> float:
    """Peso del provider OCR en el majority vote. Lee de DYNAMIC_CONFIG (editable
    desde el panel). Default 1.0 si el provider no está mapeado (futuro OCR
    añadido sin tunear). Tolera lecturas concurrentes (CONFIG_LOCK no necesario:
    dict[k] retorna el valor actual o el anterior, ambos válidos)."""
    key = _OCR_PROVIDER_WEIGHT_KEY.get(provider)
    if not key:
        return 1.0
    try:
        return float(DYNAMIC_CONFIG.get(key, 1) or 0)
    except (TypeError, ValueError):
        return 1.0


# Lista de proveedores analíticos (fase única en modo IMAGEN, fase 2 en modo VIDEO).
# En modo IMAGEN se filtran los text-only (deepseek, mistral) — ver _providers_for_request.
def minimax_fn(req: AskRequest) -> dict:
    """MiniMax-M2 vía Novita (OpenAI-compat) — analyzer de RAZONAMIENTO de texto.

    Razonamiento de frontera con arquitectura ligera (~10B activos) → potente y
    RÁPIDO. Texto-only: en modo VIDEO el prompt ya viene enriquecido con el OCR
    fusionado. Endpoint: https://api.novita.ai/openai/v1/chat/completions.
    Auth: Bearer NOVITA_API_KEY. Si la key falta → NoApiKeyError → se salta.
    """
    model = DYNAMIC_CONFIG.get("MINIMAX_MODEL") or MINIMAX_MODEL
    base = (DYNAMIC_CONFIG.get("NOVITA_BASE_URL")
            or "https://api.novita.ai/openai/v1").rstrip("/")
    def _call(api_key: str) -> dict:
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type":  "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": req.system or ""},
                {"role": "user",   "content": req.prompt},
            ],
            "temperature":  0.0,
            "max_tokens":   8192,   # margen para que razone Y emita el <FINAL>
            "stream":       False,
        }
        r = _http_post_with_retry(
            f"{base}/chat/completions",
            payload, headers=headers, retries=2, backoff_s=2.0,
            timeout=120.0,   # M2 es rápido, pero damos aire al razonamiento
        )
        if not r.is_success:
            raise RuntimeError(f"MiniMax {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            msg = data["choices"][0]["message"]
            txt = msg.get("content") or ""
            if isinstance(txt, list):
                txt = "".join(p.get("text", "") for p in txt if isinstance(p, dict))
            # Modelos con thinking: si `content` viene vacío, el answer puede estar
            # en `reasoning_content` → fallback para no perder la respuesta.
            if not txt:
                txt = msg.get("reasoning_content") or ""
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"MiniMax respuesta malformada: {e}")
        return {"answer": extract_answer(txt), "raw": txt, "model": model}
    return _try_with_backup(DYNAMIC_CONFIG.get("NOVITA_API_KEY", ""),
                            DYNAMIC_CONFIG.get("NOVITA_API_KEY_BACKUP", ""), _call)


_PROVIDERS = [
    ("gpt",      gpt_fn),
    ("claude",   claude_fn),
    ("gemini",   gemini_fn),
    ("deepseek", deepseek_fn),
    ("mistral",  mistral_fn),
    ("nvidia",   nvidia_fn),
    ("minimax",  minimax_fn),
]

# Providers que NO procesan imagen directamente. En modo IMAGEN se excluyen.
# En modo VIDEO sí participan (reciben el texto OCR del prompt enriquecido).
# Nemotron Super 49B es texto-only (Llama-based, sin vision), igual que DeepSeek/Mistral.
_TEXT_ONLY_PROVIDERS = {"deepseek", "mistral", "nvidia", "minimax"}


def _providers_for_request(req: AskRequest) -> list:
    """Devuelve los proveedores analíticos válidos según la request.
    - En modo IMAGEN: deepseek y mistral se excluyen (no soportan visión).
    - En modo VIDEO: los 5 modelos analizan el texto OCR previo.
    """
    if req.video_b64:
        return _PROVIDERS  # los 5
    # Modo imagen: filtrar text-only providers
    return [(name, fn) for (name, fn) in _PROVIDERS if name not in _TEXT_ONLY_PROVIDERS]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"ok": True, "service": "bolsillo-ia-relay"}

@app.get("/health")
def health(key: str = "", x_api_key: Optional[str] = Header(None)):
    """Diagnóstico completo: si algo va raro, aquí se ve.

    C9: si NO se envía key/X-Api-Key válida, devolvemos respuesta minimalista
    (solo ok=true/false). El diagnóstico completo expone IDs de Appwrite/Supabase
    y métricas internas — útil para el operador pero filtra info en un scan
    anónimo. El probe de Render (typically anon GET) sigue funcionando porque
    el modo minimalista devuelve 200 si todo OK."""
    now = time.time()
    # Modo público: si no hay auth, devolvemos minimal. Mantiene compatibilidad
    # con health checks de Render que no autentican.
    auth_key = x_api_key or key
    authorized = _is_authorized_key(auth_key)
    with LOCK:
        total      = len(JOBS)
        n_pend     = sum(1 for j in JOBS.values() if j.get("status") == "pending")
        n_review   = sum(1 for j in JOBS.values() if j.get("status") == "awaiting_review")
        n_done     = sum(1 for j in JOBS.values() if j.get("status") == "done")
        n_err      = sum(1 for j in JOBS.values() if j.get("status") == "error")
        oldest     = min((_safe_float(j.get("created"), now) for j in JOBS.values()), default=now)
        with_img   = sum(1 for j in JOBS.values() if j.get("img"))

    # Estado de los daemons (debería estar vivo todos)
    daemons = {}
    for name, st in DAEMON_HEALTH.items():
        age = int(now - st["last_ok"]) if st["last_ok"] else -1
        # Tolerancia: gc=60s, watchdog=30s, flusher=event-driven
        threshold = {"gc": 180, "watchdog": 90, "flusher": 600}.get(name, 180)
        alive = (st["last_ok"] > 0) and (age <= threshold)
        daemons[name] = {
            "alive": alive,
            "last_ok_age_s": age,
            "errors_total": st["errors"],
        }

    jobs_exists = JOBS_FILE.exists()
    overall_ok  = all(d["alive"] for d in daemons.values())

    # C9: modo público sin auth — solo lo mínimo para health-checks.
    if not authorized:
        return {
            "ok": overall_ok,
            "uptime_s": int(now - _START_TIME),
            "jobs_total": total,
        }
    return {
        "ok": overall_ok,
        "uptime_s": int(now - _START_TIME),
        "jobs": {
            "total": total, "pending": n_pend, "awaiting_review": n_review,
            "done": n_done, "error": n_err,
            "with_image_in_ram": with_img,
            "oldest_age_s": int(now - oldest) if total else 0,
        },
        "daemons": daemons,
        "config": {
            "providers_with_primary": sum(1 for k in ("ANTHROPIC_API_KEY","OPENAI_API_KEY","GEMINI_API_KEY")
                                          if DYNAMIC_CONFIG.get(k, "").strip()),
            "providers_with_backup":  sum(1 for k in ("ANTHROPIC_API_KEY_BACKUP","OPENAI_API_KEY_BACKUP","GEMINI_API_KEY_BACKUP")
                                          if DYNAMIC_CONFIG.get(k, "").strip()),
        },
        "db_backends": {
            "supabase": {
                "enabled": True,
                "last_ok_age_s": int(now - _SUPABASE_LAST_OK) if _SUPABASE_LAST_OK else -1,
            },
            "appwrite": {
                "enabled": _appwrite_enabled(),
                "last_ok_age_s": int(now - _APPWRITE_LAST_OK) if _APPWRITE_LAST_OK else -1,
                "endpoint":  APPWRITE_ENDPOINT or None,
                "database":  APPWRITE_DATABASE_ID or None,
                "collection": APPWRITE_COLLECTION_ID or None,
            },
        },
        "jobs_file": {
            "path": str(JOBS_FILE), "exists": jobs_exists,
            "mtime": int(JOBS_FILE.stat().st_mtime) if jobs_exists else 0,
            "size_bytes": JOBS_FILE.stat().st_size if jobs_exists else 0,
        },
        "limits": {
            "JOB_TTL": JOB_TTL, "JOBS_MAX": JOBS_MAX,
            "IMG_PRUNE_AFTER": IMG_PRUNE_AFTER, "STUCK_TIMEOUT": STUCK_TIMEOUT,
            "AI_HARD_TIMEOUT": AI_HARD_TIMEOUT,
        },
        "threads_alive": threading.active_count(),
        "errors": {
            "in_log": len(_ERROR_LOG),
            "last_at": _ERROR_LOG[-1]["t"] if _ERROR_LOG else 0,
        },
    }


@app.get("/api/errors")
def api_errors(key: str = "", limit: int = 50):
    """Devuelve los últimos N errores capturados (en RAM + disco)."""
    if not _is_authorized_key(key):
        raise HTTPException(status_code=401, detail="Acceso denegado")
    with _ERROR_LOG_LOCK:
        data = list(_ERROR_LOG)
    limit = max(1, min(500, limit))
    return {"count": len(data), "errors": data[-limit:][::-1]}  # más recientes primero


@app.delete("/api/errors")
def clear_errors(x_api_key: Optional[str] = Header(None)):
    check_editor_auth(x_api_key)
    with _ERROR_LOG_LOCK:
        prev = len(_ERROR_LOG)
        _ERROR_LOG.clear()
    try:
        if ERROR_LOG_FILE.exists(): ERROR_LOG_FILE.unlink()
    except Exception: pass
    # Limpiar también las DBs (best-effort) para que un cold start no recargue
    # los errores que el operador acaba de borrar.
    try: _supabase_save_errors()  # con _ERROR_LOG vacío sube lista vacía
    except Exception: pass
    try: _appwrite_save_errors()
    except Exception: pass
    return {"ok": True, "cleared": prev}


@app.get("/api/config/export")
def export_config(x_api_key: Optional[str] = Header(None), key: str = ""):
    """Descarga JSON con TODA la config actual (keys, backups, modelos).
    Pensado para que el usuario lo guarde localmente y lo pueda re-importar
    tras un cold start en FS efímero."""
    auth_key = x_api_key or key
    if not _is_authorized_key(auth_key):
        raise HTTPException(status_code=401, detail="API key inválida")
    with CONFIG_LOCK:
        snapshot = {k: DYNAMIC_CONFIG.get(k, "") for k in _CONFIG_KEYS_PERSISTED}
    snapshot["_exported_at"] = int(time.time())
    body = json.dumps(snapshot, indent=2, ensure_ascii=False)
    filename = f"relay_config_{time.strftime('%Y%m%d_%H%M%S')}.json"
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/config/import")
def import_config(payload: dict, x_api_key: Optional[str] = Header(None)):
    """Importa un JSON exportado previamente. Solo aplica las claves conocidas."""
    check_editor_auth(x_api_key)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Cuerpo debe ser un objeto JSON")
    applied = []
    with CONFIG_LOCK:
        for k in _CONFIG_KEYS_PERSISTED:
            if k not in payload:
                continue
            raw = payload[k]
            if k in _WEIGHT_KEYS:
                # Pesos: int 0..10; aceptamos int/float/string castable
                try:
                    DYNAMIC_CONFIG[k] = max(0, min(10, int(raw)))
                    applied.append(k)
                except (TypeError, ValueError):
                    pass
            elif isinstance(raw, str):
                DYNAMIC_CONFIG[k] = raw.strip()
                applied.append(k)
    persisted = _save_config_to_file() if applied else True
    logger.info("📥 Config importada: %d claves · persistido=%s", len(applied), persisted)
    return {"ok": True, "applied": applied, "persisted": persisted}

@app.post("/reset")
def reset_context(x_api_key: Optional[str] = Header(None)):
    """Llamado por el móvil al pulsar 'Iniciar Sistema'. Borra TODO el estado
    anterior (jobs, historial, correcciones) pero NO toca la config (API keys).
    También borra el JSON persistido para que un cold-start posterior empiece
    desde cero.

    C10: cancela los timers de auto_aprobar pendientes. Sin esto, los Timer
    de threading siguen disparando tras el reset; auto_aprobar es idempotente
    (encuentra `JOBS.get(jid)=None` y retorna), pero gasta slot del scheduler.
    Mejor cancelarlos de plano."""
    check_auth(x_api_key)
    with LOCK:
        prev_count = len(JOBS)
        purged_ids = list(JOBS.keys())
        JOBS.clear()
    with CORRECTIONS_LOCK:
        CORRECTIONS.clear()
    # Limpiar el contexto de imagen sticky: un examen nuevo empieza sin diagrama
    # de caso práctico arrastrado del anterior.
    _clear_sticky_context()
    # C10: cancelar todos los timers tras el clear (fuera del LOCK por si la
    # cancelación toca scheduler interno de Python).
    cancelled = 0
    for jid in purged_ids:
        if _cancel_auto_approve_timer(jid):
            cancelled += 1
    # Por si quedó algún timer huérfano de jobs ya purgados que no estaban en
    # JOBS pero sí en _AUTO_APPROVE_TIMERS:
    with _AUTO_APPROVE_LOCK:
        leftovers = list(_AUTO_APPROVE_TIMERS.items())
        _AUTO_APPROVE_TIMERS.clear()
    for jid, t in leftovers:
        try: t.cancel()
        except Exception: pass
        cancelled += 1
    # Borrar también el snapshot a disco → tras reset, restart = vacío
    for fp in (JOBS_FILE, JOBS_FILE.with_suffix(JOBS_FILE.suffix + ".bak"),
               CORRECTIONS_FILE):
        try:
            if fp.exists(): fp.unlink()
        except Exception as exc:
            logger.warning("No se pudo borrar %s: %s", fp, exc)
    # Vaciar también la versión slim en las DBs (best-effort) para que un
    # restart no recargue jobs viejos desde Supabase/Appwrite.
    try: _supabase_save_jobs()
    except Exception: pass
    try: _appwrite_save_jobs()
    except Exception: pass
    logger.info("🔄 /reset → %d jobs limpiados · %d timers cancelados · config preservada",
                prev_count, cancelled)
    return {"ok": True, "cleared_jobs": prev_count, "cancelled_timers": cancelled}


@app.get("/api/context_status")
def api_context_status(key: str = ""):
    """Estado del contexto de imagen STICKY server-side, para el banner del panel.
    Indica si hay un diagrama de caso práctico vigente que el relay está
    adjuntando a CADA /ask nuevo. NO devuelve el blob (el panel solo necesita
    saber que está activo; la imagen ya se ve en la tarjeta del job)."""
    if not _is_authorized_key(key):
        raise HTTPException(status_code=401, detail="Acceso denegado")
    st = _get_sticky_context()
    if not st:
        return {"active": False}
    b64 = st.get("b64") or ""
    return {
        "active":     True,
        "age_s":      max(0, int(time.time() - st.get("set_at", 0))),
        "size_kb":    len(b64) * 3 // 4 // 1024,
        "source_job": (st.get("source_job") or "")[:8],
        "ttl_s":      STICKY_CONTEXT_TTL_S,
    }


@app.post("/api/context_clear")
def api_context_clear(key: str = "", x_api_key: Optional[str] = Header(None)):
    """Quita la imagen-contexto sticky: las próximas /ask dejan de llevar el
    diagrama hasta que se vuelva a marcar la casilla 🖼️. Lo usa el botón
    'Quitar contexto' del panel cuando termina el caso práctico."""
    if not (_is_authorized_key(key) or _is_authorized_key(x_api_key or "")):
        raise HTTPException(status_code=401, detail="Acceso denegado")
    had = _clear_sticky_context()
    logger.info("🧹 Contexto de imagen sticky %s vía panel",
                "limpiado" if had else "ya estaba vacío")
    return {"ok": True, "was_active": had}


# ─── Fusión OCR server-side (consenso multi-VLM) ──────────────────────────────
# Antes: las N transcripciones OCR se concatenaban tal cual y se mandaban a los
# 5 analyzers como <readings>. El analyzer tenía que dedupear preguntas él
# mismo siguiendo el <protocol>, y a veces fallaba → respondía N×K letras
# (duplicación) o N preguntas por cada lectura. align_answer() y fusionar()
# parcheaban con clamps/outlier-detection, pero la causa raíz era estructural.
#
# Ahora: hacemos consenso EN PYTHON antes del analyzer. Cada analyzer recibe
# UNA transcripción canónica con K preguntas únicas. No hay forma de duplicar.
#
# Algoritmo (basado en Consensus Entropy 2025 + OCROMORE + Calamari):
#   1. Parser regex sobre el formato canónico de OCR_VIDEO_PROMPT
#   2. Cluster fuzzy por similitud de texto de pregunta (difflib, threshold 0.7)
#   3. Por cluster: pickear la versión "más completa" (menos [?], más larga)
#   4. Ordenar clusters por posición mediana entre las OCRs
#   5. Renumerar 1..K, emitir texto plano canónico
#
# Fallback: si el parser/fusión falla por cualquier excepción, el pipeline cae
# al método antiguo (concatenar <readings>) — cero riesgo de regresión.

# Threshold de similitud entre dos textos de pregunta para considerarlos la
# misma pregunta. 0.7 = 70% de caracteres coincidentes (difflib.SequenceMatcher).
# Probado: 0.6 = falsos positivos (preguntas distintas se fusionan).
#          0.8 = falsos negativos (mismas preguntas con OCR distinto se separan).
_OCR_QUESTION_SIM_THRESHOLD = 0.7

# Regex para detectar "N. <texto>" al inicio de un bloque (número + punto/:/) ).
# El OCR_VIDEO_PROMPT pide ese formato literal.
_OCR_QNUM_RE = re.compile(r'^\s*(\d{1,3})\s*[\.\:\)\-]\s*(.+?)\s*$')

# Regex para detectar "A) <texto>", "B) <texto>", etc. al inicio de un bloque.
_OCR_OPT_RE = re.compile(r'^\s*([A-Da-d])\s*[\)\.\:\-]\s*(.*?)\s*$')

# Regex para detectar líneas de sección "## SECCION: <nombre>" emitidas por el
# OCR cuando ve un encabezado de bloque en la hoja (BLOQUE IV, Preguntas de
# reserva, etc.). Tolerante: acepta variantes con/sin tilde, mayúsculas, ##/#,
# y separador : o -. Captura el nombre del bloque.
_OCR_SECTION_RE = re.compile(
    r'^\s*#{1,3}\s*SECC?I[OÓ]N\s*[:\-]\s*(.+?)\s*#*\s*$',
    re.IGNORECASE,
)

# Regex FALLBACK para detectar encabezados de bloque que NO siguen el formato
# canónico "## SECCION: …". Se aplica solo si _OCR_SECTION_RE falla. Captura
# líneas tipo "# BLOQUE I Y II", "## PARTE 2", "### TEMA 3", "BLOQUE IV" en
# OCRs sloppy (Gemini truncado, MiMo, PaddleOCR). Sin esto, OCRs rotos
# parsean sus preguntas con section="" mientras el spine principal las tiene
# con sec="bloque_i_y_ii" → el merge estructural por (sec_key, num) NO las
# fusiona y aparecen preguntas FANTASMA en la salida fusionada.
#
# Palabras clave: BLOQUE / PARTE / TEMA / CAPITULO / RESERVA / SECCION / SECCIÓN.
# Aceptamos con/sin "#" prefijo y con/sin separador. La captura es la línea
# entera (limpiada) — la usaremos como nombre de sección consensual.
_OCR_SECTION_FALLBACK_RE = re.compile(
    r'^\s*#{0,3}\s*'
    r'(BLOQUE|PARTE|TEMA|CAP[IÍ]TULO|RESERVA|SECC?I[OÓ]N)'
    r'\b[\s\-:.]*(.*?)\s*#*\s*$',
    re.IGNORECASE,
)


def _parse_ocr_text(text: str) -> list:
    """Parsea una transcripción OCR al formato canónico de OCR_VIDEO_PROMPT.

    Input esperado (formato canónico generado por OCR_VIDEO_PROMPT):
        1. ¿Cuál es la capital de Francia?
        A) Madrid
        B) París
        C) Berlín
        D) Roma

        2. El símbolo químico del [?] es Au.
        A) Plata
        ...

    Devuelve: lista de dicts ``{num, question, options, section, pos}`` donde:
      - num: int (número ORIGINAL impreso en la hoja, no renumerado)
      - question: str (enunciado sin el "N." inicial)
      - options: dict[str, str] con letra (A..D) → texto opción
      - section: str (último encabezado "## SECCION: …" visto antes; "" si ninguno)
      - pos: int (posición ordinal 0..K-1 dentro de ESTE OCR)

    Las líneas "## SECCION: <nombre>" actualizan la sección actual sin emitir
    pregunta. Permite distinguir "Preguntas de reserva / 1" de "BLOQUE IV / 1"
    en la fusión (con misma N pero pertenecen a bloques distintos).

    NUNCA lanza: si el formato es raro, simplemente devuelve lista vacía o
    parcial. Cualquier excepción interna se traga (return []).
    """
    if not text:
        return []
    try:
        questions: list = []
        current_section: str = ""
        # Normalizar saltos de línea Windows/Mac.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Split por línea en blanco — cada bloque debe ser una pregunta o un
        # marcador de sección (o ruido).
        raw_blocks = re.split(r'\n\s*\n+', text.strip())

        # ─── PRE-PASS: re-anclar a su pregunta cualquier bloque que NO sea
        # un arranque legítimo (sección "## SECCION:" o pregunta REAL con
        # enunciado sustancial). Motivo: con OCR sloppy aparecen líneas en
        # blanco dentro de una misma pregunta — sea entre el enunciado y las
        # opciones, sea partiendo código XSD/HTML/CSS multilínea — que dejan
        # fragmentos huérfanos como bloques sueltos. Sin re-anclar:
        #   1) "10.1.0.0" / "802.11r" / "100 - Mbps" matchean \d{1,3}[\.\:\)\-]…
        #      y se cuelan como preguntas fantasma (bug reportado: páginas con
        #      5 preguntas se contaban como 9).
        #   2) Opciones "A) …" / "B) …" quedan separadas de su enunciado y se
        #      pierden, dejando la pregunta sin alternativas.
        #
        # "Pregunta real" = matchea _OCR_QNUM_RE Y el texto capturado tiene
        # ≥ 5 letras (alpha). Eso descarta ruido tipo "10.1.0.0" → captura
        # texto "1.0.0" con 0 letras → fusiona; pero conserva "3. El
        # identificador..." → 20+ letras → arranque legítimo.
        def _is_real_question_start(first_line: str) -> bool:
            mq = _OCR_QNUM_RE.match(first_line)
            if not mq:
                return False
            captured = mq.group(2).strip()
            return sum(1 for c in captured if c.isalpha()) >= 5

        def _match_section_any(first_line: str):
            """Devuelve (matched, section_name) — intenta primero el formato
            canónico (## SECCION: …) y, si falla, el fallback flexible que
            acepta encabezados tipo "# BLOQUE I Y II", "## PARTE 2",
            "BLOQUE IV" (sin la palabra SECCION). Esto cubre OCRs sloppy
            (Gemini truncado, MiMo, PaddleOCR) que NO siguen el formato
            canónico del OCR_VIDEO_PROMPT. Sin esto, sus preguntas se
            parseaban con section="" y NO se fusionaban con las del spine
            que sí tenía section="BLOQUE I Y II" → aparecían como preguntas
            FANTASMA en la salida."""
            m = _OCR_SECTION_RE.match(first_line)
            if m:
                return True, m.group(1).strip()
            mf = _OCR_SECTION_FALLBACK_RE.match(first_line)
            if mf:
                # Solo aceptamos el fallback si la línea es CORTA (típico
                # encabezado, no una pregunta que empiece con "TEMA de…").
                # Una pregunta real raramente empieza con BLOQUE/PARTE/TEMA
                # solos y suele tener >60 chars.
                if len(first_line) <= 60:
                    kw = mf.group(1).upper()
                    rest = mf.group(2).strip()
                    name = (f"{kw} {rest}".strip() if rest else kw).strip()
                    return True, name
            return False, ""

        blocks: list = []
        for block in raw_blocks:
            first_line = block.split("\n", 1)[0].strip()
            if not first_line:
                continue
            is_section, _ = _match_section_any(first_line)
            is_real_q  = _is_real_question_start(first_line)
            # Arranque legítimo de bloque → nuevo bloque.
            # Cualquier otra cosa con bloque previo → fusionar (re-anclar a
            # la pregunta anterior).
            if blocks and not is_section and not is_real_q:
                blocks[-1] = blocks[-1] + "\n" + block
            else:
                blocks.append(block)

        for block in blocks:
            lines = [ln for ln in (b.strip() for b in block.split("\n")) if ln]
            if not lines:
                continue
            # ¿Es este bloque (o su primera línea) un marcador de sección?
            # Aceptamos secciones bien en su propio bloque, bien al inicio de
            # un bloque que también contiene una pregunta. Usamos el matcher
            # extendido que también captura "# BLOQUE I Y II" (sin la palabra
            # "SECCION") emitido por OCRs sloppy.
            is_section, sec_name = _match_section_any(lines[0])
            if is_section:
                current_section = sec_name
                # Si el bloque solo era la sección, pasar al siguiente.
                if len(lines) == 1:
                    continue
                lines = lines[1:]
            # Primera línea (restante) debe ser "N. <pregunta>". Si no, saltamos.
            m_num = _OCR_QNUM_RE.match(lines[0])
            if not m_num:
                continue
            try:
                num = int(m_num.group(1))
            except (TypeError, ValueError):
                continue
            question = m_num.group(2).strip()
            # Si la pregunta continúa en líneas siguientes ANTES de que aparezca
            # "A)", concatenarlas (preguntas multi-línea son comunes en exámenes).
            options: dict = {}
            i = 1
            while i < len(lines):
                m_opt = _OCR_OPT_RE.match(lines[i])
                if m_opt:
                    break
                # Sigue siendo enunciado
                question = (question + " " + lines[i]).strip()
                i += 1
            # A partir de la primera "A)..." parsear opciones
            current_letter = None
            current_text   = ""
            while i < len(lines):
                m_opt = _OCR_OPT_RE.match(lines[i])
                if m_opt:
                    # Guardar opción anterior si la había
                    if current_letter:
                        options[current_letter] = current_text.strip()
                    current_letter = m_opt.group(1).upper()
                    current_text   = m_opt.group(2).strip()
                else:
                    # Continuación de la opción anterior (opción multi-línea)
                    if current_letter:
                        current_text = (current_text + " " + lines[i]).strip()
                i += 1
            # Cerrar la última opción
            if current_letter:
                options[current_letter] = current_text.strip()
            # ─── Validación: descartar "preguntas fantasma" del regex laxo ───
            # _OCR_QNUM_RE matchea cualquier "N. <texto>" con N de 1-3 dígitos.
            # Eso captura ruido como "10.1.0.0" (Q10 texto "1.0.0"),
            # "802.11r" (Q802 texto "11r"), "100 - Mbps" (Q100 texto "Mbps"),
            # etc., que son trozos de enunciados u opciones, NO preguntas.
            # Una pregunta de verdad cumple AL MENOS UNA de estas dos:
            #   (a) tiene ≥ 1 opción A/B/C/D detectada, o
            #   (b) tiene un enunciado con ≥ 10 caracteres y ≥ 5 letras
            #       (descarta números/IPs/códigos sueltos).
            if not question:
                continue
            alpha_count = sum(
                1 for c in question
                if c.isalpha()
            )
            has_options    = len(options) >= 1
            substantial_q  = len(question) >= 10 and alpha_count >= 5
            if not (has_options or substantial_q):
                logger.debug(
                    "_parse_ocr_text: descartado falso positivo "
                    "num=%s q=%r opts=%s alpha=%d len=%d",
                    num, question[:40], list(options.keys()), alpha_count, len(question),
                )
                continue
            questions.append({
                "num":      num,
                "question": question,
                "options":  options,
                "section":  current_section,
                "pos":      len(questions),
            })
        return questions
    except Exception as exc:
        # ERROR (no warning): el parser fallando es crítico — sin él la fusión
        # cae al modo <readings> crudo que produce peor calidad y puede generar
        # respuestas duplicadas (bug "13 preguntas en lugar de 9").
        logger.error("_parse_ocr_text: parser falló (text len=%d): %s",
                     len(text) if text else 0, exc)
        return []


def _section_key(s: str) -> str:
    """Normaliza el nombre de sección para comparación entre OCRs.
    Quita tildes, mayúsculas, puntuación, espacios extra. Así "BLOQUE IV" y
    "bloque iv." y "Bloque-IV" cuentan como la misma sección."""
    if not s:
        return ""
    s = s.lower()
    # Quitar tildes básicas
    s = (s.replace("á","a").replace("é","e").replace("í","i")
           .replace("ó","o").replace("ú","u").replace("ñ","n"))
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    return s


def _normalize_for_match(s: str) -> str:
    """Normaliza un texto para comparación fuzzy: lowercase, sin marcas [?],
    espacios colapsados. NO modifica el texto original que se emitirá al final."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r'\[[\?\.]+\]', ' ', s)        # quitar [?], [...?], etc.
    s = re.sub(r'[¿\?¡!\.,;:\(\)\[\]"\'`]', ' ', s)  # puntuación
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _similarity(a: str, b: str) -> float:
    """Similitud léxica multi-señal entre dos textos (0..1).

    Combina varias métricas con MAX:
      • difflib SequenceMatcher (Ratcliff-Obershelp sobre caracteres): bueno
        para typos y cambios pequeños. Sensible al ORDEN de palabras.
      • rapidfuzz.token_set_ratio: divide en palabras, compara intersección y
        diferencias. Robusto a orden alterado y palabras duplicadas/omitidas.
      • rapidfuzz.token_sort_ratio: ordena palabras alfabéticamente antes de
        comparar. Captura "misma frase con palabras movidas".
      • rapidfuzz.partial_ratio: encuentra la mejor subcadena alineada. Bueno
        cuando un OCR transcribió solo media frase.
      • Jaccard de palabras: |A∩B| / |A∪B|. Fallback léxico clásico.

    Tomamos el MAX porque dos frases pueden parecerse fuerte en UNA métrica y
    poco en otras (ej. "POP3 puerto 110" vs "El puerto del POP3 es 110": orden
    distinto + palabras extra → difflib bajo pero token_set alto). Usar MAX
    evita falsos negativos sin coste extra: ya pagamos por calcular todas.

    Defensivo: None, vacío y excepciones → 0.0.
    """
    try:
        a_norm = _normalize_for_match(a)
        b_norm = _normalize_for_match(b)
        if not a_norm or not b_norm:
            return 0.0
        if a_norm == b_norm:
            return 1.0

        # 1) Difflib (caracteres, Ratcliff-Obershelp)
        from difflib import SequenceMatcher
        sim_diff = SequenceMatcher(None, a_norm, b_norm).ratio()

        # 2) RapidFuzz (palabras, robusto a orden). Import lazy: si no está
        # instalado por algún motivo (ej. python embebido), caemos a difflib
        # como antes sin romper.
        sim_tset = sim_tsort = sim_partial = 0.0
        try:
            from rapidfuzz import fuzz as _rf
            sim_tset    = _rf.token_set_ratio(a_norm, b_norm)  / 100.0
            sim_tsort   = _rf.token_sort_ratio(a_norm, b_norm) / 100.0
            sim_partial = _rf.partial_ratio(a_norm, b_norm)    / 100.0
        except ImportError:
            pass

        # 3) Jaccard de palabras (intersección / unión)
        words_a = set(a_norm.split())
        words_b = set(b_norm.split())
        if words_a or words_b:
            inter = len(words_a & words_b)
            union = len(words_a | words_b)
            sim_jacc = inter / union if union else 0.0
        else:
            sim_jacc = 0.0

        return max(sim_diff, sim_tset, sim_tsort, sim_partial, sim_jacc)
    except Exception:
        return 0.0


def _completeness_score(text: str) -> tuple:
    """Score para elegir la "mejor" versión de un texto entre varias OCRs.
    Mayor es mejor. Penaliza marcadores [?] y prefiere textos más largos.
    Devuelve tupla para tie-breaking estable."""
    if not text:
        return (-1000, 0, 0)
    n_questionmarks = text.count("[?]") + text.count("[...?]") + text.count("[…?]")
    length = len(text.strip())
    # Score = -penalización_por_[?] + longitud + bonus si tiene puntuación real
    has_real_punct = 1 if re.search(r'[¿\?¡!\.,;:]', text) else 0
    return (-n_questionmarks, length, has_real_punct)


# ─── Needleman-Wunsch alignment (1970) ────────────────────────────────────────
# Algoritmo clásico de alineación de secuencias usado en bioinformática.
# Encuentra la alineación ÓPTIMA entre dos listas ordenadas, permitiendo
# "gaps" (saltos) cuando una secuencia tiene elementos que la otra no.
#
# Por qué aquí: 2 OCRs leyendo el mismo examen pueden ver número distinto de
# preguntas (una se salta una, otra alucina una de más). NW alinea elementos
# que sí coinciden y deja "huecos" donde una OCR se perdió/inventó algo.
#
# Scoring (revisado tras analizar fallos reales de OCR muy alucinógenos):
#   • Señal PRIMARIA: (sección, num) — si ambos OCRs leen "BLOQUE IV / 1",
#     es casi seguro la misma pregunta aunque hayan transcrito enunciados
#     completamente distintos (cosa que pasa cuando la cámara está mal). Esto
#     evita que la fusión multiplique preguntas (3 OCRs × 9 hallucinaciones =
#     27 clusters de basura).
#   • Señal SECUNDARIA: similitud de texto (difflib). Sirve cuando el OCR
#     no consigue extraer sección o número fiable.
#   • Señal TERCIARIA: posición ordinal (bonus pequeño para empates).
#   • Penalización fuerte si las secciones difieren con num igual ("Reserva 1"
#     ≠ "BLOQUE IV 1") — fuerza gap.

_NW_MATCH_BASELINE = 0.25
_NW_GAP_PENALTY   = -0.05
_NW_MIN_KEEP_SIM  = 0.20    # tras backtrack, sim<0.20 se divide en dos gaps


def _nw_pair_score(qa: dict, qb: dict, sim: float) -> float:
    """Score de match entre dos preguntas (qa de OCR_A, qb de OCR_B) para NW.
    Combina similitud textual con señales fuertes de (sección, num).

    Devuelve un valor en torno a [-0.5, +0.7]. Comparado con gap_penalty (-0.05),
    valores >0 favorecen match; valores <0 favorecen gap."""
    sec_a = _section_key(qa.get("section") or "")
    sec_b = _section_key(qb.get("section") or "")
    num_a = qa.get("num")
    num_b = qb.get("num")
    both_have_section = bool(sec_a) and bool(sec_b)
    same_section = both_have_section and sec_a == sec_b
    same_num = (num_a is not None and num_b is not None and num_a == num_b)

    # Caso 1: ambos OCRs vieron sección Y ambos vieron num → señal fuerte
    if both_have_section:
        if same_section and same_num:
            # Misma sección + mismo num → casi seguro misma pregunta.
            # Forzamos match aunque la similitud sea baja: los OCRs pueden
            # haber leído contenidos divergentes por mala visión.
            return max(sim - _NW_MATCH_BASELINE, 0.55)
        if (not same_section) and same_num:
            # Mismo num pero secciones distintas → casi seguro DIFERENTES
            # preguntas (p.ej. "Reserva 1" vs "BLOQUE IV 1"). Penaliza fuerte.
            return (sim - _NW_MATCH_BASELINE) - 0.50
        if same_section and not same_num:
            # Misma sección, num distinto → probablemente preguntas distintas
            # dentro del mismo bloque. Penalización suave.
            return (sim - _NW_MATCH_BASELINE) - 0.15
        # Distinta sección, distinto num → distintas preguntas
        return (sim - _NW_MATCH_BASELINE) - 0.30

    # Caso 2: al menos un OCR no tiene sección — usamos num como ancla primaria.
    # El nuevo prompt pide preservar el número ORIGINAL de la hoja, así que si
    # ambos OCRs ven "2." casi seguro están leyendo la misma pregunta nº 2 —
    # aunque hayan transcrito el enunciado de forma totalmente distinta. La
    # consigna del usuario es CLARA: en ese caso hay que COMBINAR las dos
    # lecturas en UNA pregunta (no emitir dos preguntas).
    score = sim - _NW_MATCH_BASELINE
    if same_num:
        # Match forzado por num (señal estructural primaria).
        return max(score, 0.50)
    if num_a is not None and num_b is not None and num_a != num_b:
        # Ambos OCRs leyeron num explícito y son distintos → probable que sean
        # preguntas distintas. Penalización moderada para favorecer gap.
        return score - 0.20
    # Falta num en alguno: solo texto + bonus suave de posición ordinal.
    if qa.get("pos") is not None and qa.get("pos") == qb.get("pos"):
        score += 0.05
    return score


def _nw_align(seq_a: list, seq_b: list, sim_fn=None) -> list:
    """Needleman-Wunsch alignment de dos listas de preguntas.

    Args:
      seq_a, seq_b: listas de dicts (cada uno con campo 'question', y
                    opcionalmente 'pos' y 'num').
      sim_fn:       función (q_a, q_b) → similarity ∈ [0..1]. Si None usa
                    _similarity sobre el campo 'question'.

    Returns:
      Lista de pares (i_a, i_b) en orden, donde None = "gap" (la otra
      secuencia tiene un elemento que esta no). Ejemplos:
        [(0,0), (1,1), (2,None), (3,2)]  → A tiene 4, B tiene 3, B se saltó A[2]
        [(None,0), (0,1), (1,2)]         → B tiene 3, A tiene 2, A se saltó B[0]
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return [(None, j) for j in range(m)]
    if m == 0:
        return [(i, None) for i in range(n)]

    if sim_fn is None:
        def sim_fn(qa, qb):
            return _similarity(qa.get("question", ""), qb.get("question", ""))

    # Cache de similitudes (evita recomputar — cada par se llama 2+ veces)
    sim_cache: dict = {}
    def _sim_cached(i, j):
        if (i, j) not in sim_cache:
            sim_cache[(i, j)] = sim_fn(seq_a[i], seq_b[j])
        return sim_cache[(i, j)]

    # DP[i][j] = score máximo de alinear seq_a[:i] con seq_b[:j]
    # trace[i][j] = 0 diag (match/mismatch), 1 up (gap en b), 2 left (gap en a)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * _NW_GAP_PENALTY
        trace[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j * _NW_GAP_PENALTY
        trace[0][j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = _sim_cached(i - 1, j - 1)
            qa = seq_a[i - 1]
            qb = seq_b[j - 1]
            match_score = _nw_pair_score(qa, qb, sim)
            choices = (
                dp[i-1][j-1] + match_score,  # 0: match/mismatch
                dp[i-1][j]   + _NW_GAP_PENALTY,  # 1: gap en b (skip a[i])
                dp[i][j-1]   + _NW_GAP_PENALTY,  # 2: gap en a (skip b[j])
            )
            best_k = 0
            best_v = choices[0]
            for k in (1, 2):
                if choices[k] > best_v:
                    best_v = choices[k]
                    best_k = k
            dp[i][j] = best_v
            trace[i][j] = best_k

    # Backtrack
    pairs: list = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 0:
            sim = _sim_cached(i - 1, j - 1)
            qa = seq_a[i - 1]
            qb = seq_b[j - 1]
            # Mantener el match si la similitud textual es razonable, O si
            # (sección + num) coinciden exactamente. Esto último es la señal
            # que más nos importa: dos OCRs viendo "BLOQUE IV / 1" SON la
            # misma pregunta aunque hayan transcrito textos muy distintos.
            sec_a = _section_key(qa.get("section") or "")
            sec_b = _section_key(qb.get("section") or "")
            structural_match = (
                bool(sec_a) and sec_a == sec_b and
                qa.get("num") is not None and qa.get("num") == qb.get("num")
            )
            if sim >= _NW_MIN_KEEP_SIM or structural_match:
                pairs.append((i - 1, j - 1))
            else:
                # Match forzado por DP pero ni texto ni estructura coinciden →
                # tratamos como gap doble (cada uno standalone)
                pairs.append((i - 1, None))
                pairs.append((None, j - 1))
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or trace[i][j] == 1):
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return list(reversed(pairs))


def _build_clusters_multi_ocr(parsed_per_ocr: dict) -> list:
    """Construye clusters multi-OCR de forma INCREMENTAL con Needleman-Wunsch.

    Algoritmo:
      1. Toma la primera OCR como "spine" (esqueleto): cada pregunta = un cluster.
      2. Para cada OCR siguiente, alinea con el spine usando NW.
      3. Cada par alineado → la pregunta se añade al cluster del spine.
      4. Cada gap en B (spine tiene, B no) → cluster intacto.
      5. Cada gap en A (B tiene una que el spine no) → nuevo cluster al spine.
      6. Cap final: si tras añadir todos los OCRs hay MÁS clusters que el OCR
         que más preguntas vio, podamos los menos respaldados — la fusión
         nunca debe emitir más preguntas que cualquier OCR individual.

    El orden de OCRs en parsed_per_ocr afecta sutilmente el resultado (la
    primera es el "ancla"). Ordenamos por nº preguntas detectadas (más → primero)
    para que el spine sea lo más completo posible desde el inicio.

    Returns: lista de clusters [list[(provider, q_dict)], ...]
    """
    if not parsed_per_ocr:
        return []

    # Ordenar OCRs por nº de preguntas detectadas (desc) para que el ancla
    # sea la más informativa. Empate → orden insertion (estable).
    providers_sorted = sorted(
        parsed_per_ocr.keys(),
        key=lambda p: -len(parsed_per_ocr[p])
    )

    # Spine inicial = preguntas del primer OCR, cada una en su cluster
    first = providers_sorted[0]
    spine: list = [[(first, q)] for q in parsed_per_ocr[first]]

    for prov_b in providers_sorted[1:]:
        qs_b = parsed_per_ocr[prov_b]
        if not qs_b:
            continue
        # Representante de cada cluster = la primera pregunta añadida.
        # Es el ancla con el que comparamos los nuevos.
        seq_a = [cluster[0][1] for cluster in spine]
        pairs = _nw_align(seq_a, qs_b)

        new_spine: list = []
        for i_a, j_b in pairs:
            if i_a is not None and j_b is not None:
                # Match: añadir qs_b[j_b] al cluster existente
                cluster = spine[i_a]
                existing_provs = {p for p, _ in cluster}
                if prov_b not in existing_provs:
                    cluster.append((prov_b, qs_b[j_b]))
                new_spine.append(cluster)
            elif i_a is not None:
                # Gap en B: el cluster del spine se queda igual
                new_spine.append(spine[i_a])
            elif j_b is not None:
                # Gap en A: pregunta nueva que el spine no tenía → nuevo cluster
                new_spine.append([(prov_b, qs_b[j_b])])
        spine = new_spine

    # ──── Pase de merge estructural: fusionar clusters con misma (sec, num) ────
    # Red de seguridad por si NW dejó separadas dos preguntas que SÍ son la
    # misma (mismo número en la misma sección). Esto materializa la consigna:
    # "si 2 OCRs leen num=2 deben combinarse en UNA pregunta, no aparecer dos
    # veces". Procedemos con la moda de (sec_key, num) consensuada por cluster
    # como clave de agrupación; clusters sin num quedan separados.
    from collections import Counter as _StructCounter
    def _cluster_struct_key(cluster):
        sec_votes = _StructCounter(_section_key(q.get("section") or "") for _, q in cluster)
        num_votes = _StructCounter(q.get("num") for _, q in cluster if q.get("num") is not None)
        if not num_votes:
            return None  # sin num → no agrupable estructuralmente
        sec_key = sec_votes.most_common(1)[0][0] if sec_votes else ""
        num_key = num_votes.most_common(1)[0][0]
        return (sec_key, num_key)

    merged_by_key: dict = {}
    spine_after_merge: list = []
    for cluster in spine:
        key = _cluster_struct_key(cluster)
        if key is None:
            # Sin num consensuado: no se puede agrupar estructuralmente
            spine_after_merge.append(cluster)
            continue
        if key in merged_by_key:
            # Ya existe un cluster con esta (sec, num) → fusionar.
            target_cluster = merged_by_key[key]
            existing_provs = {p for p, _ in target_cluster}
            for prov, q in cluster:
                if prov not in existing_provs:
                    target_cluster.append((prov, q))
                    existing_provs.add(prov)
        else:
            merged_by_key[key] = cluster
            spine_after_merge.append(cluster)

    # ──── Segundo pase: absorber clusters con sec_key="" en clusters con
    # misma `num` y sec_key no vacío. Motivo: un OCR sloppy puede haber
    # parseado preguntas con section="" (porque su encabezado "# BLOQUE I"
    # no matchea el regex canónico, aunque el fallback ya lo cubre, este
    # pase es defensa adicional). Las versiones de la misma pregunta no
    # deben quedar como clusters separados sólo porque a un OCR le faltó
    # detectar la sección. Si la num coincide, las fusionamos al cluster
    # con sección — eso es la pregunta REAL del examen, no una fantasma.
    by_num_with_sec: dict = {}   # num → cluster del primer match con sec_key != ""
    for cluster in spine_after_merge:
        key = _cluster_struct_key(cluster)
        if key is None:
            continue
        sec_k, num_k = key
        if sec_k and num_k not in by_num_with_sec:
            by_num_with_sec[num_k] = cluster

    spine_after_orphan_absorb: list = []
    absorbed_count = 0
    for cluster in spine_after_merge:
        key = _cluster_struct_key(cluster)
        if key is not None:
            sec_k, num_k = key
            if not sec_k and num_k in by_num_with_sec:
                # Cluster huérfano (sin sección) con un num que YA existe en
                # un cluster con sección → absorber en el cluster con sección.
                target = by_num_with_sec[num_k]
                existing_provs = {p for p, _ in target}
                for prov, q in cluster:
                    if prov not in existing_provs:
                        target.append((prov, q))
                        existing_provs.add(prov)
                absorbed_count += 1
                continue
        spine_after_orphan_absorb.append(cluster)

    if absorbed_count > 0:
        logger.info(
            "Absorción de clusters huérfanos (sec=\"\" → cluster con sección y misma num): %d absorbidos",
            absorbed_count,
        )

    if len(spine_after_orphan_absorb) != len(spine):
        logger.info(
            "Merge estructural por (sec, num) + absorción huérfanos: %d → %d clusters",
            len(spine), len(spine_after_orphan_absorb),
        )
    spine = spine_after_orphan_absorb

    # ──── Filtro de CONSENSO DE VÍDEO: descartar singletons ───────────────────
    # Todas las IAs de VÍDEO procesan el MISMO clip completo, así que una
    # pregunta REAL la leen ≥2 de ellas. Una pregunta vista por UN SOLO OCR de
    # vídeo (y por ningún otro de vídeo) es una alucinación de ese modelo → se
    # descarta, AUNQUE un OCR page-based la haya visto: un page-based ve un
    # frame parcial/recortado y no puede "rescatar" una alucinación de vídeo.
    #
    # Regla (decidida por el usuario): conservar un cluster SII lo corroboran
    # ≥2 OCR de VÍDEO distintos. Los page-based siguen VOTANDO texto/numeración
    # dentro de los clusters que pasan, pero su presencia no mantiene vivo un
    # cluster por sí sola.
    #
    # Fallbacks (para no vaciar la fusión en configuraciones degeneradas):
    #   • <2 OCR de vídeo con texto → exigir ≥2 OCR cualesquiera (vídeo o page).
    #   • <2 OCR en total → no hay consenso posible → conservar todo.
    #   • Si el filtro dejaría 0 clusters (los OCR de vídeo no coinciden en
    #     NINGUNA pregunta) → NO se aplica; lo resuelve el cap por conteo de
    #     abajo. Emitir 0 preguntas tiraría toda la fusión a <readings> crudo.
    n_total_ocrs = len(parsed_per_ocr)
    n_video_ocrs = sum(1 for p in parsed_per_ocr if p in _VIDEO_OCR_NAMES)
    if n_total_ocrs >= 2:
        if n_video_ocrs >= 2:
            _min_video, _min_any, _rule = 2, 0, "≥2 OCR de vídeo"
        else:
            _min_video, _min_any, _rule = 0, 2, "≥2 OCR cualquiera (fallback <2 vídeo)"

        def _passes_consensus(cluster) -> bool:
            provs = {p for p, _ in cluster}
            n_video = sum(1 for p in provs if p in _VIDEO_OCR_NAMES)
            return n_video >= _min_video and len(provs) >= _min_any

        kept: list = []
        dropped: list = []
        for c in spine:
            (kept if _passes_consensus(c) else dropped).append(c)

        if not kept:
            logger.warning(
                "Consenso vídeo (%s): el filtro dejaría 0 de %d clusters "
                "(los OCR no coinciden en ninguna pregunta) — NO se aplica; "
                "lo resolverá el cap por conteo.",
                _rule, len(spine),
            )
        elif dropped:
            for c in dropped:
                nv = _StructCounter(q.get("num") for _, q in c
                                    if q.get("num") is not None)
                logger.info(
                    "Consenso vídeo: descarto singleton num=%s vídeo=%d total=%d "
                    "providers=%s",
                    (nv.most_common(1)[0][0] if nv else "?"),
                    sum(1 for p, _ in c if p in _VIDEO_OCR_NAMES),
                    len(c), [p for p, _ in c],
                )
            logger.info(
                "Consenso vídeo (%s): %d → %d clusters (%d singleton(s) descartado(s))",
                _rule, len(spine), len(kept), len(dropped),
            )
            spine = kept

    # ──── Cap: control de proliferación de clusters ───────────────────────────
    # Dos casos a distinguir:
    #
    #   A) Los OCRs COOPERAN: al menos UN cluster tiene 2+ OCRs (alguna pregunta
    #      vista por varias IAs). En ese caso confiamos en la identificación
    #      estructural — cada cluster con (sec, num) único es una pregunta real,
    #      aunque solo un OCR la haya leído (otros pueden haberse perdido esa
    #      parte de la hoja). NO capamos clusters identificados; solo podamos
    #      clusters huérfanos (sin num consensuado) si sobran.
    #
    #   B) Los OCRs no se ponen de acuerdo en NADA (todos los clusters son
    #      singletons, no hay consenso): alucinación pura, cada OCR ve un
    #      examen distinto. En ese caso cap estricto a max(OCR_counts).
    #
    # Esto evita el bug del usuario: "OCR1 vio 1,2,3 y OCR2 vio 3,4 → total 4
    # preguntas únicas, no 3 con la 4ª descartada".
    # FIX BUG "21 ≠ 13": cuatro mejoras combinadas para evitar que un OCR
    # alucinado o un OCR de imagen con frame recortado distorsione el conteo:
    #   1. Cluster priority INCLUYE suma de pesos de OCRs del cluster
    #      (antes solo consensus=len). Singletons de Qwen alucinado pierden
    #      vs singletons de OCRs con más peso.
    #   2. Mediana PONDERADA por peso de OCR en lugar de mediana simple.
    #      Si un OCR con peso alto dice 13 y uno con peso bajo dice 21, el
    #      OCR con MÁS peso tiene mayor influencia en el target.
    #   3. Cap también aplica a `identified` cuando excede target.
    #      Antes solo capábamos `orphan`, así que un OCR alucinando 21
    #      clusters con num=1..21 únicos (todos "identified" porque tienen
    #      num consensuado consigo mismos) pasaba sin cap.
    #   4. SOLO OCRs de VIDEO cuentan para el cap (no page-based).
    #      Los page-based (Mistral OCR / GLM / DeepSeek OCR / Anthropic /
    #      OpenAI) ven UN frame extraído del video. Si la cámara se movió o
    #      el frame mejor estaba CORTADO, ven menos preguntas (8 en lugar
    #      de 13). Si su count contara para el cap, descartaríamos preguntas
    #      reales que SÍ están en el video. Los page-based SIGUEN
    #      contribuyendo al voto del TEXTO/NUM de cada pregunta — su
    #      precisión por-pregunta es alta, solo el CONTEO total no es
    #      fiable cuando ven un crop parcial.
    _video_counts_weighted = [
        (len(qs), _ocr_provider_weight(p), p)
        for p, qs in parsed_per_ocr.items()
        if qs and p in _VIDEO_OCR_NAMES
    ]
    _page_counts_weighted = [
        (len(qs), _ocr_provider_weight(p), p)
        for p, qs in parsed_per_ocr.items()
        if qs and p in _PAGE_OCR_NAMES
    ]

    if len(_video_counts_weighted) >= 2:
        # Caso óptimo: 2+ OCRs de video con texto. Usamos solo esos para el cap.
        ocr_counts_weighted = [(c, w) for c, w, _ in _video_counts_weighted]
    elif len(_video_counts_weighted) == 1:
        # SANITY CHECK con UN ÚNICO OCR de video: si su count es muy distinto
        # del MÁXIMO entre los page-based, probablemente alucinó. Usamos el
        # MIN(video, max_page) como target conservador — preferimos quedarnos
        # cortos (alguna pregunta del page-based crop puede faltar pero al
        # menos no añadimos fantasmas).
        #
        # Si page-based no tienen texto, vamos solo con el video.
        v_count, v_w, v_p = _video_counts_weighted[0]
        if _page_counts_weighted:
            page_max = max(c for c, _, _ in _page_counts_weighted)
            page_median = int(sorted(c for c, _, _ in _page_counts_weighted)[
                len(_page_counts_weighted) // 2])
            # Si el video OCR dice MUCHO más que el máximo page-based (>1.5×),
            # asumimos alucinación del video → cap = page_max (cubre todo lo
            # que los page-based vieron, conservador respecto al video loco).
            if v_count > page_max * 1.5:
                logger.warning(
                    "Cluster cap: video OCR único '%s' dice %d preguntas pero "
                    "page-based máximo es %d (sospecha alucinación) — capando a %d",
                    v_p, v_count, page_max, page_max,
                )
                # Inyectamos un "voto virtual" del page_max con peso del video
                # para que el target sea conservador.
                ocr_counts_weighted = [(c, w) for c, w, _ in _video_counts_weighted]
                ocr_counts_weighted.append((page_max, v_w))
            else:
                # Counts coherentes (video ≤ 1.5× page_max) → confiamos en el video.
                ocr_counts_weighted = [(c, w) for c, w, _ in _video_counts_weighted]
        else:
            ocr_counts_weighted = [(c, w) for c, w, _ in _video_counts_weighted]
    else:
        # NINGÚN OCR de video con texto. Fallback a page-based (mejor que nada).
        ocr_counts_weighted = [(c, w) for c, w, _ in _page_counts_weighted]
        if ocr_counts_weighted:
            logger.warning(
                "Cluster cap: sin OCRs de video con texto — usando page-based "
                "para el cap (puede ser imperfecto si los frames estaban cortados). "
                "Providers que aportan al cap: %s",
                [p for _, _, p in _page_counts_weighted],
            )
    if ocr_counts_weighted:
        identified = [c for c in spine if _cluster_struct_key(c) is not None]
        orphan     = [c for c in spine if _cluster_struct_key(c) is None]
        has_consensus = any(len(c) >= 2 for c in identified)

        def _cluster_priority(cluster):
            consensus = len(cluster)
            # NUEVO: peso TOTAL del cluster = suma de pesos de los OCRs que
            # lo vieron. Un cluster respaldado por Mistral OCR (peso 5) +
            # Claude (3) tiene peso 8; uno respaldado solo por Qwen (4) tiene
            # peso 4. Al ordenar por priority, los clusters con menos peso
            # caen primero al podar.
            weight_sum = sum(_ocr_provider_weight(p) for p, _ in cluster)
            best = max(
                (_completeness_score(q.get("question", "")) for _, q in cluster),
                default=(-1000, 0, 0),
            )
            earliest_pos = min(
                (q.get("pos", 9999) for _, q in cluster),
                default=9999,
            )
            # Orden: 1º consensus, 2º weight_sum, 3º completeness, 4º posición.
            return (consensus, weight_sum, best, -earliest_pos)

        # ALGORITMO ROBUSTO v2: "todas las IAs de video ven el MISMO clip".
        #
        # PRINCIPIO: si los OCRs procesan el MISMO video, su count tiene que ser
        # PARECIDO. Si una diverge mucho del resto, está alucinando. La que
        # alucina NO debe contar — independientemente de su peso individual.
        #
        # Pipeline en 2 pasos:
        #
        #   PASO 1 — REJECT OUTLIERS (sin peso):
        #     Descartar counts fuera del rango [mediana × 0.6, mediana × 1.4].
        #     El peso NO interviene aquí: aunque la IA divergente sea Mistral OCR
        #     (peso 5), si su count es físicamente imposible (todas vieron lo
        #     mismo), se descarta.
        #
        #   PASO 2 — DECIDIR ENTRE INLIERS (con peso):
        #     a) Si ≥2 inliers coinciden EXACTAMENTE → ese count gana.
        #        Empates por suma de pesos, luego menor count (conservador).
        #     b) Si todos los inliers son únicos → mediana ponderada por peso.
        #
        # Casos cubiertos:
        #   • [12, 12, 12, 26]: outlier 26 descartado → consenso 12 ✓
        #   • [13, 13, 26] (3 OCRs, uno aluciena): 26 outlier → consenso 13 ✓
        #   • [12 (w=1), 12 (w=1), 12 (w=1), 26 (w=10)]: 26 outlier → 12 ✓
        #     (consenso vence al peso individual del alucinador)
        #   • [11, 12, 13, 26]: 26 outlier → mediana ponderada {11,12,13} = 12 ✓
        #   • [12, 13]: ambos inliers, sin consenso → mediana ponderada
        def _robust_target(pairs):
            """pairs: list[(count, weight)] de OCRs con texto. Devuelve target int."""
            if not pairs:
                return 0
            if len(pairs) == 1:
                return int(pairs[0][0])

            from collections import Counter as _Cnt
            from statistics import median as _med
            counts_only = [c for c, _ in pairs]
            n = len(pairs)
            median_simple = _med(counts_only)

            # PASO 1: rechazar outliers IGNORANDO peso (rango ±40% de la mediana)
            lo = median_simple * 0.6
            hi = median_simple * 1.4
            inliers  = [(c, w) for c, w in pairs if lo <= c <= hi]
            outliers = [(c, w) for c, w in pairs if not (lo <= c <= hi)]

            # Fallback defensivo: si <2 inliers (caso extraordinario donde la
            # mediana es 0 o todos son outliers respecto a sí mismos), usar pairs
            if len(inliers) < 2:
                inliers = list(pairs)
                outliers = []

            if outliers:
                logger.info(
                    "_robust_target: %d outlier(s) descartados counts=%s "
                    "(mediana=%g, rango=[%g,%g], inliers=%s)",
                    len(outliers), [c for c, _ in outliers],
                    median_simple, lo, hi, [c for c, _ in inliers],
                )

            # PASO 2: decidir entre INLIERS con peso
            inlier_counts = [c for c, _ in inliers]
            inlier_freq = _Cnt(inlier_counts)
            top_freq_in = max(inlier_freq.values())
            tied_in = [c for c, f in inlier_freq.items() if f == top_freq_in]

            def _resolve_tie(candidates: list) -> int:
                """Desempate: máxima suma de pesos, luego menor count (conservador)."""
                if len(candidates) == 1:
                    return int(candidates[0])
                weights_per_count: dict = {c: 0.0 for c in candidates}
                for cc, ww in inliers:
                    if cc in weights_per_count:
                        weights_per_count[cc] += ww
                return int(max(candidates,
                                key=lambda c: (weights_per_count[c], -c)))

            # 2a: si ≥2 inliers coinciden EXACTAMENTE → consenso
            if top_freq_in >= 2:
                chosen = _resolve_tie(tied_in)
                logger.info(
                    "_robust_target: CONSENSO en inliers %d/%d en count=%d "
                    "(total_pairs=%d, outliers=%d)",
                    top_freq_in, len(inliers), chosen, n, len(outliers),
                )
                return chosen

            # 2b: todos los inliers únicos → mediana ponderada
            sorted_inliers = sorted(inliers, key=lambda p: p[0])
            total_w = sum(w for _, w in sorted_inliers)
            if total_w <= 0:
                return int(_med(inlier_counts))
            cumul = 0.0
            chosen_count = sorted_inliers[-1][0]
            for c, w in sorted_inliers:
                cumul += w
                if cumul >= total_w / 2:
                    chosen_count = c
                    break
            logger.info(
                "_robust_target: SIN CONSENSO en inliers, mediana ponderada=%d "
                "(inlier_counts=%s, outliers=%d)",
                int(chosen_count), inlier_counts, len(outliers),
            )
            return int(chosen_count)

        ocr_count_median_w = _robust_target(ocr_counts_weighted)
        ocr_count_max      = max(c for c, _ in ocr_counts_weighted)
        ocr_count_min      = min(c for c, _ in ocr_counts_weighted)

        if has_consensus:
            # Caso A: hay cooperación. Target = mediana ponderada por peso.
            # Si identified > target, PODAMOS también identified (los menos
            # respaldados caen primero por _cluster_priority).
            target = ocr_count_median_w
            # 1) Primero capar identified si excede.
            if len(identified) > target:
                identified_ranked = sorted(identified, key=_cluster_priority,
                                            reverse=True)
                identified_kept = identified_ranked[:target]
                logger.info(
                    "Cluster cap identified: %d → %d (target=%d, "
                    "min=%d, max=%d, mediana_w=%d)",
                    len(identified), len(identified_kept),
                    target, ocr_count_min, ocr_count_max, target,
                )
                identified = identified_kept
            # 2) Después rellenar con huérfanos solo hasta target total.
            target_orphans = max(0, target - len(identified))
            orphan_ranked = sorted(orphan, key=_cluster_priority, reverse=True)
            orphan_kept = orphan_ranked[:target_orphans]
            new_spine = identified + orphan_kept
            if len(new_spine) != len(spine):
                logger.info(
                    "Cluster cap (consenso): %d → %d (identified=%d, "
                    "orphans=%d→%d, target_w=%d, min=%d, max=%d)",
                    len(spine), len(new_spine),
                    len(identified), len(orphan), len(orphan_kept),
                    target, ocr_count_min, ocr_count_max,
                )
            spine = new_spine
        else:
            # Caso B: alucinación total. Cap estricto a la MEDIANA PONDERADA.
            target = ocr_count_median_w
            if len(spine) > target:
                pruned = sorted(spine, key=_cluster_priority, reverse=True)[:target]
                logger.info(
                    "Cluster cap (alucinación): %d → %d (target_w=%d, "
                    "min=%d, max=%d)",
                    len(spine), len(pruned), target,
                    ocr_count_min, ocr_count_max,
                )
                spine = pruned
    return spine


def _vote_text(versions: list, providers: Optional[list] = None) -> tuple:
    """Mayoría sobre N versiones de un texto (enunciado o opción) con WEIGHTED
    voting cuando se pasa la lista de proveedores correspondientes.

    Regla:
      1. Si DOS o más OCRs coinciden tras normalizar mayúsculas/puntuación,
         estamos en modo "vote". Si hay empate de COUNT entre varias normas,
         se rompe por SUMA DE PESOS de los OCRs que apoyan cada norma
         (peso por provider definido en `_OCR_PROVIDER_WEIGHTS`). Entre las
         versiones literales de la norma ganadora, se elige la más COMPLETA
         (menos `[?]`, más larga, con puntuación real).
      2. Si nadie coincide tras normalizar → MEDOID: la versión más central
         por similitud media con las demás, ponderada por peso del provider
         de cada versión comparada (un OCR dedicated lejano cuenta menos
         como "outlier" que uno general lejano).

    `providers` es opcional para no romper callers viejos; si es None,
    todos los pesos son 1.0 (equivalente al comportamiento anterior).

    Returns: tupla (chosen_text, vote_stats) donde vote_stats = {
        "n":        nº de versiones de entrada,
        "majority": nº de OCRs que coinciden con la versión ganadora (>=1),
        "weight":   suma de pesos de los OCRs que apoyan al ganador,
        "method":   "unanimous" | "vote" | "vote_weighted_tiebreak" | "medoid",
    }
    """
    if not versions:
        return ("", {"n": 0, "majority": 0, "weight": 0.0, "method": "empty"})
    if len(versions) == 1:
        return (versions[0], {"n": 1, "majority": 1, "weight": 1.0,
                              "method": "unanimous"})

    if providers is None or len(providers) != len(versions):
        weights = [1.0] * len(versions)
    else:
        weights = [_ocr_provider_weight(p) for p in providers]

    from collections import Counter
    normalized = [_normalize_for_match(v) for v in versions]
    counter = Counter(normalized)
    ranked = counter.most_common()
    top_count = ranked[0][1]

    if top_count >= 2:
        # Mayoría: alguna norma tiene >= 2 OCRs físicos detrás. Si VARIAS
        # normas tienen el mismo top_count, desempate por SUMA DE PESOS de
        # sus OCRs (los dedicated OCR pesan 1.5x → tres LLMs coincidiendo
        # pueden ser tumbados por dos dedicated coincidiendo).
        tied = [norm for norm, c in ranked if c == top_count]
        method = "vote"
        if len(tied) > 1:
            method = "vote_weighted_tiebreak"
        def _weight_of(norm: str) -> float:
            return sum(w for n, w in zip(normalized, weights) if n == norm)
        best_norm = max(tied, key=_weight_of)
        winning_weight = _weight_of(best_norm)
        # Entre las versiones literales con la norma ganadora, elige la más
        # COMPLETA. (Si dos OCRs escriben lo mismo con tipografía/case
        # distinta, la versión más rica gana.)
        candidates = [v for v, n in zip(versions, normalized) if n == best_norm]
        chosen = max(candidates, key=_completeness_score)
        return (chosen, {"n": len(versions), "majority": top_count,
                         "weight": winning_weight, "method": method})

    # No hay mayoría exacta (todas las normas tienen count=1) → MEDOID.
    # Peso del comparado interviene: un outlier "raro" pesa menos vs otros.
    def weighted_avg_sim(idx: int, v: str) -> float:
        sims_w = [(_similarity(v, other), weights[k])
                  for k, other in enumerate(versions) if k != idx]
        if not sims_w:
            return 0.0
        total_w = sum(w for _, w in sims_w) or 1.0
        return sum(s * w for s, w in sims_w) / total_w
    indexed = list(enumerate(versions))
    indexed.sort(key=lambda iv: (weighted_avg_sim(iv[0], iv[1]),
                                  weights[iv[0]],
                                  _completeness_score(iv[1])),
                 reverse=True)
    chosen = indexed[0][1]
    chosen_weight = weights[indexed[0][0]]
    return (chosen, {"n": len(versions), "majority": 1,
                     "weight": chosen_weight, "method": "medoid"})


def _fuse_ocr_transcriptions(ocr_results: dict) -> tuple:
    """Fusiona N transcripciones OCR en UNA canónica con consenso multi-señal.

    Pipeline:
      1. Parsear cada OCR (regex sobre formato canónico OCR_VIDEO_PROMPT).
      2. Alinear OCRs con Needleman-Wunsch (1970) — robusto a off-by-one,
         alucinaciones puntuales y OCRs que se saltan preguntas. Pos/num
         coincidentes dan bonus en el scoring.
      3. Por cada cluster (preguntas equivalentes entre OCRs), VOTACIÓN POR
         MAYORÍA sobre enunciado y cada opción A/B/C/D: si 2+ OCRs coinciden
         tras normalizar, esa versión gana; si todas discrepan, el medoid
         (versión más central) gana.

    Args:
      ocr_results: dict[provider_name] = {"ok": bool, "text": str, ...}

    Returns: tupla (fused_text, n_questions, stats) — NUNCA lanza.
    """
    try:
        # 1) Parsear cada OCR exitoso
        parsed_per_ocr: dict = {}
        skipped: list = []
        for provider, r in ocr_results.items():
            if not isinstance(r, dict) or not r.get("ok"):
                continue
            txt = (r.get("text") or "").strip()
            if not txt or txt.upper() == "NO_LEGIBLE":
                continue
            qs = _parse_ocr_text(txt)
            if not qs:
                skipped.append(provider)
                continue
            parsed_per_ocr[provider] = qs

        if not parsed_per_ocr:
            return ("", 0, {
                "parsed_per_ocr": {},
                "clusters":       0,
                "skipped":        skipped,
                "error":          "ningún OCR produjo texto parseable",
            })

        # 2) Construcción de clusters con NW incremental
        clusters = _build_clusters_multi_ocr(parsed_per_ocr)
        if not clusters:
            return ("", 0, {
                "parsed_per_ocr": {p: len(q) for p, q in parsed_per_ocr.items()},
                "clusters":       0,
                "skipped":        skipped,
                "error":          "alineación NW devolvió 0 clusters",
            })

        # 3) Anotar cada cluster con (sección consenso, num consenso, pos mediana)
        # y ordenar por (orden de aparición de sección, num, pos). Así respetamos
        # la estructura del documento: primero las preguntas sin sección, luego
        # las de cada bloque en el orden en que apareció.
        from statistics import median
        from collections import Counter as _Counter

        def _mode_or_first(values: list):
            """Moda; en empates devuelve el primero. None si lista vacía."""
            vs = [v for v in values if v is not None]
            if not vs:
                return None
            c = _Counter(vs)
            top_count = c.most_common(1)[0][1]
            for v in vs:
                if c[v] == top_count:
                    return v
            return vs[0]

        # Asignar metadatos consensuados a cada cluster. Como el spine respeta
        # el orden del documento (NW alinea preservando posiciones), iterar los
        # clusters en orden nos da el orden REAL de aparición de las secciones.
        annotated: list = []
        section_order: dict = {}  # _section_key → índice de aparición (0, 1, 2…)
        for cluster in clusters:
            sec_versions = [q.get("section") or "" for _, q in cluster]
            num_versions = [q.get("num") for _, q in cluster]
            pos_list     = [q.get("pos", 9999) for _, q in cluster]
            # Sección consenso: la más vista entre los OCRs del cluster
            sec_raw = _mode_or_first(sec_versions) or ""
            sec_key = _section_key(sec_raw)
            num_chosen = _mode_or_first(num_versions)
            try: pos_median = median(pos_list)
            except Exception: pos_median = 9999
            # Registrar orden secuencial de aparición de la sección
            if sec_key not in section_order:
                section_order[sec_key] = len(section_order)
            annotated.append({
                "cluster": cluster,
                "sec_raw": sec_raw,
                "sec_key": sec_key,
                "num":     num_chosen if num_chosen is not None else 9999,
                "pos":     pos_median,
            })

        # Ordenar: por orden de aparición de la sección (no por pos_median —
        # eso fallaba cuando una pregunta sin sección estaba al inicio y otra
        # con sección también en pos 0, empate roto por num en lugar de orden
        # de sección). Luego dentro de la sección por num, luego por pos.
        annotated.sort(key=lambda a: (
            section_order.get(a["sec_key"], 9999),
            a["num"],
            a["pos"],
        ))

        # 4) Por cada cluster, votación por mayoría sobre enunciado + opciones,
        # emitiendo además el marcador de sección cuando cambia y el N ORIGINAL.
        fused_blocks: list = []
        cluster_votes: list = []
        last_sec_key: str = "__none__"
        for entry in annotated:
            cluster   = entry["cluster"]
            sec_raw   = entry["sec_raw"]
            sec_key   = entry["sec_key"]
            num_orig  = entry["num"]

            # Votación sobre el enunciado — weighted (dedicated OCR pesan 1.5×)
            q_versions  = [q.get("question", "") for _, q in cluster]
            q_providers = [p for p, _ in cluster]
            best_question, q_vote = _vote_text(q_versions, providers=q_providers)

            # Votación por letra A/B/C/D — también weighted; cada letra mantiene
            # su propia lista de providers (no todas las IAs leen todas las opciones)
            merged_opts: dict = {}
            opt_votes: dict = {}
            for letter in ("A", "B", "C", "D"):
                versions: list = []
                opt_providers: list = []
                for p, q in cluster:
                    v = (q.get("options") or {}).get(letter, "")
                    if v and v.strip():
                        versions.append(v.strip())
                        opt_providers.append(p)
                if not versions:
                    continue
                best_opt, vote = _vote_text(versions, providers=opt_providers)
                merged_opts[letter] = best_opt
                opt_votes[letter] = vote

            # Emitir marcador de sección cuando entra una nueva (solo si tiene nombre)
            if sec_key != last_sec_key and sec_raw.strip():
                fused_blocks.append(f"## SECCION: {sec_raw.strip()}")
                last_sec_key = sec_key
            elif sec_key != last_sec_key:
                # Cambio a "sin sección" — reseteamos el tracking
                last_sec_key = sec_key

            # Construir bloque canónico con el num ORIGINAL (no renumerado)
            num_label = str(num_orig) if num_orig != 9999 else "?"
            block_lines = [f"{num_label}. {best_question.strip()}"]
            for letter in ("A", "B", "C", "D"):
                if letter in merged_opts:
                    block_lines.append(f"{letter}) {merged_opts[letter]}")
            fused_blocks.append("\n".join(block_lines))

            cluster_votes.append({
                "providers":   [p for p, _ in cluster],
                "size":        len(cluster),
                "section":     sec_raw,
                "num":         num_orig if num_orig != 9999 else None,
                "question":    q_vote,
                "options":     opt_votes,
                # Texto fusionado REAL (q_vote/opt_votes son solo stats de voto).
                # Lo consume _build_review_questions para renderizar las cards
                # alineadas 1:1 con merged_answer SIN re-parsear ocr_fused_text:
                # el re-parseo reasignaba `pos` (0..K-1 ordinal sobre el set
                # re-parseado) y, si el filtro anti-fantasma fusionaba/descartaba
                # un bloque, desplazaba la respuesta de toda pregunta posterior.
                "q_text":      best_question.strip(),
                "opts_text":   {L: merged_opts[L] for L in ("A", "B", "C", "D") if L in merged_opts},
            })

        # 5) Estadística: voted_count = nº de preguntas más repetido entre OCRs
        from collections import Counter
        ocr_counts = [len(qs) for qs in parsed_per_ocr.values()]
        voted_count = Counter(ocr_counts).most_common(1)[0][0] if ocr_counts else 0

        fused_text = "\n\n".join(fused_blocks)
        # n_questions = nº de preguntas reales (cada cluster votado = 1 pregunta)
        # NO cuenta los marcadores "## SECCION: ..." que están entremedias.
        n_questions = len(cluster_votes)
        stats = {
            "parsed_per_ocr": {p: len(q) for p, q in parsed_per_ocr.items()},
            "clusters":       len(clusters),
            "skipped":        skipped,
            "voted_count":    voted_count,
            "fused_chars":    len(fused_text),
            "method":         "needleman_wunsch + weighted_majority_vote",
            "cluster_votes":  cluster_votes,
        }
        return (fused_text, n_questions, stats)
    except Exception as exc:
        logger.error("💥 _fuse_ocr_transcriptions falló: %s", exc)
        traceback.print_exc(file=sys.stdout)
        return ("", 0, {"error": f"{type(exc).__name__}: {exc}"})


def _tavily_build_query(q: dict) -> str:
    """Construye un query corto y específico para Tavily a partir del dict de
    pregunta parseado (_parse_ocr_text). Concatena enunciado + opciones, recorta
    al cap (TAVILY_QUERY_MAX_CHARS) y limpia saltos de línea.

    Razón: Tavily acepta queries hasta ~400 chars; queries más cortas devuelven
    fuentes más relevantes. Incluir las opciones A/B/C/D ayuda a desambiguar
    preguntas técnicas (p.ej. "puerto X" sin opciones es ambiguo, con opciones
    A) 80 B) 443 C) 22 D) 25 Tavily indexa mejor)."""
    parts: list = []
    enunciado = (q.get("question") or "").strip()
    if enunciado:
        parts.append(enunciado)
    opts = q.get("options") or {}
    for letter in ("A", "B", "C", "D"):
        if letter in opts and opts[letter]:
            parts.append(f"{letter}) {opts[letter]}")
    raw = " ".join(parts)
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) > TAVILY_QUERY_MAX_CHARS:
        raw = raw[:TAVILY_QUERY_MAX_CHARS].rstrip() + "…"
    return raw


def _tavily_search_one(query: str, api_key: str, timeout_s: float,
                       max_results: int, search_depth: str) -> Optional[dict]:
    """Ejecuta UNA búsqueda en Tavily. Devuelve el JSON de respuesta o None si
    falló (timeout, 4xx/5xx, red rota, key inválida). NUNCA lanza.

    Doc oficial: https://docs.tavily.com/documentation/api-reference/endpoint/search
    """
    if not query or not api_key:
        return None
    payload = {
        "query":            query,
        "search_depth":     search_depth or "basic",
        "max_results":      int(max_results),
        "include_answer":   False,
        "include_raw_content": False,
        "include_images":   False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    try:
        # NO usamos _http_post_with_retry porque queremos SIN reintentos: si
        # Tavily falla, mejor dejarlo fuera del prompt que retrasar el job.
        # Usamos TAVILY_HTTP_CLIENT (pool dedicado) para NO agotar el pool del
        # HTTP_CLIENT compartido que necesitan los analyzers tras este paso.
        resp = TAVILY_HTTP_CLIENT.post(TAVILY_SEARCH_URL, json=payload, headers=headers,
                                 timeout=httpx.Timeout(connect=min(3.0, timeout_s),
                                                       read=timeout_s,
                                                       write=min(5.0, timeout_s),
                                                       pool=min(3.0, timeout_s)))
        if not resp.is_success:
            logger.warning("Tavily HTTP %d para query=%r: %s",
                           resp.status_code, query[:60], resp.text[:200])
            return None
        return resp.json()
    except Exception as exc:
        logger.warning("Tavily falló para query=%r: %s: %s",
                       query[:60], type(exc).__name__, exc)
        return None


def _tavily_format_results(results_per_q: list) -> str:
    """Renderiza los resultados de Tavily en un bloque XML <internet_context>
    que se inyecta al prompt del analyzer. Si no hay resultados utilizables
    para ninguna pregunta, devuelve "" (caller debe omitir el bloque).

    Formato pensado para que el analyzer entienda que es info SECUNDARIA, no
    autoritaria: el orden es OCR (preguntas) → instrucciones → contexto web."""
    blocks: list = []
    used_questions = 0
    for entry in results_per_q:
        num     = entry.get("num")
        sources = entry.get("sources") or []
        if not sources:
            continue
        used_questions += 1
        num_label = f"P{num}" if num is not None and num != 9999 else f"P?"
        src_lines: list = [f'  <question id="{num_label}">']
        for s in sources:
            title  = (s.get("title")   or "").strip().replace("\n", " ")
            url    = (s.get("url")     or "").strip()
            content = (s.get("content") or "").strip().replace("\n", " ")
            if len(content) > TAVILY_SNIPPET_MAX_CHARS:
                content = content[:TAVILY_SNIPPET_MAX_CHARS].rstrip() + "…"
            # C6: usar HTML entities en vez de caracteres Unicode raros (‹›).
            # Antes reemplazábamos `<` → `‹` que cambiaba semánticamente texto
            # como "x < 5" a "x ‹ 5", confundiendo al modelo. Las entities son
            # XML-safe y el modelo las entiende como su carácter original.
            # También escapamos `&` y `"` para mantener la integridad del atributo.
            def _xml_safe(s: str) -> str:
                return (s.replace("&", "&amp;").replace("<", "&lt;")
                         .replace(">", "&gt;").replace('"', "&quot;"))
            title_safe   = _xml_safe(title)
            content_safe = _xml_safe(content)
            url_safe     = _xml_safe(url)
            src_lines.append(
                f'    <source title="{title_safe[:160]}" url="{url_safe[:300]}">{content_safe}</source>'
            )
        src_lines.append('  </question>')
        blocks.append("\n".join(src_lines))
    if not blocks or used_questions == 0:
        return ""
    return (
        "<internet_context source=\"tavily\" note=\"Información de internet "
        "como referencia secundaria. NO es autoritaria — úsala solo para "
        "confirmar/desambiguar. Si contradice el sentido común de la "
        "pregunta, IGNÓRALA.\">\n"
        + "\n".join(blocks)
        + "\n</internet_context>"
    )


def _tavily_enrich_questions(questions: list, job_id: str) -> tuple:
    """Para CADA pregunta del examen lanza UNA búsqueda Tavily en paralelo.
    Espera HASTA TAVILY_TOTAL_DEADLINE_S; cualquier hilo no terminado a esa
    hora se DESCARTA (no bloqueamos al usuario).

    Devuelve (block_text, stats). block_text es "" si:
      - Tavily está desactivado en config.
      - No hay API key configurada.
      - Ninguna búsqueda terminó dentro del deadline.
      - Todas las búsquedas devolvieron resultados vacíos.

    El caller (pipeline) DEBE tolerar block_text="" — el prompt sigue sin él.
    """
    stats = {
        "enabled":      False,
        "n_questions":  len(questions or []),
        "n_started":    0,
        "n_completed":  0,
        "n_with_sources": 0,
        "elapsed_ms":   0,
        "skipped":      "",
        # Lista detallada para que el panel muestre exactamente qué se buscó:
        # [{num, query, n_sources, sources:[{title,url,content_trunc}]}]
        "queries":      [],
    }
    # 1) ¿Está habilitado? ¿Hay key?
    enabled = bool(DYNAMIC_CONFIG.get("TAVILY_ENABLED", TAVILY_ENABLED))
    if not enabled:
        stats["skipped"] = "tavily_disabled_config"
        return ("", stats)
    api_key = (DYNAMIC_CONFIG.get("TAVILY_API_KEY") or "").strip()
    backup_key = (DYNAMIC_CONFIG.get("TAVILY_API_KEY_BACKUP") or "").strip()
    if not api_key and not backup_key:
        stats["skipped"] = "no_api_key"
        return ("", stats)
    if not questions:
        stats["skipped"] = "no_questions"
        return ("", stats)
    stats["enabled"] = True
    # 2) Resolver tuning actual desde DYNAMIC_CONFIG (con defaults seguros).
    try:    max_results = int(DYNAMIC_CONFIG.get("TAVILY_MAX_RESULTS", TAVILY_MAX_RESULTS))
    except (TypeError, ValueError): max_results = TAVILY_MAX_RESULTS
    max_results = max(1, min(10, max_results))
    search_depth = str(DYNAMIC_CONFIG.get("TAVILY_SEARCH_DEPTH", TAVILY_SEARCH_DEPTH)).lower()
    if search_depth not in ("basic", "advanced"):
        search_depth = "basic"
    try:    http_timeout = float(DYNAMIC_CONFIG.get("TAVILY_HTTP_TIMEOUT_S", TAVILY_HTTP_TIMEOUT_S))
    except (TypeError, ValueError): http_timeout = TAVILY_HTTP_TIMEOUT_S
    http_timeout = max(1.0, min(30.0, http_timeout))
    try:    total_deadline = float(DYNAMIC_CONFIG.get("TAVILY_TOTAL_DEADLINE_S", TAVILY_TOTAL_DEADLINE_S))
    except (TypeError, ValueError): total_deadline = TAVILY_TOTAL_DEADLINE_S
    total_deadline = max(1.0, min(60.0, total_deadline))

    # 3) Lanzar UN thread por pregunta. Cada uno escribe su resultado en `out`.
    #    Si la primary falla y existe backup, _tavily_search_one se llama 2 veces.
    out: dict = {}     # idx → {"num", "sources":[{title,url,content},...]}
    out_lock = threading.Lock()
    t0 = time.monotonic()

    def _worker(idx: int, q: dict):
        worker_t0 = time.monotonic()
        query = _tavily_build_query(q)
        entry = {
            "num":       q.get("num"),
            "query":     query,
            "ok":        False,
            "error":     "",
            "n_sources": 0,
            "ms":        0,
            "sources":   [],
        }
        try:
            if not query:
                entry["error"] = "query_vacía"
                with out_lock: out[idx] = entry
                return
            data = _tavily_search_one(query, api_key, http_timeout,
                                       max_results, search_depth)
            used_backup = False
            if data is None and backup_key:
                used_backup = True
                data = _tavily_search_one(query, backup_key, http_timeout,
                                           max_results, search_depth)
            if not isinstance(data, dict):
                entry["error"] = "sin_respuesta" + (" (backup también falló)" if used_backup else "")
                with out_lock: out[idx] = entry
                return
            results = data.get("results") or []
            sources: list = []
            for r in results[:max_results]:
                if not isinstance(r, dict):
                    continue
                sources.append({
                    "title":   r.get("title")   or "",
                    "url":     r.get("url")     or "",
                    "content": r.get("content") or "",
                })
            entry["ok"]        = True
            entry["n_sources"] = len(sources)
            entry["sources"]   = sources
            with out_lock:
                out[idx] = entry
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
            try:
                with out_lock: out[idx] = entry
            except Exception:
                pass
            logger.warning("Tavily worker[%d] falló: %s: %s",
                           idx, type(exc).__name__, exc)
        finally:
            entry["ms"] = int((time.monotonic() - worker_t0) * 1000)

    # B9: usar ThreadPoolExecutor con worker count acotado. Si el OCR alucinó
    # 40 preguntas, antes lanzábamos 40 threads en paralelo y saturábamos el
    # scheduler + el pool HTTP. Ahora limitamos a max_workers para que el coste
    # sea predecible y el GIL no se queme. El deadline absoluto sigue mandando.
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
    # max_workers: 10 cubre el caso normal (8-12 preguntas) sin colas; si hay
    # más, las extra esperan turno → al deadline pueden quedar sin disparar,
    # pero el job NO se retrasa.
    _max_workers = min(10, max(1, len(questions)))
    deadline = time.monotonic() + total_deadline
    futures = []
    # NO usamos el context manager `with`: su __exit__ llama a shutdown(wait=True),
    # que vuelve a BLOQUEAR hasta que terminen TODOS los workers en cola — aunque
    # el deadline ya expiró. Eso violaba el "deadline absoluto" (podía bloquear
    # 12-24s extra en vez de respetar `total_deadline`). Lo gestionamos a mano.
    pool = ThreadPoolExecutor(max_workers=_max_workers,
                              thread_name_prefix=f"tavily-{job_id[:6]}")
    try:
        for idx, q in enumerate(questions):
            try:
                fut = pool.submit(_worker, idx, q)
                futures.append(fut)
                stats["n_started"] += 1
            except Exception as exc:
                # ERROR: si el submit a executor falla (recursos, thread starvation),
                # esa pregunta NO se busca en Tavily — degrada la fase de
                # enriquecimiento web. Visible en panel.
                logger.error("Tavily submit %d falló: %s", idx, exc)
        # Esperamos hasta el deadline absoluto. Lo que no termine se descarta.
        remaining = max(0.0, deadline - time.monotonic())
        wait(futures, timeout=remaining, return_when=ALL_COMPLETED)
    finally:
        # wait=False: liberamos al caller EN CUANTO vence el deadline; los HTTP en
        # curso no se abortan (httpx no es cancelable así) pero ya no bloquean —
        # sus hilos son daemon y mueren con su propio read-timeout.
        pool.shutdown(wait=False, cancel_futures=True)

    stats["elapsed_ms"]  = int((time.monotonic() - t0) * 1000)
    # n_completed = workers que entregaron resultado o error (no se quedaron colgados).
    stats["n_completed"] = sum(1 for e in out.values() if e.get("ok") or e.get("error"))

    # 5) Ordenar por idx original y renderizar.
    ordered = [out[i] for i in sorted(out.keys())]
    stats["n_with_sources"] = sum(1 for e in ordered if e.get("sources"))
    # Lista compacta para el panel: query + sources resumidas (sin content largo).
    stats["queries"] = [
        {
            "num":       e.get("num"),
            "query":     (e.get("query") or "")[:300],
            "ok":        bool(e.get("ok")),
            "error":     (e.get("error") or "")[:150],
            "ms":        e.get("ms", 0),
            "n_sources": e.get("n_sources", 0),
            "sources":   [
                {"title": (s.get("title") or "")[:150],
                 "url":   (s.get("url")   or "")[:300],
                 "snippet": ((s.get("content") or "")[:200])}
                for s in (e.get("sources") or [])[:max_results]
            ],
        }
        for e in ordered
    ]
    block_text = _tavily_format_results(ordered)
    return (block_text, stats)


def _run_video_pipeline(job_id: str, req: AskRequest, providers: list, expected: int):
    """Pipeline para modo VIDEO: N OCRs en paralelo → fusión texto → analyzers.
    Corre en su propio hilo daemon para no bloquear /ask.
    Reporta progreso fino vía job['phase'] para el panel."""
    # monotonic: mediciones de duración del pipeline (no es timestamp absoluto).
    pipeline_t0 = time.monotonic()
    video_kb = (len(req.video_b64) * 3 // 4 // 1024) if req.video_b64 else 0
    logger.info("🎬 VIDEO PIPELINE job=%s arranca · ~%d KB · OCR_providers=%s · analyzers=%s",
                job_id[:8], video_kb,
                [n for n,_ in _OCR_PROVIDERS], [n for n,_ in providers])
    try:
        # ──── Pre-extracción de frames (para mostrar en dashboard) ─────────
        # Sacamos top-3 frames más nítidos del video UNA vez aquí, antes de
        # spawn los OCRs. Razones:
        #   1. El dashboard puede mostrarlos al instante (no espera a que
        #      ningún OCR termine).
        #   2. Las funciones OCR de imagen (mistral_ocr, deepseek_ocr,
        #      glm_ocr, anthropic_ocr, openai_ocr) seguían extrayendo cada
        #      una por su cuenta — coste duplicado y posibles inconsistencias.
        #   3. Quedan guardados en job["extracted_frames"] para auditoría
        #      vía /api/frame/{job_id}/{idx}.
        # Sin LOCK porque la extracción puede tardar 200-400ms. El job
        # acaba de crearse y nadie más toca extracted_frames todavía.
        if req.video_b64:
            try:
                _frames = _extract_top_k_frames_jpeg_b64(
                    req.video_b64, req.video_mime, top_k=3,
                )
                if _frames:
                    with LOCK:
                        j = JOBS.get(job_id)
                        if j is not None:
                            j["extracted_frames"] = _frames
                            j["extracted_frames_count"] = len(_frames)
                    logger.info("📸 pre-extraídos %d frames del video · job=%s",
                                len(_frames), job_id[:8])
            except Exception as exc:
                logger.warning("pre-extracción de frames falló (no crítico, los OCRs extraerán por su cuenta): %s",
                               exc)

        # ──── Fusión multi-frame "imagen casi-perfecta" en BACKGROUND ─────
        # Lanzamos stacking (lucky+flow+anti-ghost+CLAHE+unsharp) + optional
        # Topaz Wonder 3 en un hilo aparte. NO bloquea la fase OCR — los
        # primeros OCRs (qwen/gemini/kimi/mimo) trabajan sobre el video crudo.
        # Cuando la fusión esté lista, los OCR de página única (Mistral OCR,
        # DeepSeek OCR, GLM OCR) que se lanzan AL FINAL de la fase 1 ya la
        # encontrarán en job["fused_image_b64"] vía _publish_partial.
        # Si todo falla, los OCR page-based caen al comportamiento clásico
        # (extraer top_k=1 frame por su cuenta).
        def _fuse_pipeline_bg_impl():
            t_stack = time.monotonic()
            try:
                stacked = _stack_top_frames_jpeg_b64(
                    req.video_b64, req.video_mime,
                    # Calidad MÁXIMA en Render Standard 2GB+: 8 frames stacking
                    # da √8≈2.8× reducción de ruido, ideal para OCR sobre folios
                    # con texto pequeño. samples_total=24 da margen amplio al
                    # Lucky Imaging para descartar frames borrosos. Peak temporal
                    # ~150MB durante 10-30s del stacking — holgado en 2GB.
                    n_keep=8, samples_total=24, jpeg_quality=92,
                    max_side=2400,   # subido de 2048: una A4 densa necesita más resolución para OCR
                )
            except Exception as exc:
                # ERROR (visible en VER ERRORES): el stacking en background
                # fallando significa que NO hay imagen "casi-perfecta" para
                # los OCR page-based ni para el panel — calidad OCR degradada.
                logger.error("[fuse] stacking lanzó: %s", exc)
                stacked = None
            stack_ms = int((time.monotonic() - t_stack) * 1000)
            if not stacked:
                logger.info("[fuse] stacking devolvió None (cv2 no disponible "
                            "o video corrupto) · job=%s", job_id[:8])
                return
            logger.info("[fuse] ✓ stacking local OK (%d ms · %d KB) · job=%s",
                        stack_ms, len(stacked[0]) * 3 // 4 // 1024, job_id[:8])
            # DEWARP: enderezar la hoja (cámara en el pecho → ángulo oblicuo).
            # Corregir la perspectiva mejora mucho el OCR de imagen. ADITIVO: si
            # no detecta una hoja clara, conserva el stacking tal cual (sin riesgo).
            try:
                _dw = _dewarp_document(stacked[0], stacked[1])
            except Exception:
                _dw = None
            if _dw:
                best_b64, best_mime = _dw
                local_b64, local_mime = _dw
                _fused_src = "stacking_local+dewarp"
                logger.info("[fuse] ✓ dewarp aplicado (hoja enderezada) · job=%s", job_id[:8])
            else:
                best_b64, best_mime = stacked
                local_b64, local_mime = stacked  # backup por si Topaz devuelve mala
                _fused_src = "stacking_local"

            # Cachear ya el stacking local — si Topaz tarda o falla, el dashboard
            # y los OCRs ya tienen una versión utilizable.
            with LOCK:
                j = JOBS.get(job_id)
                if j is not None:
                    j["fused_image_b64"]      = best_b64
                    j["fused_image_mime"]     = best_mime
                    j["fused_image_source"]   = _fused_src
                    j["fused_image_stack_ms"] = stack_ms
                    j["has_fused_image"]      = True
            # Inyectar también al `req` para que las OCR page-based esperen
            # sobre `req.fused_image_b64` (pueden estar ya corriendo). Esto es
            # la señal "ya hay algo utilizable" — Topaz puede pisar después.
            try:
                req.fused_image_b64  = best_b64
                req.fused_image_mime = best_mime
            except Exception:
                pass

            # Pase opcional Topaz Wonder 3. Si no hay key/falla, fused queda
            # como stacking_local (ya cacheado arriba).
            if DYNAMIC_CONFIG.get("TOPAZ_ENABLED") and DYNAMIC_CONFIG.get("TOPAZ_API_KEY"):
                t_topaz = time.monotonic()
                try:
                    topaz_result = _topaz_enhance_image(best_b64, best_mime)
                except Exception as exc:
                    logger.warning("[fuse] Topaz lanzó: %s", exc)
                    topaz_result = None
                topaz_ms = int((time.monotonic() - t_topaz) * 1000)
                if topaz_result:
                    t_b64, t_mime = topaz_result
                    logger.info("[fuse] ✓ Topaz Wonder 3 OK (%d ms · %d KB) · job=%s",
                                topaz_ms, len(t_b64) * 3 // 4 // 1024, job_id[:8])
                    with LOCK:
                        j = JOBS.get(job_id)
                        if j is not None:
                            j["fused_image_b64"]     = t_b64
                            j["fused_image_mime"]    = t_mime
                            j["fused_image_source"]  = "topaz_wonder3"
                            j["fused_image_topaz_ms"] = topaz_ms
                            # Conservamos también el local por si el dashboard quiere comparar
                            j["fused_image_local_b64"]  = local_b64
                            j["fused_image_local_mime"] = local_mime
                    # Pisar el req con la versión mejorada para que cualquier
                    # OCR page-based que aún no haya empezado use la de Topaz.
                    try:
                        req.fused_image_b64  = t_b64
                        req.fused_image_mime = t_mime
                    except Exception:
                        pass
                else:
                    logger.info("[fuse] Topaz falló/timeout (%d ms) · fused queda como stacking_local · job=%s",
                                topaz_ms, job_id[:8])
                    with LOCK:
                        j = JOBS.get(job_id)
                        if j is not None:
                            j["fused_image_topaz_ms"]    = topaz_ms
                            j["fused_image_topaz_status"] = "failed"

        def _fuse_pipeline_bg():
            # Guard de NIVEL SUPERIOR: este pipeline corre en un daemon thread.
            # Sin este try/except, CUALQUIER excepción no controlada en el cuerpo
            # (el `with LOCK`, el desempaquetado de dewarp, len() sobre algo nulo,
            # etc.) mataría el thread EN SILENCIO — sin entrada en VER ERRORES y,
            # peor, dejando a los OCR page-based esperando un fused_image que no
            # llegará. Aquí lo registramos visiblemente; los OCR caen a su
            # fallback clásico (extraer su propio frame).
            try:
                _fuse_pipeline_bg_impl()
            except Exception as exc:
                logger.error("[fuse] pipeline de fusión lanzó excepción no "
                             "controlada (thread no muere en silencio): %s",
                             exc, exc_info=True)

        if req.video_b64:
            threading.Thread(target=_fuse_pipeline_bg,
                              daemon=True,
                              name=f"fuse-{job_id[:6]}").start()

        # ──── Fase 1: OCR paralelo (Qwen + Gemini Video + Kimi + MiMo) ─────
        _set_job_phase(job_id, "ocr_running", {
            "ocr_providers": [n for n,_ in _OCR_PROVIDERS],
            "video_kb": video_kb,
        })

        ocr_results: dict = {}
        ocr_lock = threading.Lock()
        ocr_done_count = [0]  # mutable contador en closure

        def _publish_ocr_result(name: str, entry: dict) -> int:
            """Publica el resultado de un OCR EN TIEMPO REAL al job para que el panel
            lo vea según vayan terminando, no al final del pipeline. Thread-safe.

            HARDENING: cada paso (ocr_lock, LOCK, mark_dirty) está envuelto en
            try/except independiente para que un fallo en uno NO impida los
            demás. Si el LOCK del job está bloqueado por otra operación, el
            snapshot no se actualiza pero el resultado SÍ se registra
            internamente — así el pipeline puede continuar.

            Devuelve el contador de OCRs completados (o 0 si no se pudo contar).
            """
            done_now = 0
            # 1) Actualizar el dict interno ocr_results (siempre debe funcionar).
            try:
                with ocr_lock:
                    ocr_results[name] = entry
                    ocr_done_count[0] += 1
                    done_now = ocr_done_count[0]
            except Exception as exc:
                logger.error("💥 _publish_ocr_result[%s] ocr_lock falló: %s",
                             name, exc)
                # Continuar igual — el resultado se persiste de la siguiente forma.
                ocr_results[name] = entry
            # 2) Snapshot al job (puede fallar si LOCK ocupado — no crítico).
            try:
                with LOCK:
                    j = JOBS.get(job_id)
                    if j is not None:
                        j["ocr_results"] = dict(ocr_results)
            except Exception as exc:
                # ERROR: si el LOCK snapshot falla, el panel pierde la
                # actualización en tiempo real del OCR. El ocr_results
                # internal SÍ se persiste (paso 1) pero el operador no lo ve
                # hasta el siguiente flush completo. Worth knowing.
                logger.error("_publish_ocr_result[%s] LOCK snapshot falló: %s",
                             name, exc)
            # 3) Marcar dirty para persistencia (best-effort).
            try:
                _mark_jobs_dirty()
            except Exception:
                pass
            return done_now

        def _ocr_runner(name, fn):
            """Corre UN OCR de forma totalmente aislada del resto.

            HARDENING:
            - Timeout independiente por OCR (AI_HARD_TIMEOUT, override por provider).
            - Try/except envuelve TODO incluyendo la publicación del resultado.
            - Si _publish_ocr_result lanza, lo capturamos para que el thread no muera
              silenciosamente — el log queda como evidencia.
            - El _set_job_phase es BEST-EFFORT: si falla, no rompe el OCR.
            - TODOS los OCRs corren en paralelo sin límite (en Render Standard 2GB
              hay margen de sobra para 9 simultáneos × video b64 ~3MB en buffers).
            """
            # monotonic: igual que call_ai_task — mediciones de duración inmunes a NTP.
            t0 = time.monotonic()
            logger.info("  🔍 OCR[%s] arranca (job=%s)", name, job_id[:8])
            # Timeout específico por OCR — algunos providers son consistentemente
            # más lentos (Kimi 30-50s, MiMo 20-40s). Usamos override si está,
            # si no AI_HARD_TIMEOUT. Cada OCR tiene su PROPIO timeout aquí —
            # esto NO depende del OCR_PHASE_DEADLINE_S del wait de fuera.
            ocr_timeout = _PROVIDER_TIMEOUTS.get(name, AI_HARD_TIMEOUT)
            try:
                res = _call_ai_with_timeout(fn, req, ocr_timeout)
                txt = (res or {}).get("raw", "") or ""
                ms = int((time.monotonic()-t0)*1000)
                # Cap defensivo del texto OCR EN MEMORIA: si un VLM aluciona y
                # devuelve MB+ de texto (visto con prompts mal interpretados o
                # videos corruptos), 9 OCRs × 6 jobs × 5MB = 270MB en JOBS sin
                # control. 200KB cubre cualquier transcripción real de examen
                # (un examen 40 preguntas × ~400 chars = 16 KB).
                _OCR_TEXT_MAX_CHARS = 200_000
                txt_truncated_from = 0
                if isinstance(txt, str) and len(txt) > _OCR_TEXT_MAX_CHARS:
                    txt_truncated_from = len(txt)
                    txt = txt[:_OCR_TEXT_MAX_CHARS]
                # Diagnóstico fino — categorizar la respuesta para el panel y los logs.
                #   empty: el modelo no devolvió texto (safety filter, MAX_TOKENS sin
                #          contenido, formato de respuesta no esperado, vídeo no
                #          procesado, etc.). Lo más alarmante — necesita revisión.
                #   no_legible: el modelo procesó pero declaró el vídeo ilegible
                #          siguiendo OCR_VIDEO_PROMPT. Es estado "honesto", no fallo.
                #   ok: texto útil presente.
                txt_strip = txt.strip()
                txt_upper = txt_strip.upper()
                if not txt_strip:
                    ocr_quality = "empty"
                elif txt_upper == "NO_LEGIBLE" or txt_upper.startswith("NO_LEGIBLE"):
                    ocr_quality = "no_legible"
                else:
                    ocr_quality = "ok"
                model_used = (res or {}).get("model", "")
                preview = txt_strip[:200].replace("\n", " ⏎ ")
                _publish_payload = {
                    "ok": True, "text": txt, "ms": ms,
                    "model": model_used,
                    "quality": ocr_quality,        # "ok" | "empty" | "no_legible"
                    "preview_diag": preview[:200], # para que el panel lo muestre rápido sin /partial
                }
                if txt_truncated_from:
                    _publish_payload["text_truncated_from"] = txt_truncated_from
                try:
                    done_now = _publish_ocr_result(name, _publish_payload)
                except Exception as pub_exc:
                    # Falló la publicación pero NO el OCR. Lo registramos.
                    logger.error("💥 OCR[%s] publish falló (texto sí obtenido %d chars): %s",
                                 name, len(txt), pub_exc)
                    done_now = 0
                # Log por calidad — distingue claramente los tres casos para diagnóstico.
                if ocr_quality == "empty":
                    logger.warning(
                        "  ⚠️ OCR[%s] devolvió texto VACÍO · %dms · model=%s · "
                        "raw_len=%d · (posible safety filter, MAX_TOKENS, formato no esperado, "
                        "o vídeo no procesado — comprobar key/quota/mime)",
                        name, ms, model_used, len(txt))
                elif ocr_quality == "no_legible":
                    logger.info(
                        "  ⓘ OCR[%s] dice NO_LEGIBLE · %dms · model=%s · "
                        "(modelo procesó vídeo pero declaró ilegible — vídeo borroso, "
                        "fuera de foco, sin contenido)",
                        name, ms, model_used)
                else:
                    logger.info(
                        "  ✅ OCR[%s] OK %d chars · %dms (%d/%d done) · model=%s · preview=%r",
                        name, len(txt), ms, done_now, len(_OCR_PROVIDERS),
                        model_used, preview[:120])
                # _set_job_phase es BEST-EFFORT: si falla no propaga.
                try:
                    _set_job_phase(job_id, "ocr_running", {
                        "ocr_done": done_now,
                        "ocr_total": len(_OCR_PROVIDERS),
                        "ocr_last_provider": name,
                    })
                except Exception as phase_exc:
                    logger.warning("OCR[%s] _set_job_phase falló (no crítico): %s",
                                   name, phase_exc)
            except NoApiKeyError as e:
                # Sin key configurada → no es fallo, simplemente no participa.
                ms = int((time.monotonic()-t0)*1000)
                logger.info("  ⏭ OCR[%s] sin key configurada → no participa (%dms)",
                            name, ms)
                try:
                    _publish_ocr_result(name, {
                        "ok": False, "text": "",
                        "error": "Sin API key configurada", "ms": ms,
                        "status": "no_key",
                    })
                except Exception as pub_exc:
                    logger.error("💥 OCR[%s] publish no_key falló: %s", name, pub_exc)
            except Exception as e:
                ms = int((time.monotonic()-t0)*1000)
                err_msg = f"{type(e).__name__}: {e}"
                # Aislamiento total: aunque _publish_ocr_result lance, capturamos
                # para que el thread del OCR termine limpio y no afecte a los demás.
                try:
                    _publish_ocr_result(name, {
                        "ok": False, "text": "", "error": err_msg[:200], "ms": ms,
                    })
                except Exception as pub_exc:
                    logger.error("💥 OCR[%s] publish error falló: %s · err_original=%s",
                                 name, pub_exc, err_msg[:120])
                logger.error("  ❌ OCR[%s] FAIL · %dms · %s", name, ms, err_msg[:120])

        ocr_threads = []
        for name, fn in _OCR_PROVIDERS:
            try:
                t = threading.Thread(target=_ocr_runner, args=(name, fn),
                                     daemon=True, name=f"ocr-{name}-{job_id[:6]}")
                t.start()
                ocr_threads.append((name, t))
            except Exception as spawn_exc:
                # Spawn falló (thread starvation, OS limit, etc.) → registrar como
                # OCR fallido para que la lógica de "todos fallaron" funcione.
                # Los demás OCRs siguen lanzándose normalmente — aislamiento total.
                logger.error("💥 No se pudo arrancar OCR thread %s: %s", name, spawn_exc)
                try:
                    _publish_ocr_result(name, {
                        "ok": False, "text": "",
                        "error": f"thread spawn falló: {spawn_exc}",
                        "ms": 0,
                    })
                except Exception:
                    pass

        # ─── Timeout TOTAL para fase OCR ─────────────────────────────────────
        # Antes: join secuencial por thread con timeout 125s/thread → en el peor
        # caso el bucle esperaba hasta 3 × 125s. Ahora: 90s ABSOLUTOS desde que
        # arrancan los OCRs. Si alguno tarda más, lo marcamos como timed_out y
        # arrancamos la fase 2 con los OCRs que sí terminaron (siempre que haya
        # al menos uno). Esto evita que un solo OCR lento (Kimi 30-50s) o colgado
        # haga esperar a los demás cuando ya tenemos texto suficiente.
        OCR_PHASE_DEADLINE_S = float(os.environ.get("OCR_PHASE_TIMEOUT", "90"))
        # monotonic: el deadline es duración relativa, no fecha absoluta.
        ocr_deadline = time.monotonic() + OCR_PHASE_DEADLINE_S
        for name, t in ocr_threads:
            remaining = max(0.05, ocr_deadline - time.monotonic())
            t.join(timeout=remaining)
            if t.is_alive():
                # Thread sigue corriendo tras la deadline → marcamos timed_out.
                # NO podemos matar el thread (Python threads no son cancelables),
                # pero al menos lo registramos y el resultado tardío se descarta
                # porque la fase 2 ya habrá arrancado.
                if name not in ocr_results:
                    _publish_ocr_result(name, {
                        "ok": False, "text": "",
                        "error": f"timeout fase OCR ({OCR_PHASE_DEADLINE_S:.0f}s totales agotados)",
                        "ms": int(OCR_PHASE_DEADLINE_S * 1000),
                    })
                    # ERROR (no warning): un OCR perdido degrada la fusión y
                    # puede causar respuestas incorrectas. Visible en VER ERRORES.
                    logger.error("  ⏱ OCR[%s] TIMEOUT tras %ds — sigue procesando en background, su resultado se descartará",
                                 name, int(OCR_PHASE_DEADLINE_S))

        with LOCK:
            job = JOBS.get(job_id)
            if not job:
                # ERROR: el job desapareció entre /ask y OCR_done. Significa
                # que gc_jobs lo purgó (job inactivo >2h) o /reset corrió
                # durante el pipeline. Bug grave: hemos gastado 9 OCRs en vano.
                logger.error("⚠️ job %s desapareció antes de OCR done — todos los OCRs descartados",
                             job_id[:8])
                return
            job["ocr_results"] = dict(ocr_results)

        # ──── Filtrar OCRs que produjeron texto real ──────────────────────────
        # Antes de fusionar, identificamos cuántas OCRs emitieron contenido útil.
        # Una OCR con ok=True pero texto vacío o "NO_LEGIBLE" no aporta nada.
        # Clasificación en 3 grupos (NO confundir):
        #   - ok_ocr_providers: corrieron y devolvieron texto VÁLIDO (las "buenas").
        #   - no_text_count:    corrieron OK pero SIN texto (empty / NO_LEGIBLE) →
        #                       voto "el folio es ilegible".
        #   - excluded_count:   error de red/API o sin key (ok=False) → NO procesaron
        #                       la imagen, así que NO son señal de legibilidad y NO
        #                       cuentan ni a favor ni en contra.
        ok_ocr_providers = []
        no_text_count = 0
        excluded_count = 0
        for name, r in ocr_results.items():
            if not isinstance(r, dict) or not r.get("ok"):
                excluded_count += 1
                continue
            txt = (r.get("text") or "").strip()
            if not txt or txt.upper() == "NO_LEGIBLE":
                no_text_count += 1
                continue
            ok_ocr_providers.append(name)
        ok_count = len(ok_ocr_providers)

        if ok_count == 0:
            # Ningún OCR produjo texto válido → marcar error directo sin lanzar analyzers
            errs = [f"{n}: {r.get('error','?')[:80]}"
                    for n, r in ocr_results.items() if not r.get("ok")]
            err_text = ("OCR de video falló · " + " | ".join(errs)) if errs else \
                       "OCR de video: ningún proveedor detectó texto"
            with LOCK:
                job = JOBS.get(job_id)
                if job:
                    job["status"]   = "error"
                    job["finished"] = time.time()
                    job["error"]    = err_text
            _set_job_phase(job_id, "error_ocr")
            logger.error("💥 Job %s → error (OCR video falló · %s)", job_id[:8], err_text[:200])
            _mark_jobs_dirty()
            return

        # GUARD ANTI-ALUCINACIÓN: si MÁS de OCR_NO_TEXT_ERROR_THRESHOLD OCRs corrieron
        # pero NO detectaron texto, el folio es casi seguro ilegible (borroso, fuera de
        # cuadro, vacío) y las pocas OCR que sí "leyeron" algo probablemente lo
        # ALUCINARON — los frontier vision LLMs inventan preguntas plausibles ante una
        # imagen basura. Las OCR con error/sin-key NO cuentan (no procesaron la imagen).
        # Preferimos marcar error y que el siguiente ciclo reintente con captura nueva
        # antes que vibrarle al usuario respuestas inventadas.
        if no_text_count > OCR_NO_TEXT_ERROR_THRESHOLD:
            err_text = (f"Imagen probablemente ilegible: {no_text_count} OCRs no detectaron "
                        f"texto y solo {ok_count} sí (riesgo de alucinación) · "
                        f"{excluded_count} con error/sin-key no cuentan")
            with LOCK:
                job = JOBS.get(job_id)
                if job:
                    job["status"]   = "error"
                    job["finished"] = time.time()
                    job["error"]    = err_text
            _set_job_phase(job_id, "error_ocr")
            logger.error("💥 Job %s → error (%s)", job_id[:8], err_text)
            _mark_jobs_dirty()
            return

        # ──── Fusión OCR server-side (consenso) ─────────────────────────────
        # _fuse_ocr_transcriptions parsea las N OCRs y produce UNA transcripción
        # canónica con K preguntas únicas. Esto elimina el problema de
        # "respuestas duplicadas con un montón de letras" porque el analyzer ya
        # NO ve N transcripciones — solo una. NUNCA lanza: si falla devuelve
        # texto vacío + stats con error, y caemos al método anterior <readings>.
        fused_text, n_fused_questions, fusion_stats = _fuse_ocr_transcriptions(ocr_results)
        # use_fused estricto: el texto fusionado debe tener contenido REAL
        # (no solo whitespace/saltos de línea) Y al menos 1 pregunta detectada.
        # Si cualquiera falla → fallback a <readings> crudo (cero regresión).
        use_fused = (
            isinstance(fused_text, str)
            and bool(fused_text.strip())
            and isinstance(n_fused_questions, int)
            and n_fused_questions > 0
        )
        # Persistir la fusión en el job para que el panel pueda mostrarla y para
        # debugging. NO bloquea si LOCK está ocupado mucho tiempo — es solo info.
        try:
            with LOCK:
                j = JOBS.get(job_id)
                if j is not None:
                    j["ocr_fused_text"]  = fused_text
                    j["ocr_fusion_stats"] = fusion_stats
                    j["ocr_fusion_used"]  = use_fused
        except Exception as exc:
            # ERROR: sin persistir el ocr_fused_text, el panel no muestra
            # la transcripción fusionada — el operador no puede auditar qué
            # texto vieron realmente los analyzers.
            logger.error("No se pudo persistir ocr_fused_text: %s", exc)

        if use_fused:
            logger.info("🧬 Job %s · fusión OCR exitosa · %d preguntas únicas · "
                        "stats=%s",
                        job_id[:8], n_fused_questions, fusion_stats)
        else:
            logger.warning("⚠️ Job %s · fusión OCR falló o vacía · fallback a "
                           "<readings> crudas · stats=%s",
                           job_id[:8], fusion_stats)

        # ──── Construcción del bloque de OCR para el analyzer ────────────────
        # Caso A (preferido): tenemos fusión Python → bloque <exam> único con K preguntas
        # Caso B (fallback):  fusión falló → bloque <readings> con N transcripciones crudas
        # A9: el texto OCR puede contener (alucinación del modelo o adversarial en
        # el examen real) la cadena literal "</exam>", "</reading>", "</readings>"
        # — si dejamos pasar, cierra el bloque XML antes de tiempo y el resto del
        # texto se interpreta como instrucción del razonador (prompt injection).
        # Sanitización mínima: escapamos los cierres exactos. No tocamos el resto
        # porque las IAs toleran XML mal formado y romperíamos contenido legítimo
        # tipo "x < 5" o "f(x) > 0".
        def _sanitize_xml_close(s: str) -> str:
            return (s.replace("</exam>", "<\\/exam>")
                     .replace("</reading>", "<\\/reading>")
                     .replace("</readings>", "<\\/readings>"))

        if use_fused:
            ocr_for_prompt = (
                f"<exam questions=\"{n_fused_questions}\">\n"
                f"{_sanitize_xml_close(fused_text)}\n"
                f"</exam>"
            )
        else:
            # Fallback: emitir <readings> tal como el método antiguo
            blocks = []
            for idx, name in enumerate(ok_ocr_providers, start=1):
                txt = (ocr_results[name].get("text") or "").strip()
                txt = _sanitize_xml_close(txt)
                blocks.append(
                    f'  <reading index="{idx}" source="ocr_{idx}">\n'
                    f'{txt}\n'
                    f'  </reading>'
                )
            ocr_for_prompt = (
                f"<readings count=\"{ok_count}\">\n"
                + "\n".join(blocks)
                + f"\n</readings>"
            )

        ocr_phase_ms = int((time.monotonic() - pipeline_t0) * 1000)
        logger.info("📝 Job %s · OCR listo para analyzer · %d chars · %d/%d providers OK · "
                    "fusion_used=%s · %dms total",
                    job_id[:8], len(ocr_for_prompt), ok_count, len(_OCR_PROVIDERS),
                    use_fused, ocr_phase_ms)
        _set_job_phase(job_id, "ocr_done", {
            "ocr_chars":      len(ocr_for_prompt),
            "ocr_ok_count":   ok_count,
            "ocr_total":      len(_OCR_PROVIDERS),
            "ocr_phase_ms":   ocr_phase_ms,
            "fusion_used":    use_fused,
            "fused_questions": n_fused_questions if use_fused else 0,
        })

        # ──── Fase 1.5 (intermedia): Tavily web search por pregunta ──────────
        # Entre el OCR y el razonamiento añadimos contexto de internet — UNA
        # búsqueda por pregunta detectada, ejecutadas en paralelo, con un
        # deadline TOTAL absoluto (TAVILY_TOTAL_DEADLINE_S). Si Tavily no
        # responde a tiempo, el pipeline sigue exactamente como antes (el
        # bloque <internet_context> simplemente no se añade al prompt).
        #
        # Fuente de "preguntas": re-parseamos fused_text con _parse_ocr_text.
        # En fallback (sin fusión) parseamos la mejor OCR cruda — así Tavily
        # funciona en ambos caminos. Si nada parsea, omitimos el paso.
        tavily_block = ""
        tavily_stats = {"enabled": False, "skipped": "no_parsed_questions"}
        try:
            _set_job_phase(job_id, "tavily_running", {})
            if use_fused and fused_text:
                parsed_for_tavily = _parse_ocr_text(fused_text)
            else:
                # Sin fusión: la OCR más completa (mayor n preguntas parseables).
                best_parsed: list = []
                for nm in ok_ocr_providers:
                    raw_txt = (ocr_results.get(nm, {}).get("text") or "").strip()
                    if not raw_txt:
                        continue
                    pq = _parse_ocr_text(raw_txt)
                    if len(pq) > len(best_parsed):
                        best_parsed = pq
                parsed_for_tavily = best_parsed
            if parsed_for_tavily:
                tavily_t0 = time.monotonic()
                tavily_block, tavily_stats = _tavily_enrich_questions(
                    parsed_for_tavily, job_id,
                )
                tavily_ms = int((time.monotonic() - tavily_t0) * 1000)
                if tavily_stats.get("enabled"):
                    logger.info("🌐 Job %s · Tavily · %d/%d preguntas con fuentes · "
                                "%d/%d hilos completados · %dms",
                                job_id[:8],
                                tavily_stats.get("n_with_sources", 0),
                                tavily_stats.get("n_questions", 0),
                                tavily_stats.get("n_completed", 0),
                                tavily_stats.get("n_started", 0),
                                tavily_ms)
                else:
                    logger.info("🌐 Job %s · Tavily SKIP · motivo=%s",
                                job_id[:8], tavily_stats.get("skipped", "?"))
            # Persistir stats + bloque inyectado al job para el panel.
            # tavily_block: el XML <internet_context> EXACTO que se añade al prompt,
            # para que el usuario pueda verificar qué recibió el razonador.
            try:
                with LOCK:
                    j = JOBS.get(job_id)
                    if j is not None:
                        j["tavily_stats"]    = tavily_stats
                        j["tavily_used"]     = bool(tavily_block)
                        j["tavily_chars"]    = len(tavily_block) if tavily_block else 0
                        j["tavily_block"]    = tavily_block or ""
            except Exception:
                pass
        except Exception as exc:
            # Aislamiento total: si CUALQUIER cosa de Tavily explota, seguimos
            # sin internet_context — el pipeline NO debe morir por esto.
            logger.warning("Tavily enrichment falló (no crítico): %s: %s",
                           type(exc).__name__, exc)
            tavily_block = ""
            tavily_stats = {"enabled": False, "skipped": f"exception:{type(exc).__name__}"}
        _set_job_phase(job_id, "tavily_done", {
            "tavily_used":           bool(tavily_block),
            "tavily_n_with_sources": tavily_stats.get("n_with_sources", 0) if isinstance(tavily_stats, dict) else 0,
            "tavily_elapsed_ms":     tavily_stats.get("elapsed_ms", 0) if isinstance(tavily_stats, dict) else 0,
        })

        # ──── Fase 2: prompt enriquecido ────────────────────────────────────
        # Estructura basada en best practices de Anthropic 2026:
        #   1. Bloque OCR (fusión <exam> o fallback <readings>) PRIMERO
        #      (modelos grandes responden hasta +30% mejor con datos al inicio).
        #   2. Rol e instrucciones del Android (req.prompt) — la "voz del system".
        #   3. <protocol> con POE + self-check, JUSTO ANTES de la query final.
        #   4. Query final corta, accionable.
        #
        # CLAVE: el protocolo cambia según si usamos fusión o fallback.
        # - FUSIÓN: protocolo simple (1 transcripción → 1 respuesta por pregunta).
        #           NO existe el riesgo "responder 3 veces" porque solo hay 1 lectura.
        # - FALLBACK: protocolo de fusión manual (instruye al analyzer a dedupar
        #           él mismo, como antes).
        if use_fused:
            # Protocolo simplificado: una sola transcripción, K preguntas claras.
            # Sin riesgo de multiplicar respuestas — solo hay un <exam>.
            protocol = f"""<protocol>
  Tienes UNA transcripción del examen en el bloque <exam questions="{n_fused_questions}">
  al principio de este mensaje. Esa transcripción YA está fusionada y deduplicada
  por el servidor — tiene exactamente {n_fused_questions} preguntas únicas
  numeradas del 1 al {n_fused_questions}.

  PASO 1 — INVENTARIO:
    Hay EXACTAMENTE {n_fused_questions} preguntas. Cuéntalas para confirmar.
    Si tu conteo NO da {n_fused_questions}, asume que es {n_fused_questions} —
    el servidor ya hizo la deduplicación. NO inventes preguntas adicionales,
    NO repitas preguntas.

  PASO 2 — RESOLUCIÓN POR ELIMINACIÓN (POE):
    Para CADA una de las {n_fused_questions} preguntas, EN ORDEN:
      a) Cita brevemente la pregunta (1 línea).
      b) Elimina las 1-3 opciones claramente erróneas, di por qué en 1 frase.
      c) De las opciones que quedan, escoge la más correcta.
      d) Si NINGUNA elimina con seguridad → tu mejor apuesta razonada.
      e) Solo X si la pregunta es 100% incomprensible.

  PASO 3 — AUTO-VERIFICACIÓN:
    Cuenta las letras que pondrás en <FINAL>.
    Debe ser EXACTAMENTE {n_fused_questions}.
    Si te sale otro número, recuenta y corrige antes de escribir <FINAL>.
</protocol>"""
        elif ok_count > 1:
            # Fallback con múltiples OCRs: el analyzer tiene que dedupar él mismo.
            protocol = f"""<protocol>
  Tienes {ok_count} transcripciones independientes del MISMO examen filmado en
  video, listadas en el bloque <readings> al principio de este mensaje. Procede:

  PASO 1 — INVENTARIO ÚNICO:
    Recorre las {ok_count} lecturas y construye UNA lista única de preguntas.
    • Si una pregunta aparece en varias lecturas, cuenta UNA sola vez.
    • Si una pregunta aparece solo en una lectura, igualmente la incluyes.
    • Si las lecturas discrepan en el texto, elige por mayoría ({ok_count // 2 + 1} de {ok_count}).
      Si no hay mayoría, la versión más larga/completa gana.
    • Si un trozo está marcado [?] en una lectura pero es claro en otra, usa el claro.
    Antes de seguir, declara: "He identificado K preguntas distintas" (K es número).
    CRÍTICO: K NO se multiplica por {ok_count}. Si la lectura más completa tiene
    8 preguntas, K=8 — NUNCA 24.

  PASO 2 — RESOLUCIÓN POR ELIMINACIÓN (POE):
    Para CADA pregunta del inventario, en orden:
      a) Cita brevemente la pregunta (1 línea).
      b) Elimina las 1-3 opciones claramente erróneas, di por qué en 1 frase.
      c) De las opciones que quedan, escoge la más correcta.
      d) Si NINGUNA elimina con seguridad → tu mejor apuesta razonada.
      e) Solo X si la pregunta es 100% incomprensible incluso fusionando lecturas.

  PASO 3 — AUTO-VERIFICACIÓN:
    Antes de cerrar, cuenta las letras que pondrás en <FINAL>.
    Debe ser EXACTAMENTE K (el número declarado en PASO 1).
    Si te sale otro número, recuenta y corrige antes de escribir <FINAL>.
</protocol>"""
        else:
            # Fallback con UNA sola OCR: protocolo simple.
            protocol = """<protocol>
  Tienes UNA transcripción OCR del examen en el bloque <readings> arriba.
  Procede:

  PASO 1 — INVENTARIO: Cuenta las preguntas distintas. Declara: "He identificado
    K preguntas". K es el número.

  PASO 2 — RESOLUCIÓN POR ELIMINACIÓN (POE):
    Para cada pregunta: cita 1 línea, elimina las opciones obviamente
    incorrectas, escoge la más correcta. Si dudas, mejor apuesta razonada.

  PASO 3 — AUTO-VERIFICACIÓN: Cuenta las letras de <FINAL>. Debe ser exactamente K.
</protocol>"""

        # Texto de cierre depende del bloque usado (<exam> vs <readings>)
        if use_fused:
            final_instruction = (
                "Aplica el <protocol> sobre las preguntas del <exam> y emite tu respuesta. "
                "Termina obligatoriamente con el bloque <FINAL>...</FINAL> definido "
                "en el formato de respuesta."
            )
        else:
            final_instruction = (
                "Aplica el <protocol> sobre las <readings> y emite tu respuesta. "
                "Termina obligatoriamente con el bloque <FINAL>...</FINAL> definido "
                "en el formato de respuesta."
            )

        # Bloque opcional de contexto de internet (Tavily). Si vacío se omite,
        # así el orden del prompt es idéntico al original. Va DESPUÉS del OCR y
        # ANTES de las instrucciones — el modelo lee primero los datos crudos
        # del examen, luego la info de internet, luego qué hacer.
        internet_section = (f"{tavily_block}\n\n" if tavily_block else "")
        prompt_enriched = (
            # 1) DATOS PRIMERO (best practice multi-doc Anthropic)
            f"{ocr_for_prompt}\n\n"
            # 1.5) Contexto de internet (Tavily) — referencia secundaria
            f"{internet_section}"
            # 2) Rol e instrucciones generales (vienen del Android SYSTEM_PROMPT)
            f"{req.prompt}\n\n"
            # 3) Protocolo específico para este job (con POE + self-check)
            f"{protocol}\n\n"
            # 4) Query final corta
            f"{final_instruction}"
        )

        # Si fusionamos, override expected_questions con n_fused_questions cuando
        # el cliente no lo especificó. Esto permite que align_answer/fusionar
        # usen el conteo correcto para clampear respuestas. Si el cliente
        # especificó expected_questions, RESPETAMOS su valor (es la fuente de
        # verdad cuando el caller la conoce).
        effective_expected = req.expected_questions
        if use_fused and (not effective_expected or effective_expected <= 0):
            effective_expected = n_fused_questions
            # También actualizar el job para que fusionar() (que lee
            # job.get("expected_questions")) use el conteo correcto. Sin esto,
            # la fusión por votación caería al método de "moda ponderada" que es
            # menos robusto cuando una IA responde con N letras erróneas.
            try:
                with LOCK:
                    j = JOBS.get(job_id)
                    if j is not None and not j.get("expected_questions"):
                        j["expected_questions"]    = n_fused_questions
                        j["expected_from_fusion"] = True
            except Exception:
                pass
        analytic_req = AskRequest(
            prompt = prompt_enriched,
            system = req.system,
            image_b64 = None,    # ya no se necesita imagen; analyzers procesan texto
            video_b64 = None,    # idem
            expected_questions = effective_expected,
            review_timeout_seconds = req.review_timeout_seconds,
        )

        # ──── Fase 2: lanzar los analyzers en paralelo ──────────────────────
        _set_job_phase(job_id, "analysis_running", {
            "analyzers": [n for n,_ in providers],
        })
        logger.info("🧠 Job %s · lanzando %d analyzers: %s",
                    job_id[:8], len(providers), [n for n,_ in providers])
        spawn_failures = 0
        for name, fn in providers:
            try:
                threading.Thread(target=call_ai_task, args=(job_id, name, fn, analytic_req),
                                 daemon=True, name=f"ia-{name}-{job_id[:6]}").start()
            except Exception as spawn_exc:
                # Defensive: spawn falló (raro, ocurre bajo presión OS o thread starvation).
                # Marcamos este analyzer como error para que la lógica de _publish_partial
                # ("todos fallaron") detecte la situación correctamente. Los demás
                # analyzers siguen lanzándose normalmente — fallo aislado.
                spawn_failures += 1
                logger.error("💥 No se pudo arrancar analyzer thread %s: %s",
                             name, spawn_exc)
                try:
                    _publish_partial(job_id, name, {
                        "status": "error", "ok": False,
                        "error": f"thread spawn falló: {spawn_exc}",
                        "provider": name, "ms": 0,
                    }, effective_expected or 0)
                except Exception as pub_exc:
                    logger.error("💥 No se pudo publicar error de spawn %s: %s",
                                 name, pub_exc)
        if spawn_failures == len(providers):
            logger.error("💥 Job %s · TODOS los analyzer spawns fallaron · marcando job error",
                         job_id[:8])
            # B14: garantía adicional: aunque _publish_partial debería haber
            # marcado el job como error si todos fallan, si los publish también
            # fallaron en cascada el job se queda en pending hasta el watchdog.
            # Aquí forzamos el error directamente para que la app móvil no se
            # quede esperando STUCK_TIMEOUT (15min).
            with LOCK:
                j = JOBS.get(job_id)
                if j and j.get("status") not in ("done", "error"):
                    j["status"]   = "error"
                    j["finished"] = time.time()
                    j["error"]    = j.get("error") or ("Todos los analyzer thread-spawns fallaron "
                                                       "(OS bajo presión). Reintenta en unos segundos.")
                    _set_job_phase_locked(j, "error_analysis")
            _mark_jobs_dirty()
    except Exception as exc:
        logger.error("💥 _run_video_pipeline[%s]: %s", job_id[:8], exc)
        traceback.print_exc(file=sys.stdout)
        try:
            with LOCK:
                j = JOBS.get(job_id)
                if j and j.get("status") == "pending":
                    j["status"]   = "error"
                    j["error"]    = f"Pipeline video falló: {exc}"
                    j["finished"] = time.time()
            _set_job_phase(job_id, "error_pipeline")
            _mark_jobs_dirty()
        except Exception:
            pass
    finally:
        # FIX OOM Render 512MB: forzar GC al terminar el pipeline. Liberamos
        # promptamente los buffers del stacking (peak ~100MB con uint8/float32
        # de Optical Flow Farneback) que el GC generacional no recolectaba
        # hasta el próximo ciclo (~10s después), demasiado tarde para evitar
        # OOM si llega otro /ask casi simultáneo. gc.collect() es O(N
        # objects); en un proceso con 50k objects tarda <50ms — despreciable.
        try:
            gc.collect()
        except Exception:
            pass


@app.post("/ask")
def ask(req: AskRequest, x_api_key: Optional[str] = Header(None)):
    check_auth(x_api_key)
    gc_jobs()
    jid     = uuid.uuid4().hex
    created = time.time()
    context_source = "none"
    # CONTEXTO STICKY server-side: el relay adjunta él mismo el diagrama del caso
    # práctico a las peticiones nuevas, sin depender de que el móvil lo reenvíe.
    #   • Si la petición TRAE su propia imagen-contexto → la respetamos y de paso
    #     refrescamos el sticky (el panel mostrará la más reciente).
    #   • Si NO la trae pero hay un sticky vigente → lo INYECTAMOS en `req` para
    #     que se adjunte a TODAS las IAs igual que si lo hubiera mandado el móvil.
    # Mutar `req` es seguro (ya se hace con fused_image_b64). NUNCA bloquea /ask.
    try:
        if req.context_image_b64:
            _set_sticky_context(req.context_image_b64, req.context_image_mime, jid)
            context_source = "mobile"
        else:
            _sticky_ctx = _get_sticky_context()
            if _sticky_ctx:
                req.context_image_b64  = _sticky_ctx["b64"]
                req.context_image_mime = _sticky_ctx.get("mime") or "image/jpeg"
                context_source = "sticky"
                logger.info("🖼️ Job %s · contexto sticky inyectado (server-side · %d KB · de job %s)",
                            jid[:8], len(_sticky_ctx["b64"]) * 3 // 4 // 1024,
                            (_sticky_ctx.get("source_job") or "?")[:8])
    except Exception as _ctx_exc:
        logger.warning("sticky-context inject falló (no crítico): %s", _ctx_exc)
    tout    = max(10, min(600, req.review_timeout_seconds))
    expected = req.expected_questions or 0
    # Resolver providers analíticos según el modo (IMAGEN excluye DeepSeek).
    providers = _providers_for_request(req)
    is_video  = bool(req.video_b64)
    # phase: estado intermedio fino, separado del 'status' (que se mantiene en
    # 'pending' hasta que el revisor cierre el job). Permite que el panel
    # muestre exactamente qué se está haciendo:
    #   IMAGEN  → 'analysis_running' (directo, sin OCR)
    #   VIDEO   → 'received' → 'ocr_running' → 'ocr_done' → 'analysis_running'
    #            → 'analysis_done' → 'awaiting_review' → 'done'
    initial_phase = "received" if is_video else "analysis_running"
    with LOCK:
        JOBS[jid] = {
            "id":                     jid,
            "status":                 "pending",
            "phase":                  initial_phase,
            "phase_history":          [{"phase": initial_phase, "t": created}],
            "created":                created,
            "review_timeout_seconds": tout,
            "review_deadline":        0,          # se actualiza al llegar la 1ª IA
            "expected_questions":     expected,
            "has_img":                bool(req.image_b64),
            "img":                    req.image_b64,
            "img_mime":               req.image_mime,
            "has_video":              is_video,
            # Almacenar el video base64 para poder visualizarlo desde el panel.
            # Se sirve vía /api/video/{job_id} igual que /api/image/.
            "video":                  req.video_b64 if is_video else None,
            "video_mime":             req.video_mime if is_video else None,
            "video_size_b64":         len(req.video_b64) if is_video and req.video_b64 else 0,
            # Diagrama de contexto (caso práctico). Cuando viene, se adjunta como
            # 2ª imagen a todos los analyzers/OCRs multimodales. Se sirve en
            # /api/context_image/{job_id} para el dashboard. has_context_image se
            # cachea para evitar tener que decodificar el b64 en cada poll.
            "has_context_image":      bool(req.context_image_b64),
            "context_source":         context_source,  # mobile | sticky | none
            "context_image":          req.context_image_b64,
            "context_image_mime":     req.context_image_mime,
            # Imagen "casi-perfecta" generada por stacking + opcional Topaz.
            # Se rellena en _run_video_pipeline en un hilo aparte tras la
            # pre-extracción de frames. Inicialmente None — has_fused_image=False
            # hasta que el background thread la genere.
            "has_fused_image":        False,
            "fused_image_b64":        None,
            "fused_image_mime":       None,
            "fused_image_source":     None,   # "stacking_local" | "topaz_wonder3"
            "fused_image_stack_ms":   0,
            "fused_image_topaz_ms":   0,
            "merged_answer":          "",
            "answer":                 "",
            "_touched":               False,
            "_timer_started":         False,
            # Lista de providers efectivos en este job (4 si video, 3 si imagen)
            "_providers":             [name for name, _ in providers],
            # Telemetría SIM (del cliente Android, opcional). El panel la usa para
            # mostrar qué SIM se usó y con qué cobertura se mandó la foto/video.
            # No viajan en _HEAVY_FIELDS → pasan por _strip_heavy_for_list a /api/jobs.
            "sim_operator":           req.sim_operator,
            "sim_dbm":                req.sim_dbm,
            "sim_slot":               req.sim_slot,
            "sim_summary":            req.sim_summary,
            "responses": {
                name: {"status": "waiting", "ok": None, "answer": "",
                       "provider": name, "ms": 0}
                for name, _ in providers
            },
        }
    logger.info("📨 /ask job=%s  is_video=%s  providers=%s  size_b64=%d  ctx=%s",
                jid[:8], is_video, [n for n,_ in providers],
                len(req.video_b64) if is_video and req.video_b64 else len(req.image_b64 or ""),
                context_source)
    _mark_jobs_dirty()
    if is_video:
        # Modo VIDEO: pipeline OCR → análisis. Corre en thread aparte para no
        # bloquear la respuesta HTTP de /ask (el cliente espera job_id en <1s).
        try:
            threading.Thread(target=_run_video_pipeline,
                             args=(jid, req, providers, expected),
                             daemon=True, name=f"video-pipe-{jid[:6]}").start()
        except Exception as spawn_exc:
            # Spawn del pipeline thread falló → marcar job como error y devolver
            # job_id igual (cliente verá error al pollear, caerá a Mode B).
            logger.error("💥 No se pudo arrancar video pipeline thread: %s", spawn_exc)
            with LOCK:
                j = JOBS.get(jid)
                if j:
                    j["status"]  = "error"
                    j["error"]   = f"No se pudo arrancar pipeline video: {spawn_exc}"
                    j["finished"] = time.time()
            _mark_jobs_dirty()
    else:
        # Modo IMAGEN: lanzamos analyzers directamente (sin DeepSeek, ver _providers_for_request)
        spawn_failures = 0
        for name, fn in providers:
            try:
                threading.Thread(target=call_ai_task, args=(jid, name, fn, req),
                                 daemon=True, name=f"ia-{name}-{jid[:6]}").start()
            except Exception as spawn_exc:
                # Defensive: spawn falló → marcamos este analyzer como error y
                # seguimos con los demás. Fallo aislado por proveedor.
                spawn_failures += 1
                logger.error("💥 No se pudo arrancar analyzer thread %s: %s",
                             name, spawn_exc)
                try:
                    _publish_partial(jid, name, {
                        "status": "error", "ok": False,
                        "error": f"thread spawn falló: {spawn_exc}",
                        "provider": name, "ms": 0,
                    }, expected)
                except Exception:
                    pass
        if spawn_failures == len(providers) and len(providers) > 0:
            logger.error("💥 Job %s · TODOS los analyzer spawns fallaron en /ask (IMAGEN)",
                         jid[:8])
            # B14: idem que el path de video — forzamos el error explícito.
            with LOCK:
                j = JOBS.get(jid)
                if j and j.get("status") not in ("done", "error"):
                    j["status"]   = "error"
                    j["finished"] = time.time()
                    j["error"]    = j.get("error") or "Todos los analyzer thread-spawns fallaron en /ask"
            _mark_jobs_dirty()
    return {"job_id": jid}


_PHASE_HISTORY_MAX = 200  # cap del array por job (A6)


def _set_job_phase_locked(j: dict, phase: str, extra: Optional[dict] = None) -> bool:
    """Variante de `_set_job_phase` que asume que el caller ya tiene LOCK.
    Usada para hacer la transición de status + phase ATÓMICAMENTE en
    `_publish_partial` y `auto_aprobar`, evitando la race A3 donde un
    `_set_job_phase` posterior pisaba la fase final con una intermedia.

    Devuelve True si se appendeó una entry al historial."""
    prev = j.get("phase")
    hist = j.setdefault("phase_history", [])
    if prev == phase:
        last = hist[-1] if hist else {}
        last_extra = {k: v for k, v in last.items() if k not in ("phase", "t")}
        if (extra or {}) == last_extra:
            return False
    j["phase"] = phase
    entry = {"phase": phase, "t": time.time()}
    if extra:
        entry.update(extra)
    hist.append(entry)
    if len(hist) > _PHASE_HISTORY_MAX:
        del hist[:-_PHASE_HISTORY_MAX]
    return True


def _set_job_phase(job_id: str, phase: str, extra: Optional[dict] = None):
    """Actualiza la phase del job + appendea al historial. Thread-safe.

    Idempotencia ampliada (B17): si la fase es la misma Y extra es idéntico al
    último entry del history, no appendeamos. Evita bloat cuando un caller
    publica la misma transición repetidamente con metadata equivalente.

    A6: cap del array a _PHASE_HISTORY_MAX para que no crezca sin techo.
    También acota el peso del snapshot de persistencia."""
    with LOCK:
        j = JOBS.get(job_id)
        if not j:
            return
        prev = j.get("phase")
        hist = j.setdefault("phase_history", [])
        if prev == phase:
            last = hist[-1] if hist else {}
            last_extra = {k: v for k, v in last.items() if k not in ("phase", "t")}
            if (extra or {}) == last_extra:
                return
        j["phase"] = phase
        entry = {"phase": phase, "t": time.time()}
        if extra:
            entry.update(extra)
        hist.append(entry)
        if len(hist) > _PHASE_HISTORY_MAX:
            del hist[:-_PHASE_HISTORY_MAX]
    logger.info("⏱ job=%s phase: %s → %s", job_id[:8], prev, phase)
    _mark_jobs_dirty()


@app.get("/result/{job_id}")
def get_result(job_id: str, x_api_key: Optional[str] = Header(None)):
    check_auth(x_api_key)
    with LOCK:
        job = dict(JOBS.get(job_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail="job no encontrado")
    # El móvil ve "pending" hasta que el revisor apruebe o el timer expire
    if job["status"] in ("pending", "awaiting_review"):
        return {"job_id": job_id, "status": "pending"}

    # Adjuntar correcciones pendientes y marcarlas como entregadas
    corrections_payload = []
    if job["status"] == "done":
        with CORRECTIONS_LOCK:
            pending = [c for c in CORRECTIONS if not c.get("delivered")]
            for c in pending:
                corrections_payload.append({
                    "source_job_id": c["source_job_id"],
                    "answer":        c["answer"],
                    "page":          c.get("page"),
                    "id":            c.get("id"),
                })
                c["delivered"]    = True
                c["delivered_at"] = time.time()
            # Limpieza: descartar entregadas con más de 1h
            now = time.time()
            CORRECTIONS[:] = [c for c in CORRECTIONS
                              if not c.get("delivered") or now - c.get("delivered_at", 0) < 3600]
        if corrections_payload:
            _save_corrections_to_file()
            logger.info("📤 Entregando %d corrección(es) al móvil", len(corrections_payload))

    resp = {k: v for k, v in job.items() if k not in _HEAVY_FIELDS}
    resp["job_id"] = job_id
    if corrections_payload:
        resp["corrections"] = corrections_payload

    # Si el cómplice marcó 🖼️ "Imagen en pregunta" y el relay cacheó la mejor
    # imagen disponible como diagrama de contexto, la incluimos en el response
    # para que el móvil la guarde. Pesa ~150-500 KB, solo se manda cuando el
    # cómplice lo pidió explícitamente. Lo enviamos en done Y en error porque
    # el cliente acepta respuestas merged-partial con error status — perder
    # el diagrama por un fallo cascada sería estúpido.
    if (resp.get("status") in ("done", "error")
            and job.get("context_image_provided_b64")):
        resp["context_image_b64"]  = job["context_image_provided_b64"]
        resp["context_image_mime"] = job.get("context_image_provided_mime") or "image/jpeg"

    # BUG-FIX (status=error con merged_answer útil): cuando el job terminó en
    # error pero `merged_answer` (o `answer`) contiene letras A/B/C/D/X reales,
    # etiquetamos provider="merged-partial". Caso típico: 5 analyzers fallaron
    # (rate limit cascade, keys mal) pero los 10 OCR sí transcribieron y el
    # último merge dejó un answer útil. Antes, el cliente Android descartaba
    # toda respuesta con status="error" → "fallback-fail-video" con answer
    # vacío aunque el dashboard mostrara respuesta válida.
    if resp.get("status") == "error":
        merged = (resp.get("answer") or resp.get("merged_answer") or "")
        if isinstance(merged, str) and any(c in "ABCDXabcdx" for c in merged):
            # Solo si no había un provider explícito (los analyzers que sí
            # contestaron ya escriben su provider; lo respetamos).
            if not resp.get("provider"):
                resp["provider"] = "merged-partial"
    return resp


@app.post("/correction/{job_id}")
def add_correction(job_id: str, req: CorrectionRequest, x_api_key: Optional[str] = Header(None)):
    """Encola una corrección retroactiva para un job ya enviado al móvil.
    Se entrega en el próximo poll del móvil."""
    check_editor_auth(x_api_key)
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job no encontrado")
        if job.get("status") not in ("done", "error"):
            raise HTTPException(
                status_code=400,
                detail=f"Solo se pueden corregir jobs cerrados (estado: {job.get('status')})"
            )
    clean = re.sub(r'[^ABCDXabcdx]', '', req.answer).upper()
    if not clean:
        raise HTTPException(status_code=400, detail="La corrección debe contener A/B/C/D/X")
    corr = {
        "id":            uuid.uuid4().hex,
        "source_job_id": job_id,
        "answer":        clean,
        "page":          req.page,
        "created":       time.time(),
        "delivered":     False,
    }
    # Cap absoluto defensivo: gc_jobs purga por TTL pero solo corre cada 60s.
    # Si por algún bug del panel se enviaran cientos de correcciones en ráfaga,
    # la lista crecería sin techo entre GCs. CORRECTIONS_MAX=500 deja margen
    # holgado para uso real (un examen tiene <50 preguntas) y evita unbounded
    # memory growth. Si se llega al cap, las MÁS antiguas no entregadas caen.
    CORRECTIONS_MAX = 500
    with CORRECTIONS_LOCK:
        CORRECTIONS.append(corr)
        if len(CORRECTIONS) > CORRECTIONS_MAX:
            # Conservar primero las NO entregadas (el móvil podría aún recogerlas);
            # de las entregadas, las más recientes. Si no hay no entregadas
            # suficientes para llegar al cap, descartamos las entregadas más viejas.
            undelivered = [c for c in CORRECTIONS if not c.get("delivered")]
            delivered   = [c for c in CORRECTIONS if c.get("delivered")]
            keep_undelivered = undelivered[-CORRECTIONS_MAX:]
            slots_left = max(0, CORRECTIONS_MAX - len(keep_undelivered))
            # Más recientes primero entre entregadas
            delivered_sorted = sorted(delivered,
                                      key=lambda c: c.get("delivered_at", 0),
                                      reverse=True)
            keep_delivered = delivered_sorted[:slots_left]
            CORRECTIONS[:] = keep_undelivered + keep_delivered
            logger.warning(
                "CORRECTIONS cap aplicado: %d → %d (undelivered=%d, delivered=%d)",
                len(undelivered) + len(delivered), len(CORRECTIONS),
                len(keep_undelivered), len(keep_delivered),
            )
    _save_corrections_to_file()
    logger.info("✏️ Corrección encolada: job=%s answer=%s page=%s",
                job_id[:8], clean, req.page)
    return {"ok": True, "correction_id": corr["id"], "answer": clean}


@app.patch("/result/{job_id}")
def edit_result(job_id: str, edit: EditRequest, x_api_key: Optional[str] = Header(None)):
    """El Cómplice envía la respuesta final → cierra el job.

    Si el cómplice marcó la casilla 🖼️ "Imagen en pregunta" (has_image=true),
    el relay elige la mejor imagen disponible y hace DOS cosas:
      1) la fija como CONTEXTO STICKY server-side (_set_sticky_context) → el
         propio relay la adjuntará a las /ask siguientes que no traigan la suya,
         sin depender de que el móvil la reenvíe;
      2) la incluye en la respuesta al móvil como `context_image_b64` (compat:
         el móvil que aún la guarde/reenvíe sigue funcionando).
    Las IAs responderán solo las preguntas nuevas usando el diagrama como
    referencia (IMAGEN 1).

    Prioridad para elegir la imagen de contexto:
      1. fused_image_b64    (stacking + Topaz Wonder 3, máxima calidad)
      2. extracted_frames[0] (top-1 Laplacian, frame nítido del video)
      3. img                (modo IMAGEN del cliente — burst original)
    """
    check_editor_auth(x_api_key)
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job no encontrado")
        if edit.answer is not None:
            clean = re.sub(r'[^ABCDXabcdx]', '', edit.answer).upper()
            if clean:
                job["answer"]      = clean
                job["_touched"]    = True
                job["status"]      = "done"
                job["_cancelled"]  = True   # señal a IAs en vuelo para abortar
                job["reviewed_at"] = time.time()
                _became_done = True
            else:
                _became_done = False
        else:
            _became_done = False
        # Si el cómplice marcó la casilla, seleccionar la mejor imagen disponible
        # AHORA (antes de soltar el LOCK) y dejarla cacheada como `context_image_*`
        # del PROPIO job — así si el móvil hace GET /result tras un poll también
        # la encuentra. Si no había nada decente (job sin video, sin frames, sin img),
        # has_image=true sin imagen disponible se ignora silenciosamente.
        ctx_b64_payload: Optional[str] = None
        ctx_mime_payload: str = "image/jpeg"
        if edit.has_image is not None:
            job["has_image"] = edit.has_image
            if edit.has_image:
                # Selección por prioridad (1: fused, 2: frame top-1, 3: img burst).
                fused = job.get("fused_image_b64")
                if fused:
                    ctx_b64_payload  = fused
                    ctx_mime_payload = job.get("fused_image_mime") or "image/jpeg"
                else:
                    ef = job.get("extracted_frames")
                    if isinstance(ef, list) and ef and isinstance(ef[0], (list, tuple)) and ef[0]:
                        ctx_b64_payload  = ef[0][0]
                        ctx_mime_payload = ef[0][1] if len(ef[0]) >= 2 else "image/jpeg"
                    elif job.get("img"):
                        ctx_b64_payload  = job["img"]
                        ctx_mime_payload = job.get("img_mime") or "image/jpeg"
                # Cap defensivo: si la imagen seleccionada pesa >3 MB (12 MB en
                # base64), preferimos NO mandarla — saturaría la ventana de
                # radio del móvil. Caso patológico: Topaz devuelve PNG full-res.
                # Mejor que el móvil siga con su contexto anterior o sin él.
                if ctx_b64_payload and len(ctx_b64_payload) > 12 * 1024 * 1024:
                    logger.warning("🗂️ ctx-image demasiado grande (%d KB) → no se envía",
                                   len(ctx_b64_payload) * 3 // 4 // 1024)
                    ctx_b64_payload = None
                if ctx_b64_payload:
                    # Marcar que este job es "fuente de contexto" para que GET /result
                    # lo siga devolviendo si el móvil hace polls extra (Modo B
                    # fetchResult retry, oneShot, etc.). Tiene TTL del propio job.
                    job["context_image_provided_b64"]  = ctx_b64_payload
                    job["context_image_provided_mime"] = ctx_mime_payload
                    # Flag LIGERO (bool) para el panel: sobrevive a _strip_heavy_for_list
                    # (el _b64 es heavy y se filtra). Permite mostrar en el dashboard
                    # "imagen-contexto enviada al móvil" sin reenviar el blob en cada poll.
                    job["context_image_provided"]      = True
                    # CONTEXTO STICKY: marcar la casilla 🖼️ es la fuente AUTORITATIVA
                    # del diagrama. El relay lo recuerda y lo adjunta él mismo a las
                    # /ask siguientes (no depende de que el móvil lo reenvíe).
                    _set_sticky_context(ctx_b64_payload, ctx_mime_payload, job_id)
                    logger.info("🗂️ has_image=true · job=%s · imagen-contexto cacheada (%d KB · source=%s)",
                                job_id[:8],
                                len(ctx_b64_payload) * 3 // 4 // 1024,
                                "fused" if job.get("fused_image_b64") else (
                                    "frame" if job.get("extracted_frames") else "img"))
    _mark_jobs_dirty()
    if _became_done:
        # El revisor cerró el job manualmente → cancelar el timer de auto-aprobado
        # pendiente. auto_aprobar YA es idempotente (ve status!=awaiting_review y
        # retorna), así que esto NO arregla corrupción; solo libera el hilo
        # threading.Timer ocioso en vez de dejarlo vivo hasta la deadline —
        # consistente con gc_jobs / /reset, que también lo cancelan.
        _cancel_auto_approve_timer(job_id)
        _set_job_phase(job_id, "done", {"reviewed": True})
    resp = {"ok": True, "job_id": job_id,
            "answer": job.get("answer"), "status": job.get("status")}
    # Si hay imagen-contexto, la añadimos al payload del PATCH para que el móvil
    # la pille sin necesidad de un GET adicional. Pesa ~150-500 KB extra solo
    # cuando el cómplice marca la casilla.
    if ctx_b64_payload:
        resp["context_image_b64"]  = ctx_b64_payload
        resp["context_image_mime"] = ctx_mime_payload
    return resp


class ResponseEditRequest(BaseModel):
    """Body para editar la respuesta de UN provider en concreto del historial."""
    # max_length=200 igual que EditRequest/CorrectionRequest: defensa contra
    # payloads gigantes que harían el re.sub() del sanitize chew CPU.
    answer: str = Field(..., max_length=200)  # letras ABCDX, se sanea


@app.patch("/api/response/{job_id}/{provider}")
def edit_provider_response(
    job_id: str, provider: str, req: ResponseEditRequest,
    x_api_key: Optional[str] = Header(None),
):
    """Edita la respuesta de UNA IA en concreto en un job ya cerrado y re-calcula
    la fusión. NO afecta a la respuesta final enviada al móvil (esa ya se envió);
    sirve para auditar el historial y corregir manualmente lo que dijo cada IA
    cuando se revisan jobs viejos.

    Si quieres cambiar la respuesta enviada al móvil, usa PATCH /result/{job_id}
    (jobs en awaiting_review) o POST /correction/{job_id} (jobs done/error)."""
    check_editor_auth(x_api_key)
    clean = re.sub(r'[^ABCDXabcdx]', '', req.answer).upper()
    if not clean:
        raise HTTPException(status_code=400, detail="answer debe contener A/B/C/D/X")
    # ── (1) Sección crítica CORTA: mutar la entry + snapshot para fusionar ──
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job no encontrado")
        responses = job.setdefault("responses", {})
        if provider not in responses:
            raise HTTPException(
                status_code=404,
                detail=f"provider '{provider}' no existe en este job. "
                       f"Válidos: {list(responses.keys())}",
            )
        entry = responses[provider]
        old = entry.get("answer", "")
        entry["answer"]    = clean
        entry["status"]    = "done"
        entry["ok"]        = True
        entry["edited"]    = True
        entry["edited_at"] = time.time()
        expected = job.get("expected_questions") or 0
        # Snapshot superficial para fusionar() FUERA del lock. fusionar() puede
        # disparar el meta-judge (llamada HTTP de hasta 60s); correrla bajo el
        # LOCK global congelaba TODO el polling de clientes y panel durante ese
        # tiempo. Mismo patrón que _publish_partial.
        fusion_snapshot = {
            "responses":          dict(responses),
            "expected_questions": job.get("expected_questions"),
            "_providers":         list(job.get("_providers") or []),
            "ocr_fused_text":     job.get("ocr_fused_text"),
            # Edición manual del operador: el job ya pasó awaiting_review hace rato
            # (timer armado), así que SÍ queremos el meta-judge en la re-fusión.
            # Lo marcamos explícito para el gate temporal de fusionar().
            "_timer_started":     True,
        }
    # ── (2) Re-fusión FUERA del lock (puede tardar ~60s si corre el meta-judge) ──
    new_merged = ""
    try:
        new_merged = fusionar(fusion_snapshot)
        if expected > 0 and new_merged:
            if len(new_merged) < expected:
                new_merged += 'X' * (expected - len(new_merged))
            elif len(new_merged) > expected:
                new_merged = new_merged[:expected]
    except Exception as exc:
        logger.error("Re-fusión tras edición falló: %s", exc)
        new_merged = ""
    # ── (3) Escribir el merged bajo lock (sección corta) ──
    with LOCK:
        job = JOBS.get(job_id)
        if job is not None and new_merged:
            job["merged_answer"] = new_merged
        final_merged = (job or {}).get("merged_answer", "")
    _mark_jobs_dirty()
    logger.info("✏️ Edit response: job=%s provider=%s %s → %s",
                job_id[:8], provider, old, clean)
    return {
        "ok": True, "job_id": job_id, "provider": provider,
        "old": old, "new": clean,
        "merged_answer": final_merged,
    }


@app.get("/api/jobs")
def api_jobs(key: str = "", limit: int = 25):
    if not _is_authorized_key(key):
        raise HTTPException(status_code=401, detail="Acceso denegado")
    limit = max(1, min(150, limit))  # tope sano
    with LOCK:
        jobs = list(JOBS.values())
    jobs.sort(key=lambda x: x.get("created", 0), reverse=True)
    # Listado ligero: excluye img/video Y los textos largos por-provider
    # (raw de cada IA, transcripción OCR completa). El expediente del job
    # carga estos detalles on-demand vía /api/partial/{job_id} al expandir.
    # Ahorra ~5-10× bandwidth en el polling de 1.5s.
    return [_strip_heavy_for_list(j) for j in jobs[:limit]]


@app.get("/api/partial/{job_id}")
def api_partial(job_id: str, key: str = ""):
    """Estado parcial para polling rápido del panel. Incluye phase + phase_history
    para que el panel pueda mostrar el progreso del pipeline en tiempo real.

    FIX 502: jobs con muchos raws/textos grandes (6 analyzers × ~20K raw + 9
    OCRs × ~50K text) producían JSON de ~500KB que, combinado con Render free
    tier cold start, daba proxy timeout 502 al hacer click en "Ver/Editar".
    Aplicamos cap defensivo igual al de Supabase: raw truncado a 20K, OCR
    text truncado a 20K. Es suficiente para el dashboard (preview + diagnóstico)
    sin saturar el proxy.
    """
    if not _is_authorized_key(key):
        raise HTTPException(status_code=401, detail="Acceso denegado")
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404)
        # Copia shallow dentro del lock; el truncado va fuera para no
        # bloquear otras operaciones mientras se procesan los strings.
        shallow = dict(job)
    out: dict = {}
    PARTIAL_MAX_RAW   = 20_000
    PARTIAL_MAX_OCR   = 20_000
    PARTIAL_MAX_TAVILY = 8_000
    for k, v in shallow.items():
        if k in _HEAVY_FIELDS:
            continue
        if k == "responses" and isinstance(v, dict):
            slim = {}
            for prov, r in v.items():
                if not isinstance(r, dict):
                    slim[prov] = r
                    continue
                rr = dict(r)
                raw = rr.get("raw")
                if isinstance(raw, str) and len(raw) > PARTIAL_MAX_RAW:
                    rr["raw"] = raw[:PARTIAL_MAX_RAW]
                    rr["raw_truncated_from"] = len(raw)
                slim[prov] = rr
            out[k] = slim
            continue
        if k == "ocr_results" and isinstance(v, dict):
            slim = {}
            for prov, r in v.items():
                if not isinstance(r, dict):
                    slim[prov] = r
                    continue
                rr = dict(r)
                txt = rr.get("text")
                if isinstance(txt, str) and len(txt) > PARTIAL_MAX_OCR:
                    rr["text"] = txt[:PARTIAL_MAX_OCR]
                    rr["text_truncated_from"] = len(txt)
                slim[prov] = rr
            out[k] = slim
            continue
        if k == "tavily_block" and isinstance(v, str) and len(v) > PARTIAL_MAX_TAVILY:
            out[k] = v[:PARTIAL_MAX_TAVILY]
            out["tavily_block_truncated_from"] = len(v)
            continue
        out[k] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# REVISIÓN POR PREGUNTA (cards de corrección del OCR fusionado)
# ─────────────────────────────────────────────────────────────────────────────
# El revisor (cómplice) corrige el OCR fusionado pregunta-a-pregunta durante la
# ventana awaiting_review: marca OK, marca "mal leída" (→ X), borra preguntas
# alucinadas y reordena. Esas marcas RECOMPONEN la cadena final job["answer"]
# (lo que recibe el móvil): borradas fuera, mal-leídas → X, en el orden final.
#
# DISEÑO: no persistimos `review_questions` (se reconstruye al vuelo desde
# ocr_fused_text, que está congelado en awaiting_review); solo `review_state`
# (las marcas, minúsculo). Identidad por `rid` = posición original de la
# pregunta en el OCR (0-based), estable para mapear votos de IA y la cadena.

def _build_review_questions(job: dict) -> list:
    """Lista de preguntas para revisión a partir del OCR fusionado del job.
    Cada item: {rid, num, section, question, options, ia_answer, votes}.
      - rid: posición 0-based en ocr_fused_text (id estable dentro del job).
      - ia_answer: letra de la fusión (merged_answer[rid]) — sugerencia inicial.
      - votes: {provider: letra} con el voto de cada IA en esa posición.
    NUNCA lanza."""
    # Si la fusión OCR NO se usó (use_fused=False → las IAs respondieron sobre
    # <readings> crudas, NO sobre ocr_fused_text), los votos no se alinean por
    # posición con el texto fusionado → no construimos cards (se corrompería la
    # respuesta). El revisor usa el editor de letras clásico en ese caso.
    if not job.get("ocr_fusion_used"):
        return []
    merged = job.get("merged_answer") or job.get("answer") or ""
    responses = job.get("responses") or {}
    provs = job.get("_providers") or ["gpt", "claude", "gemini", "deepseek", "mistral", "nvidia"]
    vote_strings: dict = {}
    for prov in provs:
        r = responses.get(prov)
        if isinstance(r, dict) and r.get("ok") and isinstance(r.get("answer"), str):
            vote_strings[prov] = r["answer"]

    def _card(pos: int, num, section, question, options) -> dict:
        ia_letter = merged[pos] if 0 <= pos < len(merged) else "X"
        votes = {}
        for prov, s in vote_strings.items():
            if 0 <= pos < len(s):
                votes[prov] = s[pos]
        return {
            "rid":      pos,
            "num":      num,
            "section":  section or "",
            "question": question or "",
            "options":  options or {},
            "ia_answer": ia_letter,
            "votes":    votes,
        }

    # FUENTE PRIMARIA: cluster_votes de la fusión. Cada entrada == 1 pregunta,
    # en el MISMO orden que merged_answer y que las cadenas de respuesta de las
    # IAs (todas se recorrieron sobre fused_text bloque-a-bloque). Por tanto
    # rid = índice da alineación EXACTA. NO re-parseamos ocr_fused_text: el
    # re-parseo reasignaba `pos` y, si _parse_ocr_text fusionaba/descartaba un
    # bloque (filtro anti-fantasma, num "?", enunciado corto), desplazaba la
    # letra de TODA pregunta posterior → cards y respuesta final corruptas.
    cvs = (job.get("ocr_fusion_stats") or {}).get("cluster_votes")
    if isinstance(cvs, list) and cvs and all(
        isinstance(c, dict) and "q_text" in c for c in cvs
    ):
        return [_card(i, c.get("num"), c.get("section"),
                      c.get("q_text"), c.get("opts_text"))
                for i, c in enumerate(cvs)]

    # FALLBACK (snapshot viejo sin q_text, o cluster_votes ausente tras recarga
    # con cap agresivo): re-parsear ocr_fused_text. SOLO construimos cards si el
    # recuento re-parseado coincide con el de la fusión (cluster_votes); si no
    # coincide o no hay con qué verificar, devolvemos [] → el revisor usa el
    # editor de letras clásico. Preferimos no mostrar cards a mostrarlas
    # desalineadas (corromperían la respuesta enviada al móvil).
    if not (isinstance(cvs, list) and cvs):
        return []
    try:
        parsed = _parse_ocr_text(job.get("ocr_fused_text") or "")
    except Exception:
        parsed = []
    if len(parsed) != len(cvs):
        return []
    return [_card(i, q.get("num"), q.get("section"),
                  q.get("question"), q.get("options"))
            for i, q in enumerate(parsed)]


def _default_review_state(rqs: list) -> dict:
    return {
        "answer":  {},                       # rid → letra (override del revisor)
        "state":   {},                       # rid → "ok"|"bad"
        "deleted": [],                        # [rid]
        "order":   [q["rid"] for q in rqs],   # orden visual
        "updated": 0,
    }


def _try_int(x) -> Optional[int]:
    """int(x) o None si no es convertible (no lanza)."""
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def _norm_int_keys(d) -> dict:
    """Normaliza claves de dict a int (JSON las trae como str)."""
    out = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except (ValueError, TypeError):
            continue
    return out


def _recompose_review_answer(job: dict, rqs: Optional[list] = None) -> str:
    """Cadena final A/B/C/D/X a partir de las marcas del revisor: recorre las
    preguntas vivas (no borradas) en el orden final; mal-leídas → X; el resto
    usa el override del revisor o la sugerencia de la fusión. NUNCA lanza."""
    try:
        if rqs is None:
            rqs = _build_review_questions(job)
        rs = job.get("review_state") or {}
        by_rid = {q["rid"]: q for q in rqs}
        answers = _norm_int_keys(rs.get("answer"))
        states = _norm_int_keys(rs.get("state"))
        deleted = set()
        for x in (rs.get("deleted") or []):
            try:
                deleted.add(int(x))
            except (ValueError, TypeError):
                continue
        order = []
        for x in (rs.get("order") or []):
            try:
                order.append(int(x))
            except (ValueError, TypeError):
                continue
        if not order:
            order = [q["rid"] for q in rqs]
        # Garantizar que toda pregunta viva aparezca aunque falte en `order`.
        for q in rqs:
            if q["rid"] not in order:
                order.append(q["rid"])
        out = []
        for rid in order:
            if rid in deleted:
                continue
            if rid < 0:                  # hueco insertado por el revisor (pregunta
                out.append("X")          # que el OCR se saltó) → X, para realinear
                continue                 # la cadena con las preguntas físicas
            q = by_rid.get(rid)
            if q is None:
                continue
            if states.get(rid) == "bad":
                out.append("X")
                continue
            letter = str(answers.get(rid) or q.get("ia_answer") or "X").upper()
            out.append(letter if letter in ("A", "B", "C", "D", "X") else "X")
        return "".join(out)
    except Exception as exc:
        logger.warning("[review] recompose falló (se conserva answer actual): %s", exc)
        return job.get("answer") or ""


@app.get("/api/review/{job_id}")
def api_review_get(job_id: str, key: str = ""):
    """Devuelve las preguntas del OCR fusionado + el estado de revisión + la
    cadena recompuesta. El panel pinta una card por pregunta en awaiting_review."""
    if not _is_authorized_key(key):
        raise HTTPException(status_code=401, detail="Acceso denegado")
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job no encontrado")
        rqs = _build_review_questions(job)
        rs = job.get("review_state") or _default_review_state(rqs)
        answer_now = _recompose_review_answer(job, rqs)
        return {
            "ok":                     True,
            "job_id":                 job_id,
            "status":                 job.get("status"),
            "review_deadline":        job.get("review_deadline", 0),
            "review_timeout_seconds": job.get("review_timeout_seconds", 90),
            "questions":              rqs,
            "state":                  rs,
            "answer":                 answer_now,
        }


class ReviewStateRequest(BaseModel):
    """Estado completo de revisión (idempotente: se manda entero en cada cambio).
    Tipos laxos (dict/list) — se sanean en el handler contra los rids válidos."""
    answer:  dict = Field(default_factory=dict)   # {rid: "A".."D"|"X"}
    state:   dict = Field(default_factory=dict)   # {rid: "ok"|"bad"}
    deleted: list = Field(default_factory=list)   # [rid]
    order:   list = Field(default_factory=list)   # [rid] orden visual


@app.patch("/api/review/{job_id}")
def api_review_patch(job_id: str, req: ReviewStateRequest,
                     x_api_key: Optional[str] = Header(None)):
    """Guarda el estado de revisión y RECOMPONE job["answer"] (lo que va al
    móvil). Marca _touched para que una IA tardía no pise la corrección. Si la
    ventana ya cerró (auto-aprobado / enviado), guarda las marcas pero NO repisa
    la answer ya entregada (devuelve closed=true)."""
    check_editor_auth(x_api_key)

    def _norm_letter(v) -> str:
        s = str(v or "").upper().strip()
        return s if s in ("A", "B", "C", "D", "X") else "X"

    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job no encontrado")
        rqs = _build_review_questions(job)
        valid = {q["rid"] for q in rqs}
        # Sanear el estado recibido contra los rids válidos.
        answer = {rid: _norm_letter(v) for k, v in (req.answer or {}).items()
                  if (rid := _try_int(k)) is not None and rid in valid}
        state = {}
        for k, v in (req.state or {}).items():
            rid = _try_int(k)
            sv = str(v or "").lower().strip()
            if rid is not None and rid in valid and sv in ("ok", "bad"):
                state[rid] = sv
        deleted = [rid for x in (req.deleted or [])
                   if (rid := _try_int(x)) is not None and rid in valid]
        # order acepta rids reales (en valid) Y huecos del revisor (rid<0): los
        # huecos recomponen como "X" para realinear cuando el OCR saltó preguntas.
        order = [rid for x in (req.order or [])
                 if (rid := _try_int(x)) is not None and (rid in valid or rid < 0)]
        for q in rqs:                                    # completar los que falten
            if q["rid"] not in order:
                order.append(q["rid"])
        # Guardamos con claves str para serializar limpio a JSON/Supabase.
        job["review_state"] = {
            "answer":  {str(k): v for k, v in answer.items()},
            "state":   {str(k): v for k, v in state.items()},
            "deleted": deleted,
            "order":   order,
            "updated": time.time(),
        }
        new_answer = _recompose_review_answer(job, rqs)
        st = job.get("status")
        closed = st not in ("awaiting_review", "pending")
        # Solo escribimos job["answer"] (lo que va al móvil) si YA hay fusión base
        # de las IAs. En 'pending' sin IAs aún, merged_answer está vacío → recomponer
        # daría "XXXX" y _touched=True BLOQUEARÍA que las IAs actualicen la respuesta
        # (las marcas se guardan igual en review_state y se aplican en cuanto el
        # revisor edita con la fusión ya presente — el PATCH manda el estado completo).
        has_base = bool((job.get("merged_answer") or "").strip())
        # No sobreescribir con cadena VACÍA (p.ej. el revisor borró TODAS las
        # preguntas, o la recomposición dio ""): dejar la fusión merged_answer
        # intacta para que el móvil NUNCA reciba una respuesta vacía. En cuanto
        # quede ≥1 pregunta viva, new_answer tiene contenido y se aplica.
        applied = (not closed) and has_base and bool(new_answer.strip())
        if applied:
            job["answer"]      = new_answer
            job["_touched"]    = True
            job["reviewed_at"] = time.time()
    _mark_jobs_dirty()
    return {"ok": applied, "closed": closed, "job_id": job_id, "answer": new_answer}


@app.get("/api/image/{job_id}")
def api_image(job_id: str, key: str = ""):
    if not _is_authorized_key(key):
        raise HTTPException(status_code=401, detail="Acceso denegado")
    with LOCK:
        job = JOBS.get(job_id)
    if not job or not job.get("img"):
        raise HTTPException(status_code=404)
    try:
        data = base64.b64decode(job["img"], validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen inválida")
    return Response(content=data, media_type=job.get("img_mime", "image/jpeg"))


@app.get("/api/video/{job_id}")
def api_video(job_id: str, key: str = ""):
    """Sirve el video MP4 base64 almacenado en el job. El panel lo usa para
    incrustar un <video> que reproduzca lo que se envió a OCR."""
    if not _is_authorized_key(key):
        raise HTTPException(status_code=401, detail="Acceso denegado")
    with LOCK:
        job = JOBS.get(job_id)
    if not job or not job.get("video"):
        raise HTTPException(status_code=404, detail="Video no disponible")
    try:
        data = base64.b64decode(job["video"], validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Video inválido")
    return Response(content=data, media_type=job.get("video_mime", "video/mp4"))


@app.get("/api/fused_image/{job_id}")
def api_fused_image(job_id: str, key: str = "", variant: str = "best"):
    """Sirve la imagen fusionada del video. `variant`:
      - "best" (default): Topaz Wonder 3 si existió, si no stacking local.
      - "local": fuerza la versión stacking sin Topaz (para comparar en panel).
    Generada por el background thread _fuse_pipeline_bg tras pre-extraer frames.
    """
    if not _is_authorized_key(key):
        raise HTTPException(status_code=401, detail="Acceso denegado")
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    if variant == "local":
        b64  = job.get("fused_image_local_b64") or job.get("fused_image_b64")
        mime = job.get("fused_image_local_mime") or job.get("fused_image_mime") or "image/jpeg"
    else:
        b64  = job.get("fused_image_b64")
        mime = job.get("fused_image_mime") or "image/jpeg"
    if not b64:
        raise HTTPException(status_code=404, detail="Imagen fusionada no disponible (aún generándose o sin video)")
    try:
        data = base64.b64decode(b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen fusionada inválida")
    return Response(content=data, media_type=mime)


@app.get("/api/context_image/{job_id}")
def api_context_image(job_id: str, key: str = ""):
    """Sirve la imagen de contexto (diagrama del caso práctico) que el cliente
    adjuntó en /ask. Igual que /api/image pero sobre `job["context_image"]`.
    El panel la muestra junto a la foto/video "buena" para que el operador vea
    exactamente qué se mandó como referencia a los analyzers."""
    if not _is_authorized_key(key):
        raise HTTPException(status_code=401, detail="Acceso denegado")
    with LOCK:
        job = JOBS.get(job_id)
    if not job or not job.get("context_image"):
        raise HTTPException(status_code=404, detail="Sin diagrama de contexto")
    try:
        data = base64.b64decode(job["context_image"], validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Diagrama inválido")
    return Response(content=data,
                    media_type=job.get("context_image_mime", "image/jpeg"))


@app.get("/api/frame/{job_id}/{idx}")
def api_frame(job_id: str, idx: int, key: str = ""):
    """Sirve el N-ésimo frame extraído del video (los mismos que se envían a
    los OCR de imagen: Mistral OCR, DeepSeek-OCR, GLM-OCR, Claude, GPT-4o).

    `idx` es 0-based. El número total de frames disponibles está en
    `job.extracted_frames_count` (devuelto por /result y /api/jobs).

    Se sirven como image/jpeg directamente desde el b64 cacheado en el job,
    sin decodificar el video de nuevo. Los frames se eligieron por Laplacian
    variance con diversidad temporal (top-3 más nítidos del clip)."""
    if not _is_authorized_key(key):
        raise HTTPException(status_code=401, detail="Acceso denegado")
    with LOCK:
        job = JOBS.get(job_id)
        frames = job.get("extracted_frames") if job else None
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    if not frames or not isinstance(frames, list):
        raise HTTPException(status_code=404, detail="Frames no extraídos (job sin video o ya purgado)")
    if idx < 0 or idx >= len(frames):
        raise HTTPException(status_code=404, detail=f"Frame {idx} fuera de rango (0..{len(frames)-1})")
    entry = frames[idx]
    if not (isinstance(entry, (list, tuple)) and len(entry) >= 1):
        raise HTTPException(status_code=500, detail="Frame mal formado")
    b64 = entry[0]
    mime = entry[1] if len(entry) >= 2 else "image/jpeg"
    try:
        data = base64.b64decode(b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Frame inválido")
    return Response(content=data, media_type=mime)


def _mask(s: str) -> str:
    """Devuelve la key truncada para mostrar en UI sin exponer toda la cadena."""
    if not s: return ""
    if len(s) <= 12: return s
    return s[:6] + "…" + s[-4:]


@app.post("/api/config/refresh")
def refresh_config(x_api_key: Optional[str] = Header(None)):
    """Fuerza un reload inmediato de la config desde las DBs (sin esperar al
    sync_loop de 60s). Útil si dos relays están desincronizados o tras un
    incidente de DB. Devuelve cuántas claves se refrescaron y de qué backend."""
    check_editor_auth(x_api_key)
    global _SUPABASE_STALE
    # Si estábamos en modo stale, este refresh es una buena oportunidad para
    # intentar Supabase de nuevo (si el operador lo dispara es porque cree que
    # ya está sano). Pero si falla, se vuelve a marcar como stale.
    n_supa = _supabase_load_config()
    n_appw = 0
    source = ""
    if n_supa > 0:
        source = "supabase"
        _SUPABASE_STALE = False
    elif _appwrite_enabled():
        n_appw = _appwrite_load_config()
        if n_appw > 0:
            source = "appwrite"
    logger.info("🔄 /api/config/refresh: source=%s keys=%d", source or "none", n_supa or n_appw)
    return {
        "ok":          bool(source),
        "source":      source or None,
        "keys_loaded": n_supa or n_appw,
        "supabase_stale": _SUPABASE_STALE,
    }


@app.get("/api/config")
def get_config(x_api_key: Optional[str] = Header(None), key: str = ""):
    """Devuelve la config actual (con keys completas para pre-cargar el panel).
    Solo accesible con la clave del editor."""
    # Permitir auth por header O por query string (cómodo para el panel)
    auth_key = x_api_key or key
    if not _is_authorized_key(auth_key):
        raise HTTPException(status_code=401, detail="API key inválida")
    with CONFIG_LOCK:
        return {
            # API keys (primarias)
            "anthropic_key":        DYNAMIC_CONFIG["ANTHROPIC_API_KEY"],
            "openai_key":           DYNAMIC_CONFIG["OPENAI_API_KEY"],
            "gemini_key":           DYNAMIC_CONFIG["GEMINI_API_KEY"],
            "deepseek_key":         DYNAMIC_CONFIG.get("DEEPSEEK_API_KEY", ""),
            "mistral_key":          DYNAMIC_CONFIG.get("MISTRAL_API_KEY", ""),
            "kimi_key":             DYNAMIC_CONFIG.get("KIMI_API_KEY", ""),
            "qwen_key":             DYNAMIC_CONFIG.get("QWEN_API_KEY", ""),
            "mimo_key":             DYNAMIC_CONFIG.get("MIMO_API_KEY", ""),
            "nvidia_key":           DYNAMIC_CONFIG.get("NVIDIA_API_KEY", ""),
            "z_ai_key":             DYNAMIC_CONFIG.get("Z_AI_API_KEY", ""),
            "tavily_key":           DYNAMIC_CONFIG.get("TAVILY_API_KEY", ""),
            # API keys (backups)
            "anthropic_key_backup": DYNAMIC_CONFIG["ANTHROPIC_API_KEY_BACKUP"],
            "openai_key_backup":    DYNAMIC_CONFIG["OPENAI_API_KEY_BACKUP"],
            "gemini_key_backup":    DYNAMIC_CONFIG["GEMINI_API_KEY_BACKUP"],
            "deepseek_key_backup":  DYNAMIC_CONFIG.get("DEEPSEEK_API_KEY_BACKUP", ""),
            "mistral_key_backup":   DYNAMIC_CONFIG.get("MISTRAL_API_KEY_BACKUP", ""),
            "kimi_key_backup":      DYNAMIC_CONFIG.get("KIMI_API_KEY_BACKUP", ""),
            "qwen_key_backup":      DYNAMIC_CONFIG.get("QWEN_API_KEY_BACKUP", ""),
            "mimo_key_backup":      DYNAMIC_CONFIG.get("MIMO_API_KEY_BACKUP", ""),
            "nvidia_key_backup":    DYNAMIC_CONFIG.get("NVIDIA_API_KEY_BACKUP", ""),
            "z_ai_key_backup":      DYNAMIC_CONFIG.get("Z_AI_API_KEY_BACKUP", ""),
            "tavily_key_backup":    DYNAMIC_CONFIG.get("TAVILY_API_KEY_BACKUP", ""),
            # Modelos (analyzers de texto/imagen)
            "anthropic_model":      DYNAMIC_CONFIG["CLAUDE_MODEL"],
            "openai_model":         DYNAMIC_CONFIG["OPENAI_MODEL"],
            "gemini_model":         DYNAMIC_CONFIG["GEMINI_MODEL"],
            "deepseek_model":       DYNAMIC_CONFIG.get("DEEPSEEK_MODEL", DEEPSEEK_MODEL),
            "mistral_model":        DYNAMIC_CONFIG.get("MISTRAL_MODEL", MISTRAL_MODEL),
            "mimo_model":           DYNAMIC_CONFIG.get("MIMO_MODEL", MIMO_MODEL),
            # Modelos de OCR de video (fase 1)
            "qwen_video_model":     DYNAMIC_CONFIG.get("QWEN_VIDEO_MODEL", QWEN_VIDEO_MODEL),
            "gemini_video_model":   DYNAMIC_CONFIG.get("GEMINI_VIDEO_MODEL", GEMINI_VIDEO_MODEL),
            "kimi_video_model":     DYNAMIC_CONFIG.get("KIMI_VIDEO_MODEL", KIMI_VIDEO_MODEL),
            "mimo_video_model":     DYNAMIC_CONFIG.get("MIMO_VIDEO_MODEL", MIMO_VIDEO_MODEL),
            "claude_video_ocr_model": DYNAMIC_CONFIG.get("CLAUDE_VIDEO_OCR_MODEL", CLAUDE_VIDEO_OCR_MODEL),
            "openai_video_ocr_model": DYNAMIC_CONFIG.get("OPENAI_VIDEO_OCR_MODEL", OPENAI_VIDEO_OCR_MODEL),
            # NVIDIA Nemotron como analyzer de texto (fase 2)
            "nvidia_model":           DYNAMIC_CONFIG.get("NVIDIA_MODEL", NVIDIA_MODEL),
            # GPT-5.5 tuning agéntico
            "openai_reasoning_effort": DYNAMIC_CONFIG.get("OPENAI_REASONING_EFFORT", OPENAI_REASONING_EFFORT),
            "openai_max_tool_calls":   DYNAMIC_CONFIG.get("OPENAI_MAX_TOOL_CALLS",   OPENAI_MAX_TOOL_CALLS),
            "openai_allowed_domains":  DYNAMIC_CONFIG.get("OPENAI_ALLOWED_DOMAINS",  list(OPENAI_ALLOWED_DOMAINS)),
            # Tavily (paso intermedio OCR → razonamiento)
            "tavily_enabled":          DYNAMIC_CONFIG.get("TAVILY_ENABLED",          TAVILY_ENABLED),
            "tavily_max_results":      DYNAMIC_CONFIG.get("TAVILY_MAX_RESULTS",      TAVILY_MAX_RESULTS),
            "tavily_search_depth":     DYNAMIC_CONFIG.get("TAVILY_SEARCH_DEPTH",     TAVILY_SEARCH_DEPTH),
            "tavily_http_timeout_s":   DYNAMIC_CONFIG.get("TAVILY_HTTP_TIMEOUT_S",   TAVILY_HTTP_TIMEOUT_S),
            "tavily_total_deadline_s": DYNAMIC_CONFIG.get("TAVILY_TOTAL_DEADLINE_S", TAVILY_TOTAL_DEADLINE_S),
            # Meta-Judge (árbitro post-fusión, opción B "siempre" con fallback)
            "meta_judge_enabled":          DYNAMIC_CONFIG.get("META_JUDGE_ENABLED",          True),
            "meta_judge_model":            DYNAMIC_CONFIG.get("META_JUDGE_MODEL",            "gpt-5"),
            "meta_judge_reasoning_effort": DYNAMIC_CONFIG.get("META_JUDGE_REASONING_EFFORT", "high"),
            "meta_judge_timeout_s":        DYNAMIC_CONFIG.get("META_JUDGE_TIMEOUT_S",        60.0),
            # Pesos en la votación de los ANALYZERS (fase 2)
            "anthropic_weight":     DYNAMIC_CONFIG.get("ANTHROPIC_WEIGHT", 1),
            "openai_weight":        DYNAMIC_CONFIG.get("OPENAI_WEIGHT", 1),
            "gemini_weight":        DYNAMIC_CONFIG.get("GEMINI_WEIGHT", 1),
            "deepseek_weight":      DYNAMIC_CONFIG.get("DEEPSEEK_WEIGHT", 1),
            "mistral_weight":       DYNAMIC_CONFIG.get("MISTRAL_WEIGHT", 1),
            "nvidia_weight":        DYNAMIC_CONFIG.get("NVIDIA_WEIGHT", 1),
            "kimi_weight":          DYNAMIC_CONFIG.get("KIMI_WEIGHT", 1),
            "mimo_weight":          DYNAMIC_CONFIG.get("MIMO_WEIGHT", 0),
            # Pesos en la votación del fusionador OCR (fase 1)
            "mistral_ocr_weight":   DYNAMIC_CONFIG.get("MISTRAL_OCR_WEIGHT", 5),
            "gemini_ocr_weight":    DYNAMIC_CONFIG.get("GEMINI_OCR_WEIGHT", 4),
            "qwen_ocr_weight":      DYNAMIC_CONFIG.get("QWEN_OCR_WEIGHT", 4),
            "glm_ocr_weight":       DYNAMIC_CONFIG.get("GLM_OCR_WEIGHT", 4),
            "anthropic_ocr_weight": DYNAMIC_CONFIG.get("ANTHROPIC_OCR_WEIGHT", 3),
            "openai_ocr_weight":    DYNAMIC_CONFIG.get("OPENAI_OCR_WEIGHT", 3),
            "deepseek_ocr_weight":  DYNAMIC_CONFIG.get("DEEPSEEK_OCR_WEIGHT", 2),
            "kimi_ocr_weight":      DYNAMIC_CONFIG.get("KIMI_OCR_WEIGHT", 1),
            "mimo_ocr_weight":      DYNAMIC_CONFIG.get("MIMO_OCR_WEIGHT", 1),
            # Topaz Labs Image API (post-stacking enhancement opcional)
            "topaz_key":            DYNAMIC_CONFIG.get("TOPAZ_API_KEY", ""),
            "topaz_enabled":        DYNAMIC_CONFIG.get("TOPAZ_ENABLED", True),
            "topaz_model":          DYNAMIC_CONFIG.get("TOPAZ_MODEL", "Wonder 3"),
            "topaz_output_height":  DYNAMIC_CONFIG.get("TOPAZ_OUTPUT_HEIGHT", 0),
            "topaz_timeout_s":      DYNAMIC_CONFIG.get("TOPAZ_TIMEOUT_S", 60.0),
        }


def _safe_int(v, default: int, lo: int = None, hi: int = None) -> int:
    """C11: int() defensivo con clamp opcional. Tolera None, float, string,
    bool. Usado en update_config para evitar que un pydantic con coercion
    laxa (o el cliente mandando "high" en un campo int) tumbe la request."""
    try:
        n = int(v) if not isinstance(v, bool) else int(v)
    except (TypeError, ValueError):
        return default
    if lo is not None: n = max(lo, n)
    if hi is not None: n = min(hi, n)
    return n


@app.post("/api/config")
def update_config(req: ConfigUpdateRequest, x_api_key: Optional[str] = Header(None)):
    check_editor_auth(x_api_key)
    # Distinguir "no enviado" (None) de "limpiar" ("" string vacía)
    changes = []
    with CONFIG_LOCK:
        if req.anthropic_key        is not None:
            DYNAMIC_CONFIG["ANTHROPIC_API_KEY"]        = req.anthropic_key.strip()
            changes.append(f"anthropic={_mask(req.anthropic_key)}")
        if req.openai_key           is not None:
            DYNAMIC_CONFIG["OPENAI_API_KEY"]           = req.openai_key.strip()
            changes.append(f"openai={_mask(req.openai_key)}")
        if req.gemini_key           is not None:
            DYNAMIC_CONFIG["GEMINI_API_KEY"]           = req.gemini_key.strip()
            changes.append(f"gemini={_mask(req.gemini_key)}")
        if req.anthropic_key_backup is not None:
            DYNAMIC_CONFIG["ANTHROPIC_API_KEY_BACKUP"] = req.anthropic_key_backup.strip()
            changes.append(f"anthropic_backup={_mask(req.anthropic_key_backup)}")
        if req.openai_key_backup    is not None:
            DYNAMIC_CONFIG["OPENAI_API_KEY_BACKUP"]    = req.openai_key_backup.strip()
            changes.append(f"openai_backup={_mask(req.openai_key_backup)}")
        if req.gemini_key_backup    is not None:
            DYNAMIC_CONFIG["GEMINI_API_KEY_BACKUP"]    = req.gemini_key_backup.strip()
            changes.append(f"gemini_backup={_mask(req.gemini_key_backup)}")
        if req.deepseek_key is not None:
            DYNAMIC_CONFIG["DEEPSEEK_API_KEY"]         = req.deepseek_key.strip()
            changes.append(f"deepseek={_mask(req.deepseek_key)}")
        if req.deepseek_key_backup is not None:
            DYNAMIC_CONFIG["DEEPSEEK_API_KEY_BACKUP"]  = req.deepseek_key_backup.strip()
            changes.append(f"deepseek_backup={_mask(req.deepseek_key_backup)}")
        if req.qwen_key is not None:
            DYNAMIC_CONFIG["QWEN_API_KEY"]             = req.qwen_key.strip()
            changes.append(f"qwen={_mask(req.qwen_key)}")
        if req.qwen_key_backup is not None:
            DYNAMIC_CONFIG["QWEN_API_KEY_BACKUP"]      = req.qwen_key_backup.strip()
            changes.append(f"qwen_backup={_mask(req.qwen_key_backup)}")
        if req.kimi_key is not None:
            DYNAMIC_CONFIG["KIMI_API_KEY"]             = req.kimi_key.strip()
            changes.append(f"kimi={_mask(req.kimi_key)}")
        if req.kimi_key_backup is not None:
            DYNAMIC_CONFIG["KIMI_API_KEY_BACKUP"]      = req.kimi_key_backup.strip()
            changes.append(f"kimi_backup={_mask(req.kimi_key_backup)}")
        if req.mistral_key is not None:
            DYNAMIC_CONFIG["MISTRAL_API_KEY"]          = req.mistral_key.strip()
            changes.append(f"mistral={_mask(req.mistral_key)}")
        if req.mistral_key_backup is not None:
            DYNAMIC_CONFIG["MISTRAL_API_KEY_BACKUP"]   = req.mistral_key_backup.strip()
            changes.append(f"mistral_backup={_mask(req.mistral_key_backup)}")
        if req.mimo_key is not None:
            DYNAMIC_CONFIG["MIMO_API_KEY"]             = req.mimo_key.strip()
            changes.append(f"mimo={_mask(req.mimo_key)}")
        if req.mimo_key_backup is not None:
            DYNAMIC_CONFIG["MIMO_API_KEY_BACKUP"]      = req.mimo_key_backup.strip()
            changes.append(f"mimo_backup={_mask(req.mimo_key_backup)}")
        # Modelos: si llega vacío, se restaura el default hardcoded (fallback)
        if req.anthropic_model is not None:
            DYNAMIC_CONFIG["CLAUDE_MODEL"] = req.anthropic_model.strip() or CLAUDE_MODEL
            changes.append(f"claude_model={DYNAMIC_CONFIG['CLAUDE_MODEL']}")
        if req.openai_model is not None:
            DYNAMIC_CONFIG["OPENAI_MODEL"] = req.openai_model.strip() or OPENAI_MODEL
            changes.append(f"openai_model={DYNAMIC_CONFIG['OPENAI_MODEL']}")
        if req.gemini_model is not None:
            DYNAMIC_CONFIG["GEMINI_MODEL"] = req.gemini_model.strip() or GEMINI_MODEL
            changes.append(f"gemini_model={DYNAMIC_CONFIG['GEMINI_MODEL']}")
        if req.deepseek_model is not None:
            DYNAMIC_CONFIG["DEEPSEEK_MODEL"] = req.deepseek_model.strip() or DEEPSEEK_MODEL
            changes.append(f"deepseek_model={DYNAMIC_CONFIG['DEEPSEEK_MODEL']}")
        if req.mistral_model is not None:
            DYNAMIC_CONFIG["MISTRAL_MODEL"] = req.mistral_model.strip() or MISTRAL_MODEL
            changes.append(f"mistral_model={DYNAMIC_CONFIG['MISTRAL_MODEL']}")
        if req.mimo_model is not None:
            DYNAMIC_CONFIG["MIMO_MODEL"] = req.mimo_model.strip() or MIMO_MODEL
            changes.append(f"mimo_model={DYNAMIC_CONFIG['MIMO_MODEL']}")
        if req.qwen_video_model is not None:
            DYNAMIC_CONFIG["QWEN_VIDEO_MODEL"] = req.qwen_video_model.strip() or QWEN_VIDEO_MODEL
            changes.append(f"qwen_video_model={DYNAMIC_CONFIG['QWEN_VIDEO_MODEL']}")
        if req.gemini_video_model is not None:
            DYNAMIC_CONFIG["GEMINI_VIDEO_MODEL"] = req.gemini_video_model.strip() or GEMINI_VIDEO_MODEL
            changes.append(f"gemini_video_model={DYNAMIC_CONFIG['GEMINI_VIDEO_MODEL']}")
        if req.kimi_video_model is not None:
            DYNAMIC_CONFIG["KIMI_VIDEO_MODEL"] = req.kimi_video_model.strip() or KIMI_VIDEO_MODEL
            changes.append(f"kimi_video_model={DYNAMIC_CONFIG['KIMI_VIDEO_MODEL']}")
        if req.mimo_video_model is not None:
            DYNAMIC_CONFIG["MIMO_VIDEO_MODEL"] = req.mimo_video_model.strip() or MIMO_VIDEO_MODEL
            changes.append(f"mimo_video_model={DYNAMIC_CONFIG['MIMO_VIDEO_MODEL']}")
        if req.claude_video_ocr_model is not None:
            DYNAMIC_CONFIG["CLAUDE_VIDEO_OCR_MODEL"] = req.claude_video_ocr_model.strip() or CLAUDE_VIDEO_OCR_MODEL
            changes.append(f"claude_video_ocr_model={DYNAMIC_CONFIG['CLAUDE_VIDEO_OCR_MODEL']}")
        if req.openai_video_ocr_model is not None:
            DYNAMIC_CONFIG["OPENAI_VIDEO_OCR_MODEL"] = req.openai_video_ocr_model.strip() or OPENAI_VIDEO_OCR_MODEL
            changes.append(f"openai_video_ocr_model={DYNAMIC_CONFIG['OPENAI_VIDEO_OCR_MODEL']}")
        if req.nvidia_model is not None:
            DYNAMIC_CONFIG["NVIDIA_MODEL"] = req.nvidia_model.strip() or NVIDIA_MODEL
            changes.append(f"nvidia_model={DYNAMIC_CONFIG['NVIDIA_MODEL']}")
        if req.nvidia_key is not None:
            DYNAMIC_CONFIG["NVIDIA_API_KEY"]         = req.nvidia_key.strip()
            changes.append(f"nvidia={_mask(req.nvidia_key)}")
        if req.nvidia_key_backup is not None:
            DYNAMIC_CONFIG["NVIDIA_API_KEY_BACKUP"]  = req.nvidia_key_backup.strip()
            changes.append(f"nvidia_backup={_mask(req.nvidia_key_backup)}")
        if req.z_ai_key is not None:
            DYNAMIC_CONFIG["Z_AI_API_KEY"]           = req.z_ai_key.strip()
            changes.append(f"z_ai={_mask(req.z_ai_key)}")
        if req.z_ai_key_backup is not None:
            DYNAMIC_CONFIG["Z_AI_API_KEY_BACKUP"]    = req.z_ai_key_backup.strip()
            changes.append(f"z_ai_backup={_mask(req.z_ai_key_backup)}")
        # GPT-5.5 tuning agéntico
        if req.openai_reasoning_effort is not None:
            ef = (req.openai_reasoning_effort or "").strip().lower()
            if ef in ("low", "medium", "high"):
                DYNAMIC_CONFIG["OPENAI_REASONING_EFFORT"] = ef
                changes.append(f"openai_reasoning_effort={ef}")
            else:
                # Valor inválido → forzamos default y registramos para no fallar silenciosamente
                DYNAMIC_CONFIG["OPENAI_REASONING_EFFORT"] = OPENAI_REASONING_EFFORT
                changes.append(f"openai_reasoning_effort=default({OPENAI_REASONING_EFFORT})")
        if req.openai_max_tool_calls is not None:
            n = _safe_int(req.openai_max_tool_calls, OPENAI_MAX_TOOL_CALLS, lo=1, hi=20)
            DYNAMIC_CONFIG["OPENAI_MAX_TOOL_CALLS"] = n
            changes.append(f"openai_max_tool_calls={n}")
        if req.openai_allowed_domains is not None:
            # Sanitizar: lista de strings ≤100 entradas, sin espacios.
            doms = req.openai_allowed_domains
            if isinstance(doms, str):
                doms = [d.strip() for d in doms.replace("\n", ",").split(",")]
            doms = [d for d in (doms or []) if isinstance(d, str) and d.strip()][:100]
            DYNAMIC_CONFIG["OPENAI_ALLOWED_DOMAINS"] = doms
            changes.append(f"openai_allowed_domains={len(doms)} dominios")
        # Tavily — keys primaria/backup y tuning del paso intermedio
        if req.tavily_key is not None:
            DYNAMIC_CONFIG["TAVILY_API_KEY"] = req.tavily_key.strip()
            changes.append(f"tavily={_mask(req.tavily_key)}")
        if req.tavily_key_backup is not None:
            DYNAMIC_CONFIG["TAVILY_API_KEY_BACKUP"] = req.tavily_key_backup.strip()
            changes.append(f"tavily_backup={_mask(req.tavily_key_backup)}")
        if req.tavily_enabled is not None:
            DYNAMIC_CONFIG["TAVILY_ENABLED"] = bool(req.tavily_enabled)
            changes.append(f"tavily_enabled={DYNAMIC_CONFIG['TAVILY_ENABLED']}")
        if req.tavily_max_results is not None:
            n = _safe_int(req.tavily_max_results, TAVILY_MAX_RESULTS, lo=1, hi=10)
            DYNAMIC_CONFIG["TAVILY_MAX_RESULTS"] = n
            changes.append(f"tavily_max_results={n}")
        if req.tavily_search_depth is not None:
            sd = (req.tavily_search_depth or "").strip().lower()
            if sd in ("basic", "advanced"):
                DYNAMIC_CONFIG["TAVILY_SEARCH_DEPTH"] = sd
                changes.append(f"tavily_search_depth={sd}")
            else:
                DYNAMIC_CONFIG["TAVILY_SEARCH_DEPTH"] = TAVILY_SEARCH_DEPTH
                changes.append(f"tavily_search_depth=default({TAVILY_SEARCH_DEPTH})")
        if req.tavily_http_timeout_s is not None:
            t = max(1.0, min(30.0, float(req.tavily_http_timeout_s)))
            DYNAMIC_CONFIG["TAVILY_HTTP_TIMEOUT_S"] = t
            changes.append(f"tavily_http_timeout_s={t}")
        if req.tavily_total_deadline_s is not None:
            t = max(1.0, min(60.0, float(req.tavily_total_deadline_s)))
            DYNAMIC_CONFIG["TAVILY_TOTAL_DEADLINE_S"] = t
            changes.append(f"tavily_total_deadline_s={t}")
        # Meta-Judge (árbitro post-fusión, opción B "siempre" + fallback robusto).
        # Cualquier fallo del árbitro (HTTP, JSON, timeout) NO tumba el job — el
        # fusionador cae a la votación ponderada local automáticamente.
        if req.meta_judge_enabled is not None:
            DYNAMIC_CONFIG["META_JUDGE_ENABLED"] = bool(req.meta_judge_enabled)
            changes.append(f"meta_judge_enabled={DYNAMIC_CONFIG['META_JUDGE_ENABLED']}")
        if req.meta_judge_model is not None:
            # String libre: el árbitro acepta cualquier modelo OpenAI chat-compatible.
            # Vacío → restaura default "gpt-5".
            m = (req.meta_judge_model or "").strip() or "gpt-5"
            DYNAMIC_CONFIG["META_JUDGE_MODEL"] = m
            changes.append(f"meta_judge_model={m}")
        if req.meta_judge_reasoning_effort is not None:
            ef = (req.meta_judge_reasoning_effort or "").strip().lower()
            if ef in ("low", "medium", "high"):
                DYNAMIC_CONFIG["META_JUDGE_REASONING_EFFORT"] = ef
                changes.append(f"meta_judge_reasoning_effort={ef}")
            else:
                DYNAMIC_CONFIG["META_JUDGE_REASONING_EFFORT"] = "high"
                changes.append("meta_judge_reasoning_effort=default(high)")
        if req.meta_judge_timeout_s is not None:
            try:
                t = max(10.0, min(180.0, float(req.meta_judge_timeout_s)))
            except (TypeError, ValueError):
                t = 60.0
            DYNAMIC_CONFIG["META_JUDGE_TIMEOUT_S"] = t
            changes.append(f"meta_judge_timeout_s={t}")
        # Pesos (clamp 0..10): 0 = ignorar IA en fusión, 1 = básica, 2 = doble, etc.
        if req.anthropic_weight is not None:
            DYNAMIC_CONFIG["ANTHROPIC_WEIGHT"] = max(0, min(10, int(req.anthropic_weight)))
            changes.append(f"anthropic_weight={DYNAMIC_CONFIG['ANTHROPIC_WEIGHT']}")
        if req.openai_weight is not None:
            DYNAMIC_CONFIG["OPENAI_WEIGHT"] = max(0, min(10, int(req.openai_weight)))
            changes.append(f"openai_weight={DYNAMIC_CONFIG['OPENAI_WEIGHT']}")
        if req.gemini_weight is not None:
            DYNAMIC_CONFIG["GEMINI_WEIGHT"] = max(0, min(10, int(req.gemini_weight)))
            changes.append(f"gemini_weight={DYNAMIC_CONFIG['GEMINI_WEIGHT']}")
        if req.deepseek_weight is not None:
            DYNAMIC_CONFIG["DEEPSEEK_WEIGHT"] = max(0, min(10, int(req.deepseek_weight)))
            changes.append(f"deepseek_weight={DYNAMIC_CONFIG['DEEPSEEK_WEIGHT']}")
        if req.mistral_weight is not None:
            DYNAMIC_CONFIG["MISTRAL_WEIGHT"] = max(0, min(10, int(req.mistral_weight)))
            changes.append(f"mistral_weight={DYNAMIC_CONFIG['MISTRAL_WEIGHT']}")
        if req.nvidia_weight is not None:
            DYNAMIC_CONFIG["NVIDIA_WEIGHT"] = max(0, min(10, int(req.nvidia_weight)))
            changes.append(f"nvidia_weight={DYNAMIC_CONFIG['NVIDIA_WEIGHT']}")
        if req.kimi_weight is not None:
            DYNAMIC_CONFIG["KIMI_WEIGHT"] = max(0, min(10, int(req.kimi_weight)))
            changes.append(f"kimi_weight={DYNAMIC_CONFIG['KIMI_WEIGHT']}")
        if req.mimo_weight is not None:
            DYNAMIC_CONFIG["MIMO_WEIGHT"] = max(0, min(10, int(req.mimo_weight)))
            changes.append(f"mimo_weight={DYNAMIC_CONFIG['MIMO_WEIGHT']}")
        # Pesos OCR de video (fase 1) — independientes de los analyzers.
        # Misma escala 0..10. 0 = el OCR sigue corriendo pero su voto pesa 0
        # en la fusión (útil para diagnosticar si un OCR está sesgando).
        for field, key in (
            ("mistral_ocr_weight",   "MISTRAL_OCR_WEIGHT"),
            ("gemini_ocr_weight",    "GEMINI_OCR_WEIGHT"),
            ("qwen_ocr_weight",      "QWEN_OCR_WEIGHT"),
            ("glm_ocr_weight",       "GLM_OCR_WEIGHT"),
            ("anthropic_ocr_weight", "ANTHROPIC_OCR_WEIGHT"),
            ("openai_ocr_weight",    "OPENAI_OCR_WEIGHT"),
            ("deepseek_ocr_weight",  "DEEPSEEK_OCR_WEIGHT"),
            ("kimi_ocr_weight",      "KIMI_OCR_WEIGHT"),
            ("mimo_ocr_weight",      "MIMO_OCR_WEIGHT"),
        ):
            val = getattr(req, field, None)
            if val is not None:
                DYNAMIC_CONFIG[key] = max(0, min(10, int(val)))
                changes.append(f"{field}={DYNAMIC_CONFIG[key]}")
        # Topaz Labs Image API (super-resolution + denoise post-stacking, opcional).
        # Totalmente defensivo: cualquier valor inválido cae al default y NUNCA
        # tumba la request. La mejora de imagen es opcional — si Topaz se desactiva
        # o falla, el pipeline sigue con el stacking local (ver _get_image_for_page_ocr).
        if req.topaz_key is not None:
            DYNAMIC_CONFIG["TOPAZ_API_KEY"] = req.topaz_key.strip()
            changes.append(f"topaz={_mask(req.topaz_key)}")
        if req.topaz_enabled is not None:
            DYNAMIC_CONFIG["TOPAZ_ENABLED"] = bool(req.topaz_enabled)
            changes.append(f"topaz_enabled={DYNAMIC_CONFIG['TOPAZ_ENABLED']}")
        if req.topaz_model is not None:
            # String libre con default "Wonder 3"; el pipeline cae a "Wonder 2"
            # automáticamente si la API responde 400 (modelo no encontrado).
            m = (req.topaz_model or "").strip() or "Wonder 3"
            DYNAMIC_CONFIG["TOPAZ_MODEL"] = m
            changes.append(f"topaz_model={m}")
        if req.topaz_output_height is not None:
            # 0 = mantener resolución del input. Clamp defensivo 0..8192.
            n = _safe_int(req.topaz_output_height, 0, lo=0, hi=8192)
            DYNAMIC_CONFIG["TOPAZ_OUTPUT_HEIGHT"] = n
            changes.append(f"topaz_output_height={n}")
        if req.topaz_timeout_s is not None:
            try:
                t = max(5.0, min(180.0, float(req.topaz_timeout_s)))
            except (TypeError, ValueError):
                t = 60.0
            DYNAMIC_CONFIG["TOPAZ_TIMEOUT_S"] = t
            changes.append(f"topaz_timeout_s={t}")
    # Persistir en Supabase + Appwrite (fuera del lock para no bloquear si tarda)
    persisted = _save_config_to_file() if changes else True
    logger.info("⚙️ Config actualizada: %s · persistido=%s",
                " · ".join(changes) if changes else "(sin cambios)", persisted)
    # Si había cambios y NINGUNA DB los aceptó → 503 para que el panel avise.
    # Los valores quedan en RAM (DYNAMIC_CONFIG) pero un reinicio los perdería.
    if changes and not persisted:
        raise HTTPException(
            status_code=503,
            detail="Cambios aplicados en memoria pero NO persistidos: Supabase y Appwrite no disponibles. Reintenta en unos minutos.",
        )
    return {"ok": True, "changes": changes, "persisted": persisted}


# ─── Panel HTML ───────────────────────────────────────────────────────────────
@app.get("/editor", response_class=HTMLResponse)
def editor_redirect(key: str = ""):
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/panel?key={key}")


@app.get("/panel", response_class=HTMLResponse)
def panel(key: str = ""):
    if not _is_authorized_key(key):
        return HTMLResponse("<h2 style='color:red;font-family:sans-serif;text-align:center;"
                            "margin-top:100px'>Acceso Denegado</h2>", status_code=401)
    # Render super-defensivo: TRES niveles de fallback para que el panel SIEMPRE
    # devuelva algo legible y la API jamás caiga por culpa de esta ruta.
    #   1) Render normal: _render_panel_html(key) construye el f-string gigante.
    #   2) Si falla: HTMLResponse con mensaje amigable + diagnóstico + link recargar.
    #   3) Si el propio fallback falla: bytes literales mínimos (sin f-string, sin
    #      ninguna dependencia de scope). Nunca se devuelve un 500 crudo al cliente.
    try:
        return _render_panel_html(key)
    except Exception as exc:
        try:
            logger.error("💥 /panel render falló: %s", exc)
            traceback.print_exc(file=sys.stdout)
        except Exception:
            pass  # ni siquiera el log debe romper la respuesta
        # safe_key: aunque el caller diera key=None u otra rareza, garantizar str.
        try:
            safe_key  = html_lib.escape(str(key or ""))
            safe_repr = html_lib.escape(repr(exc))[:500]
        except Exception:
            safe_key, safe_repr = "", "(error renderizando diagnóstico)"
        try:
            body = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>Panel error</title></head>"
                "<body style='font-family:sans-serif;background:#111;color:#eee;padding:24px'>"
                "<h2 style='color:#dc3545'>⚠ Error renderizando panel</h2>"
                "<p>El panel falló al renderizar. La API sigue funcionando — "
                "refresca la página para reintentar.</p>"
                "<pre style='background:#1a1a1a;padding:12px;border-radius:6px;"
                "color:#f88;font-size:11px;white-space:pre-wrap;word-break:break-all'>"
                + safe_repr + "</pre>"
                "<p><a href='/panel?key=" + safe_key + "' "
                "style='color:#4d9eff'>↻ Reintentar</a></p>"
                "</body></html>"
            )
            return HTMLResponse(body, status_code=200)
        except Exception:
            # Último recurso: HTML estático que NO depende de NINGUNA variable.
            return HTMLResponse(
                "<!DOCTYPE html><html><body style='font-family:sans-serif;"
                "background:#111;color:#eee;padding:24px'>"
                "<h2 style='color:#dc3545'>Panel temporalmente no disponible</h2>"
                "<p>La API sigue activa. Refresca la página.</p></body></html>",
                status_code=200,
            )


# Plantilla HTML del panel: string NORMAL (NO f-string) con marcadores
# __VAR__ que se sustituyen en runtime. Razón: el f-string anterior mezclaba
# Python ({} = placeholder), JavaScript (${} = template literal) y CSS
# (bloques entre {}). Una llave mal escapada lanzaba NameError al renderizar
# el panel — bug invisible al arrancar. Con string normal Python NO interpreta
# {} para NADA: JS y CSS quedan exactamente como deben. Imposible introducir
# el mismo bug por accidente.
_PANEL_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Panel IA</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#111;color:#eee;min-height:100vh;padding:12px}

.header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;
         gap:8px;padding:12px 16px;background:#1a1a1a;border-radius:10px;margin-bottom:16px}
.title{font-size:18px;font-weight:700}
.stats{display:flex;gap:8px;flex-wrap:wrap}
.stat{background:#222;border-radius:8px;padding:5px 12px;font-size:12px;font-weight:700}
.s-rev{color:#ffc107}.s-pen{color:#aaa}.s-ok{color:#198754}.s-err{color:#dc3545}
.s-errlog{color:#dc3545;cursor:pointer}
.s-errlog:hover{background:#2a0a0a}

.hist-corr{background:#2a1a0a;color:#fd7e14;border:1px solid #fd7e14;border-radius:5px;
            padding:4px 10px;font-size:11px;font-weight:700;cursor:pointer;white-space:nowrap}
.hist-corr:hover{background:#fd7e14;color:#fff}

/* ── config panel ── */
.cfg{background:#111;border:1px solid #292900;border-radius:8px;padding:12px;margin-bottom:14px}
.cfg summary{cursor:pointer;font-size:12px;color:#888;font-weight:700;user-select:none;padding:4px 0}
.cfg-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
           gap:7px;margin-top:6px}
.cfg-prov{background:#0a0a0a;border:1px solid #1f1f1f;border-radius:7px;padding:7px 9px}
.cfg-prov-title{font-size:12px;font-weight:700;letter-spacing:.5px;margin-bottom:3px}
.cfg-section-title{font-size:13px;font-weight:700;letter-spacing:.5px;color:#ddd;
                    margin-top:12px;padding:6px 8px;background:#0d0d0d;
                    border:1px solid #1f1f1f;border-left:3px solid #9b85ff;border-radius:5px}
.cfg-prov-shared{background:#0a0a0d;border-color:#252540;border-style:dashed}
.cfg-shared-hint{font-size:9px;color:#7a7a9c;font-style:italic;margin-top:3px;
                  padding:2px 5px;background:#0d0d14;border-radius:3px;text-align:center}
.cfg-model-badge{font-family:'SF Mono','Consolas',monospace;font-size:10px;color:#999;
                  background:#1a1a1a;border:1px solid #2a2a2a;border-radius:4px;
                  padding:3px 6px;margin-bottom:4px;text-align:center}
.cfg-pair{display:flex;gap:4px;align-items:center;margin:2px 0}
.cfg-pair input{flex:1}
.cfg input{background:#1a1a1a;color:#eee;border:1px solid #2a2a2a;border-radius:5px;
            padding:5px 8px;font-size:11px;width:100%;margin:2px 0;
            font-family:'SF Mono','Consolas',monospace}
.cfg input:focus{outline:none;border-color:#9b85ff}
.cfg-actions{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap;
              padding-top:10px;border-top:1px solid #1f1f1f}
.cfg-hint{font-size:10px;color:#666;flex:1;min-width:200px}
.btn-cfg{background:#198754;color:#fff;border:none;border-radius:5px;
          padding:8px 14px;font-size:12px;font-weight:700;cursor:pointer;white-space:nowrap}
.btn-cfg:hover{background:#157347}
.btn-eye{background:#1a1a1a;color:#666;border:1px solid #2a2a2a;border-radius:5px;
          padding:4px 9px;font-size:12px;cursor:pointer;flex-shrink:0}
.btn-eye:hover{color:#fff;background:#222}
/* ── sliders de pesos en fusión ── */
.weight-row{display:flex;align-items:center;gap:8px;background:#0a0a0a;
             border:1px solid #1f1f1f;border-radius:6px;padding:7px 10px}
.weight-row label{font-size:11px;font-weight:700;min-width:80px}
.weight-row input[type="range"]{flex:1;accent-color:#9b85ff;height:4px}

/* ── tarjetas ── */
.card{background:#1a1a1a;border-radius:12px;margin-bottom:16px;overflow:hidden}
.card-pend{border:1px solid #333}
.card-rev{border:2px solid #ffc107}
.card-done{border:1px solid #198754;opacity:.7}
.card-err{border:1px solid #dc3545;opacity:.6}

.card-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;
            padding:10px 14px;background:#1f1f1f}
.badge{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700}
.b-pend{background:#333;color:#aaa}
.b-rev{background:#ffc107;color:#000}
.b-done{background:#198754;color:#fff}
.hora{font-size:11px;color:#666}

/* countdown badge */
.cd{margin-left:auto;font-size:12px;font-weight:700;padding:3px 10px;border-radius:16px;
     background:#222;color:#fff;white-space:nowrap}
.cd-ok{background:#198754}.cd-warn{background:#fd7e14}
.cd-urg{background:#dc3545;animation:blink .7s infinite alternate}
@keyframes blink{from{opacity:1}to{opacity:.4}}

/* image */
.img-wrap{padding:10px;background:#0d0d0d;border-bottom:1px solid #222;
           display:flex;flex-direction:column;align-items:center;gap:8px}
.foto{max-width:100%;max-height:50vh;object-fit:contain;border-radius:6px;cursor:zoom-in;
       transition:max-height .3s}
.foto.big{max-height:none;cursor:zoom-out}
.card-video{width:100%;display:block;border-radius:8px;max-height:280px;background:#000;cursor:zoom-in;transition:transform .08s ease-out}
.card-video.zoomed{transform:scale(2.2)}
/* zoom hover: lupa para imágenes (JS) + scale in-place para el vídeo */
.media-col img{cursor:zoom-in}
.media-col .img-wrap{overflow:hidden}

/* ── split layout (video izquierda · opciones derecha) ───────────────
   En cards anchas (≥720px de contenedor) se parte en 2 columnas para ver
   el video y todas las IAs (OCR + analyzers) a la vez. En cards estrechas
   (móvil, tablets pequeñas) se apilan verticalmente. Usamos container
   queries para que cada card responda a SU propia anchura, no al viewport
   — así una card en sidebar reducido sigue apilándose aunque la pantalla
   sea grande. */
.card-split{display:grid;grid-template-columns:1fr;background:#0d0d0d}
.media-col,.ia-col{display:flex;flex-direction:column;min-width:0}
.media-col:empty{display:none}
/* 2 columnas + columna de MEDIOS FIJA (sticky). IMPORTANTE: usamos @media
   (viewport), NO @container. `container-type:inline-size` en .card anclaba el
   position:sticky a la PROPIA card (que scrollea con la página) en vez de al
   viewport → las fotos se "iban" al bajar por las preguntas. Con @media el
   sticky se ancla al viewport y la columna izquierda se queda quieta. */
@media (min-width:900px){
    .card-split:has(.media-col > *){
        grid-template-columns:minmax(260px,40%) minmax(0,1fr);
        align-items:start
    }
    .media-col{border-right:1px solid #1f1f1f;padding:6px;justify-content:flex-start;
               align-self:start;position:sticky;top:8px;
               max-height:calc(100vh - 16px);overflow:auto}
    .media-col .img-wrap{border-bottom:none;margin:0;padding:6px;background:transparent}
    .ia-col .ia-section{border-bottom:none}
    .card-video{max-height:380px}
    .foto{max-height:46vh}
    .ia-col .ia-row{padding:4px 4px;min-height:32px}
    .ia-col .ia-name{min-width:54px;font-size:11px}
    .ia-col .ocr-header{padding:6px 4px 2px !important}
}

/* ── filas de IA ── */
.ia-section{padding:8px 12px;border-bottom:1px solid #222;background:#0d0d0d}
.ia-row{display:flex;align-items:center;gap:7px;padding:5px 2px;
         border-bottom:1px solid #181818;min-height:36px}
.ia-row:last-child{border-bottom:none}
.ia-name{font-size:12px;font-weight:700;min-width:64px;color:#bbb;flex-shrink:0}
.ia-name small{display:block;font-size:9px;color:#555;font-weight:400}
.ia-wait{font-size:11px;color:#444;font-style:italic;flex:1}
.ia-proc{font-size:11px;color:#0af;flex:1;animation:blink 1s infinite alternate}
.ia-err{font-size:10px;color:#dc3545;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cells{display:flex;flex-wrap:wrap;gap:3px;flex:1;justify-content:center}
.cell{width:26px;height:30px;display:inline-flex;align-items:center;justify-content:center;
       font-family:monospace;font-weight:700;font-size:15px;border-radius:4px}
.c-A{background:#0a2a4a;color:#4d9eff}.c-B{background:#0a2a1a;color:#3ddc84}
.c-C{background:#2a1a0a;color:#ffb066}.c-D{background:#2a0a2a;color:#d77ee6}
.c-X{background:#1a0a0a;color:#555}.c-q{background:#222;color:#333}
.cell.diff{box-shadow:0 0 0 2px #dc3545 inset}
.ia-fusion .ia-name{color:#ffc107}
.btn-use,.btn-quick{
    background:#222;color:#aaa;border:1px solid #333;border-radius:5px;
    padding:5px 8px;font-size:13px;font-weight:700;cursor:pointer;white-space:nowrap;flex-shrink:0
}
.btn-use:hover:not(:disabled){background:#444;color:#fff;border-color:#666}
.btn-quick{background:#0a3a1a;color:#3ddc84;border-color:#1a5a2a}
.btn-quick:hover:not(:disabled){background:#198754;color:#fff;border-color:#198754;
    box-shadow:0 0 0 2px rgba(25,135,84,.3)}
.btn-quick:active:not(:disabled){transform:scale(.92)}
.btn-use:disabled,.btn-quick:disabled{background:#0d0d0d;color:#333;cursor:not-allowed;border-color:#1a1a1a}

/* ── Cómplice ── */
.complice-wrap{padding:12px;border-bottom:1px solid #222;background:#080814}
.complice-label{font-size:11px;color:#9b85ff;font-weight:700;letter-spacing:1px;
                 text-transform:uppercase;margin-bottom:8px}
.keys-row{display:flex;flex-wrap:wrap;gap:7px;justify-content:center;margin-bottom:10px}
.key{position:relative;min-width:52px;min-height:60px;background:#1a1a1a;border:2px solid #444;
      border-radius:10px;padding:5px 3px 3px;cursor:pointer;color:#fff;font-family:inherit;
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      transition:transform .08s,border-color .15s;-webkit-tap-highlight-color:transparent;user-select:none}
.key:active{transform:scale(.91)}
.key-num{position:absolute;top:2px;left:5px;font-size:10px;color:#555;font-weight:700}
.key-letter{font-size:27px;font-weight:900;font-family:monospace;line-height:1}
.k-A{border-color:#0d6efd}.k-A .key-letter{color:#4d9eff}
.k-B{border-color:#198754}.k-B .key-letter{color:#3ddc84}
.k-C{border-color:#fd7e14}.k-C .key-letter{color:#ffb066}
.k-D{border-color:#9c27b0}.k-D .key-letter{color:#d77ee6}
.k-X{border-color:#444;background:#140a0a}.k-X .key-letter{color:#555}
.complice-text{width:100%;background:#111;color:#ffc107;border:2px solid #2a2a2a;
               border-radius:7px;padding:9px 12px;font-size:18px;font-family:monospace;
               font-weight:700;letter-spacing:4px;text-transform:uppercase;text-align:center}
.complice-text:focus{outline:none;border-color:#9b85ff}

/* ── acciones ── */
.actions-row{display:flex;gap:7px;align-items:center;padding:8px 12px;
              background:#0c0c0c;border-bottom:1px solid #1a1a1a}
.check-mini{display:flex;align-items:center;gap:5px;font-size:12px;color:#888;flex:1;cursor:pointer}
.check-mini input{width:16px;height:16px;accent-color:#ffc107}
.btn-sm{background:#222;color:#888;border:1px solid #2a2a2a;border-radius:6px;
         padding:6px 10px;font-size:11px;font-weight:600;cursor:pointer}
.btn-sm:hover{color:#fd7e14;border-color:#fd7e14}
.btn-sm-red:hover{color:#dc3545;border-color:#dc3545}

/* ── enviar ── */
.btn-send{display:flex;align-items:center;justify-content:space-between;gap:10px;
           width:100%;padding:14px 18px;background:#198754;color:#fff;border:none;
           font-size:16px;font-weight:800;cursor:pointer;transition:background .15s;
           -webkit-tap-highlight-color:transparent;text-align:left}
.btn-send:hover{background:#157347}.btn-send:active{transform:scale(.99)}
.btn-send:disabled{background:#2a2a2a;color:#555;cursor:not-allowed}
.send-label{display:flex;flex-direction:column;gap:2px;min-width:0;flex:1}
.send-label-main{font-size:16px;font-weight:800;line-height:1.1}
.send-label-sub{font-size:10px;font-weight:600;opacity:.85;letter-spacing:.3px;
                 white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.send-prev{font-family:monospace;font-weight:900;font-size:20px;letter-spacing:2px;
            background:rgba(0,0,0,.3);padding:5px 10px;border-radius:5px;
            max-width:55%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
            flex-shrink:0}

/* ── historial ── */
.hist-title{font-size:13px;color:#444;font-weight:600;margin:20px 0 8px;
             padding-bottom:5px;border-bottom:1px solid #1a1a1a}
table{width:100%;border-collapse:collapse;background:#1a1a1a;border-radius:8px;overflow:hidden}
th,td{padding:9px 11px;text-align:left;border-bottom:1px solid #222;font-size:12px}
th{background:#111;color:#444;font-size:10px;text-transform:uppercase}
.mono{font-family:monospace;font-size:14px;font-weight:700}
.gray{color:#444}

/* ── toast ── */
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
        background:#198754;color:#fff;padding:12px 22px;border-radius:8px;
        font-size:14px;font-weight:700;box-shadow:0 4px 16px rgba(0,0,0,.5);
        display:none;z-index:9999;white-space:nowrap}
.empty{text-align:center;padding:40px;color:#444;font-size:14px}

@media(max-width:600px){
    .key{min-width:46px;min-height:54px}
    .key-letter{font-size:23px}
    .cell{width:22px;height:26px;font-size:13px}
    .btn-send{font-size:14px;padding:14px}
    .card-video{max-height:220px}
}
/* Pantallas medianas-grandes: una card por fila pero suficientemente ancha
   (≥720px) para que el container query interno active el layout 2-col. */
@media(min-width:760px){
    #cards-area{display:grid;grid-template-columns:1fr;gap:16px}
}
/* Pantallas grandes: dos cards por fila, cada una ≥680px → cada card sigue
   activando su split interno (vídeo izda · IAs dcha). */
@media(min-width:1480px){
    #cards-area{grid-template-columns:repeat(auto-fit,minmax(680px,1fr))}
}
/* ── revisión por pregunta (cards de corrección OCR) ── */
.rev-wrap{padding:10px 12px;background:#080814;border-bottom:1px solid #222}
.rev-head-row{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}
.rev-title{font-size:11px;color:#9b85ff;font-weight:700;letter-spacing:1px;text-transform:uppercase}
.rev-count{font-size:11px;color:#7dd87d;font-weight:700}
.rev-hint{font-size:10px;color:#666;flex:1;text-align:right;min-width:120px}
.rev-card{position:relative;background:#0e0e18;border:1px solid #262638;border-left:4px solid #3a3a52;
          border-radius:9px;padding:9px 10px;margin-bottom:8px;transition:opacity .15s,border-color .15s}
.rev-card.ok{border-left-color:#3ddc84}
.rev-card.bad{border-left-color:#e0c060}
.rev-card.del{opacity:.45;border-left-color:#dc3545;background:#160a0a}
.rev-c-head{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.rev-num{font-size:16px;font-weight:900;font-family:monospace;color:#fff;min-width:24px;text-align:center}
.rev-orig{font-size:10px;color:#666;font-family:monospace}
.rev-sec{font-size:10px;color:#9ad;background:#0d1620;border:1px solid #1d2d3d;border-radius:4px;
         padding:1px 6px;text-transform:uppercase;letter-spacing:.5px}
.rev-tools{margin-left:auto;display:flex;gap:4px;align-items:center}
.rev-tool{background:#1a1a26;color:#999;border:1px solid #2a2a3a;border-radius:6px;
          padding:4px 9px;font-size:13px;font-weight:700;cursor:pointer;line-height:1;
          -webkit-tap-highlight-color:transparent;user-select:none}
.rev-tool:hover{border-color:#555;color:#fff}
.rev-tool.act-ok{background:#123d23;border-color:#198754;color:#3ddc84}
.rev-tool.act-bad{background:#3d3512;border-color:#a8870f;color:#e0c060}
.rev-q{font-size:13px;color:#ddd;line-height:1.35;margin:3px 0 6px;white-space:pre-wrap;word-break:break-word}
.rev-opts{display:flex;flex-direction:column;gap:2px;margin-bottom:7px}
.rev-opt{font-size:12px;color:#aaa;line-height:1.3;word-break:break-word}
.rev-opt b{color:#ccc;font-family:monospace}
.rev-ans-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.rev-ans-lbl{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.5px}
.rev-letter{min-width:30px;min-height:30px;background:#1a1a1a;border:2px solid #444;border-radius:7px;
            color:#888;font-family:monospace;font-size:15px;font-weight:900;cursor:pointer;
            display:inline-flex;align-items:center;justify-content:center;padding:0 2px;
            -webkit-tap-highlight-color:transparent;user-select:none}
.rev-letter.sel.l-A{border-color:#0d6efd;color:#4d9eff;background:#0a1a2e}
.rev-letter.sel.l-B{border-color:#198754;color:#3ddc84;background:#0a2417}
.rev-letter.sel.l-C{border-color:#fd7e14;color:#ffb066;background:#2e1a08}
.rev-letter.sel.l-D{border-color:#9c27b0;color:#d77ee6;background:#240a2a}
.rev-letter.sel.l-X{border-color:#888;color:#ccc;background:#1a1a1a}
.rev-votes{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto;align-items:center}
.rev-vote{font-size:10px;font-family:monospace;color:#888;padding:1px 5px;border-radius:4px;
          border:1px solid #2a2a3a;background:#0d0d16}
.rev-vote.agree{color:#3ddc84;border-color:#1d4d33}
.rev-vote.differ{color:#ff8a8a;border-color:#5d2323}
.rev-move{display:flex;flex-direction:column;gap:1px}
.rev-mv{background:#1a1a26;border:1px solid #2a2a3a;border-radius:4px;color:#888;cursor:pointer;
        font-size:9px;line-height:1;padding:2px 5px;-webkit-tap-highlight-color:transparent}
.rev-mv:hover{color:#fff;border-color:#555}
.rev-restore{background:#123d23;border:1px solid #198754;color:#3ddc84;border-radius:6px;
             padding:4px 10px;font-size:11px;font-weight:700;cursor:pointer;margin-left:auto}
/* botón de instrucciones + huecos de realineación */
.rev-help{background:#1a1a26;color:#9b85ff;border:1px solid #3a3a52;border-radius:6px;
          padding:3px 9px;font-size:11px;font-weight:700;cursor:pointer}
.rev-help:hover{border-color:#9b85ff;color:#fff}
.rev-gap-add{display:block;width:100%;background:#0d1620;color:#9ad;border:1px dashed #2d4d6d;
             border-radius:7px;padding:7px;font-size:12px;font-weight:600;cursor:pointer;margin-bottom:8px}
.rev-gap-add:hover{border-color:#4d9eff;color:#cfe}
.rev-card.rev-gap{border-left-color:#4d9eff;background:#0a1420}
.rev-gap-lbl{font-size:11px;color:#9ad;flex:1;font-style:italic;line-height:1.3}
</style>
</head>
<body>

<div class="header">
    <span class="title">📡 Panel IA</span>
    <div class="stats">
        <span class="stat s-rev" id="stat-rev">✏️ 0 revisión</span>
        <span class="stat s-pen" id="stat-pen">⏳ 0 procesando</span>
        <span class="stat s-ok"  id="stat-ok" >✅ 0 hechos</span>
        <span class="stat s-err" id="stat-err">❌ 0 errores</span>
        <span class="stat s-errlog" id="stat-errlog" onclick="showErrors()" title="Click para ver">📋 0 logs</span>
    </div>
</div>

<!-- Modal de errores -->
<div id="errors-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);
     z-index:9999;padding:20px;overflow:auto" onclick="if(event.target===this)this.style.display='none'">
    <div style="max-width:900px;margin:20px auto;background:#1a1a1a;border-radius:10px;
                border:1px solid #dc3545;padding:16px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
            <h3 style="margin:0;color:#dc3545">📋 Log de errores</h3>
            <div style="display:flex;gap:8px">
                <button onclick="clearErrors()" style="background:#222;color:#dc3545;border:1px solid #dc3545;
                        border-radius:5px;padding:5px 10px;font-size:11px;cursor:pointer">🗑 Limpiar</button>
                <button onclick="document.getElementById('errors-modal').style.display='none'"
                        style="background:#333;color:#fff;border:none;border-radius:5px;
                        padding:5px 10px;font-size:11px;cursor:pointer">✕ Cerrar</button>
            </div>
        </div>
        <pre id="errors-body" style="background:#0a0a0a;color:#f88;padding:12px;border-radius:6px;
             max-height:60vh;overflow:auto;font-size:11px;line-height:1.5;white-space:pre-wrap;
             word-break:break-word">Cargando...</pre>
    </div>
</div>

<details class="cfg" id="cfg-panel">
    <summary>⚙️ Configuración: API Keys, Modelos y ⚖️ Pesos de fusión · <small style="color:#666;font-weight:400">click para abrir</small></summary>

    <!-- ─────────────────────── SECCIÓN 1: ANALYZERS (FASE 2) ───────────────────────
         Las IAs que reciben el enunciado de la pregunta + texto OCR fusionado y emiten
         la respuesta A/B/C/D. Aquí están sus API keys (las que también hacen OCR de
         video — Anthropic, Gemini, MiMo — reusan ESTA misma key abajo en la sección
         de OCR; allí sólo se edita el modelo, no la key). -->
    <div class="cfg-section-title">
        🧠 IAs Analizadoras de texto <small style="color:#888;font-weight:400">(Fase 2 — análisis del enunciado y emisión de respuesta)</small>
    </div>
    <div class="cfg-grid">
        <div class="cfg-prov">
            <div class="cfg-prov-title" style="color:#3ddc84">🟢 Anthropic
                <a href="https://console.anthropic.com/settings/billing" target="_blank" rel="noopener"
                   title="Consumo y facturación · console.anthropic.com"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <input id="cfg_ant_model" type="text" placeholder="modelo (vacío = default __CLAUDE_MODEL__)"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-pair">
                <input id="cfg_ant"        type="password" placeholder="API key principal">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_ant')">👁</button>
            </div>
            <div class="cfg-pair">
                <input id="cfg_ant_bk"     type="password" placeholder="API key respaldo (opcional)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_ant_bk')">👁</button>
            </div>
        </div>
        <div class="cfg-prov">
            <div class="cfg-prov-title" style="color:#4d9eff">🔵 OpenAI <small style="color:#888;font-weight:400">(web_search agéntico · cap 2 min)</small>
                <a href="https://platform.openai.com/settings/organization/billing/overview" target="_blank" rel="noopener"
                   title="Consumo y facturación · platform.openai.com"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <input id="cfg_oai_model" type="text" placeholder="modelo (vacío = default __OPENAI_MODEL__)"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-pair">
                <input id="cfg_oai"        type="password" placeholder="API key principal">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_oai')">👁</button>
            </div>
            <div class="cfg-pair">
                <input id="cfg_oai_bk"     type="password" placeholder="API key respaldo (opcional)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_oai_bk')">👁</button>
            </div>
            <!-- Tuning agéntico: reasoning effort + tope de búsquedas + dominios -->
            <div style="margin-top:8px;padding:6px 8px;background:#0d0d14;
                        border:1px dashed #2a2a4a;border-radius:5px">
                <label style="font-size:9px;color:#888;text-transform:uppercase;letter-spacing:.5px;display:block;margin:0 0 3px">⚡ Reasoning effort</label>
                <select id="cfg_oai_effort"
                        style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                               padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
                    <option value="low">low — rápido (~15s · sin deep search)</option>
                    <option value="medium" selected>medium — equilibrio (~45s · default)</option>
                    <option value="high">high — deep research (~90s · max calidad)</option>
                </select>
                <label style="font-size:9px;color:#888;text-transform:uppercase;letter-spacing:.5px;display:block;margin:2px 0 1px">🔍 Máx búsquedas en el loop</label>
                <input id="cfg_oai_max_tools" type="number" min="1" max="20" value="4"
                       style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                              padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
                <label style="font-size:9px;color:#888;text-transform:uppercase;letter-spacing:.5px;display:block;margin:2px 0 1px">📜 Dominios autorizados <small style="color:#666;text-transform:none;letter-spacing:0">(uno por línea · vacío = todo internet · máx 100)</small></label>
                <textarea id="cfg_oai_domains" rows="3" placeholder="boe.es&#10;noticias.juridicas.com&#10;tribunalconstitucional.es"
                          style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                                 padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:10px;resize:vertical;min-height:40px"></textarea>
            </div>
        </div>
        <div class="cfg-prov">
            <div class="cfg-prov-title" style="color:#ffb066">🟠 Gemini
                <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener"
                   title="API keys y cuotas · aistudio.google.com"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <input id="cfg_gem_model" type="text" placeholder="modelo (vacío = default __GEMINI_MODEL__)"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-pair">
                <input id="cfg_gem"        type="password" placeholder="API key principal">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_gem')">👁</button>
            </div>
            <div class="cfg-pair">
                <input id="cfg_gem_bk"     type="password" placeholder="API key respaldo (opcional)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_gem_bk')">👁</button>
            </div>
        </div>
        <div class="cfg-prov">
            <div class="cfg-prov-title" style="color:#a855f7">🟣 DeepSeek <small style="color:#888;font-weight:400">(text-only)</small>
                <a href="https://platform.deepseek.com/usage" target="_blank" rel="noopener"
                   title="Consumo y facturación · platform.deepseek.com"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <input id="cfg_dsk_model" type="text" placeholder="modelo (vacío = default __DEEPSEEK_MODEL__)"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-pair">
                <input id="cfg_dsk"        type="password" placeholder="API key principal">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_dsk')">👁</button>
            </div>
            <div class="cfg-pair">
                <input id="cfg_dsk_bk"     type="password" placeholder="API key respaldo (opcional)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_dsk_bk')">👁</button>
            </div>
        </div>
        <div class="cfg-prov">
            <div class="cfg-prov-title" style="color:#fb7185">🌶️ Mistral <small style="color:#888;font-weight:400">(text-only)</small>
                <a href="https://console.mistral.ai/billing" target="_blank" rel="noopener"
                   title="Consumo y facturación · console.mistral.ai"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <input id="cfg_mst_model" type="text" placeholder="modelo (vacío = default __MISTRAL_MODEL__)"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-pair">
                <input id="cfg_mst"        type="password" placeholder="API key principal">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_mst')">👁</button>
            </div>
            <div class="cfg-pair">
                <input id="cfg_mst_bk"     type="password" placeholder="API key respaldo (opcional)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_mst_bk')">👁</button>
            </div>
        </div>
        <div class="cfg-prov">
            <div class="cfg-prov-title" style="color:#76b900">🟢 NVIDIA <small style="color:#888;font-weight:400">(Nemotron Super · reasoning text-only)</small>
                <a href="https://build.nvidia.com/settings/api-keys" target="_blank" rel="noopener"
                   title="API keys y consumo · build.nvidia.com (NVIDIA NIM Cloud)"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <input id="cfg_nv_model" type="text" placeholder="modelo (vacío = default __NVIDIA_MODEL__)"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-pair">
                <input id="cfg_nv"         type="password" placeholder="API key principal (nvapi-…)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_nv')">👁</button>
            </div>
            <div class="cfg-pair">
                <input id="cfg_nv_bk"      type="password" placeholder="API key respaldo (opcional)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_nv_bk')">👁</button>
            </div>
        </div>
        <div class="cfg-prov">
            <div class="cfg-prov-title" style="color:#ff6b00">🤖 MiMo (Xiaomi) <small style="color:#888;font-weight:400">(full-modal)</small>
                <a href="https://api.xiaomimimo.com" target="_blank" rel="noopener"
                   title="Consumo y facturación · api.xiaomimimo.com"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <input id="cfg_mim_model" type="text" placeholder="modelo (vacío = default __MIMO_MODEL__)"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-pair">
                <input id="cfg_mim"        type="password" placeholder="API key principal">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_mim')">👁</button>
            </div>
            <div class="cfg-pair">
                <input id="cfg_mim_bk"     type="password" placeholder="API key respaldo (opcional)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_mim_bk')">👁</button>
            </div>
        </div>
    </div>

    <!-- ─────────────────────── SECCIÓN 2: OCR DE VIDEO (FASE 1) ───────────────────────
         Las IAs que reciben el video y devuelven la transcripción del enunciado.
         Qwen y Kimi son OCR-puros (sólo aparecen aquí). Anthropic, Gemini y MiMo
         son dual-rol — su API key se edita en la sección de analyzers de arriba
         (compartida); aquí sólo se ajusta su modelo específico para OCR-video. -->
    <div class="cfg-section-title" style="margin-top:18px">
        🎬 IAs OCR de Video <small style="color:#888;font-weight:400">(Fase 1 — transcripción del enunciado desde el video)</small>
    </div>
    <div class="cfg-grid">
        <div class="cfg-prov">
            <div class="cfg-prov-title" style="color:#22d3ee">🎬 Qwen
                <a href="https://dashscope.console.aliyun.com/billing" target="_blank" rel="noopener"
                   title="Consumo y facturación · dashscope.console.aliyun.com"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <input id="cfg_qwn_model" type="text" placeholder="modelo OCR video (vacío = default __QWEN_VIDEO_MODEL__)"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-pair">
                <input id="cfg_qwn"        type="password" placeholder="API key principal">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_qwn')">👁</button>
            </div>
            <div class="cfg-pair">
                <input id="cfg_qwn_bk"     type="password" placeholder="API key respaldo (opcional)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_qwn_bk')">👁</button>
            </div>
        </div>
        <div class="cfg-prov">
            <div class="cfg-prov-title" style="color:#facc15">🎬 Kimi
                <a href="https://platform.moonshot.ai/console/account" target="_blank" rel="noopener"
                   title="Consumo y facturación · platform.moonshot.ai (Moonshot AI)"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <input id="cfg_kmi_model" type="text" placeholder="modelo OCR video (vacío = default __KIMI_VIDEO_MODEL__)"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-pair">
                <input id="cfg_kmi"        type="password" placeholder="API key principal">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_kmi')">👁</button>
            </div>
            <div class="cfg-pair">
                <input id="cfg_kmi_bk"     type="password" placeholder="API key respaldo (opcional)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_kmi_bk')">👁</button>
            </div>
        </div>
        <!-- Mixtos: sólo input de MODELO de OCR-video. La API key se reusa de la
             sección de analyzers (no se duplica para evitar inconsistencias). -->
        <div class="cfg-prov cfg-prov-shared">
            <div class="cfg-prov-title" style="color:#3ddc84">🟢 Anthropic <small style="color:#888;font-weight:400">(visión sobre frame)</small></div>
            <label style="font-size:9px;color:#888;text-transform:uppercase;letter-spacing:.5px;display:block;margin:2px 0 1px">Modelo OCR de video (Claude)</label>
            <input id="cfg_ant_video_model" type="text" placeholder="vacío = default __CLAUDE_VIDEO_OCR_MODEL__"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-shared-hint">🔗 Key compartida con Anthropic analyzer (sección de arriba)</div>
        </div>
        <div class="cfg-prov cfg-prov-shared">
            <div class="cfg-prov-title" style="color:#ffb066">🟠 Gemini <small style="color:#888;font-weight:400">(video nativo)</small></div>
            <label style="font-size:9px;color:#888;text-transform:uppercase;letter-spacing:.5px;display:block;margin:2px 0 1px">Modelo OCR de video</label>
            <input id="cfg_gem_video_model" type="text" placeholder="vacío = default __GEMINI_VIDEO_MODEL__"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-shared-hint">🔗 Key compartida con Gemini analyzer (sección de arriba)</div>
        </div>
        <div class="cfg-prov cfg-prov-shared">
            <div class="cfg-prov-title" style="color:#ff6b00">🤖 MiMo (Xiaomi) <small style="color:#888;font-weight:400">(video nativo)</small></div>
            <label style="font-size:9px;color:#888;text-transform:uppercase;letter-spacing:.5px;display:block;margin:2px 0 1px">Modelo OCR de video</label>
            <input id="cfg_mim_video_model" type="text" placeholder="vacío = default __MIMO_VIDEO_MODEL__"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-shared-hint">🔗 Key compartida con MiMo analyzer (sección de arriba)</div>
        </div>
        <div class="cfg-prov cfg-prov-shared">
            <div class="cfg-prov-title" style="color:#4d9eff">🔵 OpenAI <small style="color:#888;font-weight:400">(GPT-4o · visión sobre frame)</small></div>
            <label style="font-size:9px;color:#888;text-transform:uppercase;letter-spacing:.5px;display:block;margin:2px 0 1px">Modelo OCR de video</label>
            <input id="cfg_oai_video_model" type="text" placeholder="vacío = default __OPENAI_VIDEO_OCR_MODEL__"
                   style="width:100%;background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                          padding:5px 8px;font-family:'SF Mono','Consolas',monospace;font-size:11px;margin-bottom:3px">
            <div class="cfg-shared-hint">🔗 Key compartida con OpenAI analyzer (sección de arriba) · sube a gpt-5.5 para máxima calidad</div>
        </div>
        <div class="cfg-prov">
            <div class="cfg-prov-title" style="color:#a78bfa">🟣 Z.AI <small style="color:#888;font-weight:400">(GLM-OCR · OCR-image puro)</small>
                <a href="https://z.ai/manage-apikey/billing" target="_blank" rel="noopener"
                   title="Consumo y facturación · z.ai (Zhipu / BigModel)"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <div class="cfg-shared-hint">SOTA OmniDocBench v1.5 (94.62) · soporta español · best-frame único top_k=1</div>
            <div class="cfg-pair">
                <input id="cfg_zai"       type="password" placeholder="API key principal">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_zai')">👁</button>
            </div>
            <div class="cfg-pair">
                <input id="cfg_zai_bk"    type="password" placeholder="API key respaldo (opcional)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_zai_bk')">👁</button>
            </div>
        </div>
    </div>

    <!-- ─────────────────────── SECCIÓN 3: BUSCADORES WEB (FASE 1.5) ────────────────
         Paso INTERMEDIO entre el OCR de video (fase 1) y los analyzers de texto
         (fase 2). Lanza UNA búsqueda por pregunta detectada en paralelo, con un
         deadline total absoluto: si no responde a tiempo el prompt se envía sin
         internet — el job NUNCA bloquea por esto. -->
    <div class="cfg-section-title" style="margin-top:18px">
        🔎 Buscadores web <small style="color:#888;font-weight:400">(Fase 1.5 — contexto de internet inyectado al prompt del razonador, con timeout duro)</small>
    </div>
    <div class="cfg-grid">
        <div class="cfg-prov" style="grid-column:1/-1">
            <div class="cfg-prov-title" style="color:#0ea5e9">🌐 Tavily <small style="color:#888;font-weight:400">(web search · paso intermedio OCR → razonamiento)</small>
                <a href="https://app.tavily.com/" target="_blank" rel="noopener"
                   title="Obtener key · app.tavily.com"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <div style="font-size:10px;color:#888;margin-bottom:6px;line-height:1.4">
                Tras la fusión OCR se lanza UNA búsqueda por pregunta (en paralelo)
                y los resultados se inyectan al prompt del razonador como
                contexto secundario. Si no responde a tiempo, el prompt se manda
                sin internet — el job NUNCA bloquea por esto.
            </div>
            <div class="cfg-pair">
                <input id="cfg_tav"       type="password" placeholder="API key principal (tvly-…)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_tav')">👁</button>
            </div>
            <div class="cfg-pair">
                <input id="cfg_tav_bk"    type="password" placeholder="API key respaldo (opcional)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_tav_bk')">👁</button>
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;align-items:center">
                <label style="font-size:11px;color:#9cf;display:flex;align-items:center;gap:4px">
                    <input id="cfg_tav_enabled" type="checkbox" checked> activado
                </label>
                <label style="font-size:10px;color:#888;display:flex;align-items:center;gap:4px">
                    profundidad
                    <select id="cfg_tav_depth"
                            style="background:#0a0a0a;color:#9cf;border:1px solid #333;border-radius:4px;
                                   padding:3px 6px;font-size:10px"
                            title="basic = rápido (<2s) · advanced = más profundo (+2-4s)">
                        <option value="basic">basic</option>
                        <option value="advanced">advanced</option>
                    </select>
                </label>
                <label style="font-size:10px;color:#888;display:flex;align-items:center;gap:4px">
                    max fuentes
                    <input id="cfg_tav_max" type="number" min="1" max="10" value="3"
                           title="Resultados por pregunta (1..10, recomendado 3)"
                           style="width:60px;background:#0a0a0a;color:#9cf;border:1px solid #333;
                                  border-radius:4px;padding:3px 6px;font-size:10px">
                </label>
                <label style="font-size:10px;color:#888;display:flex;align-items:center;gap:4px">
                    HTTP timeout (s)
                    <input id="cfg_tav_http_to" type="number" min="1" max="30" step="0.5" value="6"
                           title="Timeout HTTP por petición (1..30s, recomendado 6)"
                           style="width:70px;background:#0a0a0a;color:#9cf;border:1px solid #333;
                                  border-radius:4px;padding:3px 6px;font-size:10px">
                </label>
                <label style="font-size:10px;color:#888;display:flex;align-items:center;gap:4px">
                    deadline total (s)
                    <input id="cfg_tav_deadline" type="number" min="1" max="60" step="0.5" value="8"
                           title="Deadline TOTAL del paso entero (1..60s, recomendado 8). Si se agota, prompt se envía SIN internet"
                           style="width:70px;background:#0a0a0a;color:#9cf;border:1px solid #333;
                                  border-radius:4px;padding:3px 6px;font-size:10px">
                </label>
            </div>
        </div>
    </div>

    <!-- ─────────────── META-JUDGE (árbitro post-fusión, opción B + fallback) ───
         Tras la votación ponderada local, se llama a un LLM externo (GPT-5 por
         defecto) que ve el OCR fusionado + las respuestas A/B/C/D de cada
         analyzer y produce el veredicto final. Si la llamada falla por CUALQUIER
         motivo (red, rate limit, JSON malformado, timeout, longitud incorrecta),
         se cae automáticamente a la fusión local determinística — el job NUNCA
         se rompe por culpa del árbitro. -->
    <div class="cfg-section-title" style="margin-top:18px">
        🧑‍⚖️ Meta-Judge <small style="color:#888;font-weight:400">(Fase 3 — árbitro final post-fusión · siempre activo + fallback a fusión local)</small>
    </div>
    <div class="cfg-grid">
        <div class="cfg-prov" style="grid-column:1/-1">
            <div class="cfg-prov-title" style="color:#fbbf24">⚖️ Árbitro final <small style="color:#888;font-weight:400">(usa la OPENAI_API_KEY ya configurada arriba)</small></div>
            <div style="font-size:10px;color:#888;margin-bottom:8px;line-height:1.4">
                Después de la votación ponderada de los 5 analyzers, el árbitro recibe
                el examen OCR fusionado + las letras A/B/C/D de cada IA y produce el
                veredicto final con razonamiento profundo. <b style="color:#fbbf24">Si el árbitro
                falla por cualquier motivo, el fusionador local determinístico hace
                fallback automático</b> — el job nunca cae por esto.
            </div>
            <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:6px">
                <label style="font-size:11px;color:#fbbf24;display:flex;align-items:center;gap:4px">
                    <input id="cfg_mj_enabled" type="checkbox" checked> activado
                </label>
                <label style="font-size:10px;color:#888;display:flex;align-items:center;gap:4px">
                    modelo
                    <input id="cfg_mj_model" type="text" placeholder="gpt-5" value="gpt-5"
                           title="Modelo OpenAI chat-compatible para el árbitro (ej. gpt-5, gpt-5.5, o1, o3, o4)"
                           style="width:120px;background:#0a0a0a;color:#fbbf24;border:1px solid #333;
                                  border-radius:4px;padding:3px 6px;font-size:10px">
                </label>
                <label style="font-size:10px;color:#888;display:flex;align-items:center;gap:4px">
                    reasoning effort
                    <select id="cfg_mj_effort"
                            style="background:#0a0a0a;color:#fbbf24;border:1px solid #333;border-radius:4px;
                                   padding:3px 6px;font-size:10px"
                            title="low = más rápido · medium = balance · high = razonamiento profundo (recomendado)">
                        <option value="low">low</option>
                        <option value="medium">medium</option>
                        <option value="high" selected>high</option>
                    </select>
                </label>
                <label style="font-size:10px;color:#888;display:flex;align-items:center;gap:4px">
                    timeout (s)
                    <input id="cfg_mj_timeout" type="number" min="10" max="180" step="5" value="60"
                           title="Timeout HTTP del árbitro (10..180s). Si se agota, fallback a fusión local"
                           style="width:70px;background:#0a0a0a;color:#fbbf24;border:1px solid #333;
                                  border-radius:4px;padding:3px 6px;font-size:10px">
                </label>
            </div>
        </div>
    </div>

    <!-- ─────────────── MEJORA DE IMAGEN (super-resolution post-stacking) ───────
         Tras el stacking local multi-frame, opcionalmente se sube la imagen
         fusionada a la Image API de Topaz Labs (Wonder 3) para super-resolution
         + denoise ANTES de mandarla a los OCR de página única (Mistral/DeepSeek/
         GLM). Si Topaz se desactiva, falla, agota timeout o devuelve un modelo
         inexistente, el pipeline usa el stacking local — el job NUNCA bloquea por
         esto. En cada resultado el dashboard marca ✨ (mejorada) / 🔧 (solo stacking). -->
    <div class="cfg-section-title" style="margin-top:18px">
        🖼️ Mejora de imagen <small style="color:#888;font-weight:400">(Fase 0.5 — super-resolution del folio fusionado antes del OCR · timeout duro + fallback a stacking local)</small>
    </div>
    <div class="cfg-grid">
        <div class="cfg-prov" style="grid-column:1/-1">
            <div class="cfg-prov-title" style="color:#d4a017">✨ Topaz Labs <small style="color:#888;font-weight:400">(Image API · Wonder 3 — super-resolution + denoise post-stacking)</small>
                <a href="https://www.topazlabs.com/" target="_blank" rel="noopener"
                   title="Topaz Labs · topazlabs.com"
                   style="color:#888;text-decoration:none;font-size:10px;margin-left:6px;font-weight:400">💳 billing ↗</a>
            </div>
            <div style="font-size:10px;color:#888;margin-bottom:6px;line-height:1.4">
                La imagen fusionada (stacking multi-frame) se sube a Topaz para
                super-resolution antes del OCR de página única. <b style="color:#d4a017">Si Topaz
                se desactiva, falla, agota su timeout o devuelve un modelo
                inexistente, el pipeline usa el stacking local</b> — el job nunca
                cae por esto. En cada resultado se marca ✨ (mejorada) o 🔧 (solo stacking).
            </div>
            <div class="cfg-pair">
                <input id="cfg_tpz"       type="password" placeholder="API key (UUID Topaz)">
                <button class="btn-eye" type="button" onclick="toggleEye('cfg_tpz')">👁</button>
            </div>
            <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:8px">
                <label style="font-size:11px;color:#d4a017;display:flex;align-items:center;gap:4px">
                    <input id="cfg_tpz_enabled" type="checkbox" checked> activado
                </label>
                <label style="font-size:10px;color:#888;display:flex;align-items:center;gap:4px">
                    modelo
                    <select id="cfg_tpz_model"
                            style="background:#0a0a0a;color:#d4a017;border:1px solid #333;border-radius:4px;
                                   padding:3px 6px;font-size:10px"
                            title="Wonder 3 = última gen (recomendado) · Wonder 2 = fallback automático si Wonder 3 devuelve 400">
                        <option value="Wonder 3">Wonder 3</option>
                        <option value="Wonder 2">Wonder 2</option>
                    </select>
                </label>
                <label style="font-size:10px;color:#888;display:flex;align-items:center;gap:4px">
                    alto salida (px)
                    <input id="cfg_tpz_height" type="number" min="0" max="8192" step="128" value="0"
                           title="0 = mantener resolución del input (recomendado). >0 fuerza ese alto en px"
                           style="width:80px;background:#0a0a0a;color:#d4a017;border:1px solid #333;
                                  border-radius:4px;padding:3px 6px;font-size:10px">
                </label>
                <label style="font-size:10px;color:#888;display:flex;align-items:center;gap:4px">
                    timeout (s)
                    <input id="cfg_tpz_timeout" type="number" min="5" max="180" step="5" value="60"
                           title="Tope global POST+poll+GET (5..180s). Si se agota, fallback a stacking local"
                           style="width:70px;background:#0a0a0a;color:#d4a017;border:1px solid #333;
                                  border-radius:4px;padding:3px 6px;font-size:10px">
                </label>
            </div>
        </div>
    </div>

    <!-- ── Pesos ANALYZERS (fase 2 — análisis del OCR fusionado) ───────── -->
    <div style="margin-top:14px;border-top:1px solid #2a2a2a;padding-top:10px">
        <div style="font-size:12px;font-weight:700;letter-spacing:.5px;color:#ddd;margin-bottom:4px">
            ⚖️ Pesos ANALYZERS <small style="font-weight:400;color:#888">
            (fase 2 · cada IA vota con su peso · 0 = ignorar · 10 = vale por 10 IAs básicas
            · <span style="color:#9b85ff">💾 auto-guardado al soltar slider</span>)</small>
            <span id="weights-status" style="font-size:10px;color:#198754;margin-left:8px"></span>
        </div>
        <div style="font-size:11px;color:#888;margin-bottom:10px;line-height:1.5">
            Pesos para la votación de los analyzers (Claude/GPT/Gemini/DeepSeek/Mistral/MiMo)
            sobre el texto OCR ya fusionado. Sube el peso de las IAs más potentes
            (Claude/GPT con web search + thinking, DeepSeek-R1) para que pesen
            más en la respuesta final.
        </div>
        <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
            <!-- Columna izquierda: sliders -->
            <div style="flex:1;min-width:280px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px">
                <div class="weight-row">
                    <label style="color:#3ddc84">🟢 Anthropic</label>
                    <input id="cfg_w_ant" type="range" min="0" max="10" step="1" value="1"
                           oninput="document.getElementById('cfg_w_ant_v').textContent=this.value;renderWeightsPie()"
                           onchange="autoSaveWeights()">
                    <span id="cfg_w_ant_v" class="mono" style="color:#fff;min-width:18px;text-align:right">1</span>
                </div>
                <div class="weight-row">
                    <label style="color:#4d9eff">🔵 OpenAI</label>
                    <input id="cfg_w_oai" type="range" min="0" max="10" step="1" value="1"
                           oninput="document.getElementById('cfg_w_oai_v').textContent=this.value;renderWeightsPie()"
                           onchange="autoSaveWeights()">
                    <span id="cfg_w_oai_v" class="mono" style="color:#fff;min-width:18px;text-align:right">1</span>
                </div>
                <div class="weight-row">
                    <label style="color:#ffb066">🟠 Gemini</label>
                    <input id="cfg_w_gem" type="range" min="0" max="10" step="1" value="1"
                           oninput="document.getElementById('cfg_w_gem_v').textContent=this.value;renderWeightsPie()"
                           onchange="autoSaveWeights()">
                    <span id="cfg_w_gem_v" class="mono" style="color:#fff;min-width:18px;text-align:right">1</span>
                </div>
                <div class="weight-row">
                    <label style="color:#a855f7">🟣 DeepSeek</label>
                    <input id="cfg_w_dsk" type="range" min="0" max="10" step="1" value="1"
                           oninput="document.getElementById('cfg_w_dsk_v').textContent=this.value;renderWeightsPie()"
                           onchange="autoSaveWeights()">
                    <span id="cfg_w_dsk_v" class="mono" style="color:#fff;min-width:18px;text-align:right">1</span>
                </div>
                <div class="weight-row">
                    <label style="color:#fb7185">🌶️ Mistral</label>
                    <input id="cfg_w_mst" type="range" min="0" max="10" step="1" value="1"
                           oninput="document.getElementById('cfg_w_mst_v').textContent=this.value;renderWeightsPie()"
                           onchange="autoSaveWeights()">
                    <span id="cfg_w_mst_v" class="mono" style="color:#fff;min-width:18px;text-align:right">1</span>
                </div>
                <div class="weight-row">
                    <label style="color:#76b900">🟢 NVIDIA</label>
                    <input id="cfg_w_nv" type="range" min="0" max="10" step="1" value="1"
                           oninput="document.getElementById('cfg_w_nv_v').textContent=this.value;renderWeightsPie()"
                           onchange="autoSaveWeights()">
                    <span id="cfg_w_nv_v" class="mono" style="color:#fff;min-width:18px;text-align:right">1</span>
                </div>
                <div class="weight-row">
                    <label style="color:#ff6b00">🤖 MiMo</label>
                    <input id="cfg_w_mim" type="range" min="0" max="10" step="1" value="0"
                           oninput="document.getElementById('cfg_w_mim_v').textContent=this.value;renderWeightsPie()"
                           onchange="autoSaveWeights()">
                    <span id="cfg_w_mim_v" class="mono" style="color:#fff;min-width:18px;text-align:right">0</span>
                </div>
            </div>
            <!-- Columna derecha: pie chart con peso relativo de cada IA -->
            <div style="flex:0 0 auto;display:flex;flex-direction:column;align-items:center;gap:6px">
                <svg id="weights-pie-svg" viewBox="0 0 140 140" width="140" height="140"
                     style="background:#0a0a0a;border:1px solid #2a2a2a;border-radius:50%"></svg>
                <div id="weights-pie-legend" style="font-size:10px;color:#aaa;max-width:240px;
                     text-align:center;line-height:1.6"></div>
            </div>
        </div>
    </div>

    <!-- ── Pesos OCR de video (fase 1 — transcripción del enunciado) ───── -->
    <div style="margin-top:14px;border-top:1px solid #2a2a2a;padding-top:10px">
        <div style="font-size:12px;font-weight:700;letter-spacing:.5px;color:#ddd;margin-bottom:4px">
            🎬📄 Pesos OCR de VIDEO <small style="font-weight:400;color:#888">
            (fase 1 · cada OCR vota con su peso en la fusión multi-OCR
            · <span style="color:#9b85ff">💾 auto-guardado</span>)</small>
            <span id="weights-ocr-status" style="font-size:10px;color:#198754;margin-left:8px"></span>
        </div>
        <div style="font-size:11px;color:#888;margin-bottom:10px;line-height:1.5">
            Pesos INDEPENDIENTES de los analyzers. Defaults tier-based según evidencia
            2026 de accuracy en español: <b>Mistral OCR 3 / Gemini / Qwen3-VL</b> arrancan
            altos (TIER 1), <b>GLM-OCR</b> medio-alto (TIER 2), <b>Kimi/MiMo</b> bajos
            (entrenados con foco chino). El voto del fusionador OCR usa estos pesos
            para desempatar cuando dos transcripciones tienen el mismo número de
            OCRs apoyándolas.
        </div>
        <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
            <div style="flex:1;min-width:280px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px">
                <div class="weight-row">
                    <label style="color:#fb7185">📄 Mistral OCR</label>
                    <input id="cfg_wocr_mst" type="range" min="0" max="10" step="1" value="5"
                           oninput="document.getElementById('cfg_wocr_mst_v').textContent=this.value;renderOcrWeightsPie()"
                           onchange="autoSaveOcrWeights()">
                    <span id="cfg_wocr_mst_v" class="mono" style="color:#fff;min-width:18px;text-align:right">5</span>
                </div>
                <div class="weight-row">
                    <label style="color:#ffb066">🎬 Gemini OCR</label>
                    <input id="cfg_wocr_gem" type="range" min="0" max="10" step="1" value="4"
                           oninput="document.getElementById('cfg_wocr_gem_v').textContent=this.value;renderOcrWeightsPie()"
                           onchange="autoSaveOcrWeights()">
                    <span id="cfg_wocr_gem_v" class="mono" style="color:#fff;min-width:18px;text-align:right">4</span>
                </div>
                <div class="weight-row">
                    <label style="color:#22d3ee">🎬 Qwen OCR</label>
                    <input id="cfg_wocr_qwn" type="range" min="0" max="10" step="1" value="4"
                           oninput="document.getElementById('cfg_wocr_qwn_v').textContent=this.value;renderOcrWeightsPie()"
                           onchange="autoSaveOcrWeights()">
                    <span id="cfg_wocr_qwn_v" class="mono" style="color:#fff;min-width:18px;text-align:right">4</span>
                </div>
                <div class="weight-row">
                    <label style="color:#a78bfa">📄 GLM-OCR (Z.AI)</label>
                    <input id="cfg_wocr_glm" type="range" min="0" max="10" step="1" value="4"
                           oninput="document.getElementById('cfg_wocr_glm_v').textContent=this.value;renderOcrWeightsPie()"
                           onchange="autoSaveOcrWeights()">
                    <span id="cfg_wocr_glm_v" class="mono" style="color:#fff;min-width:18px;text-align:right">4</span>
                </div>
                <div class="weight-row">
                    <label style="color:#3ddc84">🎬 Claude OCR</label>
                    <input id="cfg_wocr_ant" type="range" min="0" max="10" step="1" value="3"
                           oninput="document.getElementById('cfg_wocr_ant_v').textContent=this.value;renderOcrWeightsPie()"
                           onchange="autoSaveOcrWeights()">
                    <span id="cfg_wocr_ant_v" class="mono" style="color:#fff;min-width:18px;text-align:right">3</span>
                </div>
                <div class="weight-row">
                    <label style="color:#4d9eff">🎬 GPT-4o OCR</label>
                    <input id="cfg_wocr_oai" type="range" min="0" max="10" step="1" value="3"
                           oninput="document.getElementById('cfg_wocr_oai_v').textContent=this.value;renderOcrWeightsPie()"
                           onchange="autoSaveOcrWeights()">
                    <span id="cfg_wocr_oai_v" class="mono" style="color:#fff;min-width:18px;text-align:right">3</span>
                </div>
                <div class="weight-row">
                    <label style="color:#a855f7">📄 DeepSeek OCR</label>
                    <input id="cfg_wocr_dsk" type="range" min="0" max="10" step="1" value="2"
                           oninput="document.getElementById('cfg_wocr_dsk_v').textContent=this.value;renderOcrWeightsPie()"
                           onchange="autoSaveOcrWeights()">
                    <span id="cfg_wocr_dsk_v" class="mono" style="color:#fff;min-width:18px;text-align:right">2</span>
                </div>
                <div class="weight-row">
                    <label style="color:#facc15">🎬 Kimi OCR</label>
                    <input id="cfg_wocr_kmi" type="range" min="0" max="10" step="1" value="1"
                           oninput="document.getElementById('cfg_wocr_kmi_v').textContent=this.value;renderOcrWeightsPie()"
                           onchange="autoSaveOcrWeights()">
                    <span id="cfg_wocr_kmi_v" class="mono" style="color:#fff;min-width:18px;text-align:right">1</span>
                </div>
                <div class="weight-row">
                    <label style="color:#ff6b00">🎬 MiMo OCR</label>
                    <input id="cfg_wocr_mim" type="range" min="0" max="10" step="1" value="1"
                           oninput="document.getElementById('cfg_wocr_mim_v').textContent=this.value;renderOcrWeightsPie()"
                           onchange="autoSaveOcrWeights()">
                    <span id="cfg_wocr_mim_v" class="mono" style="color:#fff;min-width:18px;text-align:right">1</span>
                </div>
            </div>
            <div style="flex:0 0 auto;display:flex;flex-direction:column;align-items:center;gap:6px">
                <svg id="weights-ocr-pie-svg" viewBox="0 0 140 140" width="140" height="140"
                     style="background:#0a0a0a;border:1px solid #2a2a2a;border-radius:50%"></svg>
                <div id="weights-ocr-pie-legend" style="font-size:10px;color:#aaa;max-width:240px;
                     text-align:center;line-height:1.6"></div>
            </div>
        </div>
    </div>

    <div class="cfg-actions" style="margin-top:14px">
        <span class="cfg-hint">💡 La key de respaldo se usa <b>solo si la principal falla</b>. Déjala vacía para desactivarla.</span>
        <button class="btn-cfg" onclick="saveKeys()">💾 Guardar todo</button>
        <button class="btn-cfg" onclick="loadConfig()" style="background:#1a1a1a">↻ Recargar</button>
    </div>
    <div class="cfg-actions" style="margin-top:6px">
        <span class="cfg-hint">📦 Por si el servidor reinicia (free tier): descarga un backup y re-importa cuando haga falta.</span>
        <button class="btn-cfg" onclick="exportConfig()" style="background:#0d6efd">📥 Exportar JSON</button>
        <button class="btn-cfg" onclick="document.getElementById('cfg-import-file').click()"
                style="background:#fd7e14">📤 Importar JSON</button>
        <input id="cfg-import-file" type="file" accept=".json,application/json"
               style="display:none" onchange="importConfig(this)">
    </div>
    <div id="cfg-status" style="font-size:13px;font-weight:600;color:#999;margin-top:10px;
                                  padding:8px;text-align:center;border-radius:6px;background:#0a0a0a;
                                  border:1px solid #2a2a2a">Inicializando…</div>
</details>

<div id="cards-area"><div class="empty">Sin peticiones activas</div></div>
<div class="hist-title">Historial reciente</div>
<table>
    <thead><tr><th>Hora</th><th>Estado</th><th>Respuesta enviada</th><th>IA fusión</th><th></th></tr></thead>
    <tbody id="hist-body"><tr><td colspan="5" class="gray" style="text-align:center">Vacío</td></tr></tbody>
</table>
<div class="toast" id="toast"></div>

<script>
const KEY = __KEY_REPR__;
const _LET = ['A','B','C','D','X'];
const _STATE = {};   // jid → {dirty:bool}

// ── Polling resiliente con AbortController + backoff exponencial ───────────
let _pollFails = 0;
let _pollInflight = false;

async function poll() {
    if(_pollInflight) return;   // si el anterior aún corre, no acumular
    _pollInflight = true;
    const ctrl = new AbortController();
    const timeoutId = setTimeout(() => ctrl.abort(), 8000);  // tope 8s
    try {
        // limit=100 (default era 25 → causaba que los jobs viejos cayeran del
        // listado en cuanto entraban nuevos. Cuando el usuario tenía abierta una
        // fila "Ver/Editar" de un job histórico y un nuevo job lo empujaba fuera
        // del top 25, document.getElementById('corrrow-jid') devolvía null y el
        // restore de display:'' fallaba silenciosamente → la fila "se cerraba"
        // visualmente sin avisar).
        const r = await fetch('/api/jobs?key=' + KEY + '&limit=100', {signal: ctrl.signal});
        clearTimeout(timeoutId);
        if(!r.ok) throw new Error('HTTP ' + r.status);
        applyJobs(await r.json());
        refreshContextBanner();   // estado del contexto de imagen sticky (server-side)
        if(_pollFails > 0) {
            _pollFails = 0;
            setRelayStatus('ok');
        }
    } catch(e) {
        clearTimeout(timeoutId);
        _pollFails++;
        if(_pollFails >= 2) setRelayStatus('down');
    } finally {
        _pollInflight = false;
    }
}

function setRelayStatus(state) {
    let el = document.getElementById('relay-status');
    if(!el) {
        el = document.createElement('div');
        el.id = 'relay-status';
        el.style.cssText = 'position:fixed;top:10px;right:10px;padding:6px 12px;'
                         + 'border-radius:6px;font-size:11px;font-weight:700;'
                         + 'z-index:9998;pointer-events:none';
        document.body.appendChild(el);
    }
    if(state === 'down') {
        el.style.background = '#dc3545'; el.style.color = '#fff';
        el.textContent = '⚠ Sin conexión al relay';
    } else {
        el.style.background = ''; el.style.color = '';
        el.textContent = '';
    }
}

// ── Contexto de imagen STICKY (server-side) ─────────────────────────────────
// El relay recuerda el diagrama del caso práctico (marcado con la casilla 🖼️) y
// lo adjunta él mismo a CADA /ask nuevo. Este banner avisa de que está activo —
// así nadie olvida que se está mandando a las IAs— y permite quitarlo de 1 clic.
async function refreshContextBanner(){
    try{
        const r = await fetch('/api/context_status?key=' + KEY);
        if(!r.ok) return;
        renderContextBanner(await r.json());
    }catch(_){ /* no crítico: si falla, el banner simplemente no se actualiza */ }
}
function renderContextBanner(st){
    let el = document.getElementById('ctx-banner');
    if(st && st.active){
        if(!el){
            el = document.createElement('div');
            el.id = 'ctx-banner';
            el.style.cssText = 'margin:8px 0;padding:8px 12px;border:1px solid #2e7d32;'
                + 'background:#0e1f10;color:#7fd98a;border-radius:8px;font-size:12px;'
                + 'display:flex;align-items:center;gap:10px;justify-content:space-between;flex-wrap:wrap';
            const area = document.getElementById('cards-area');
            if(area && area.parentNode){ area.parentNode.insertBefore(el, area); }
            else { document.body.insertBefore(el, document.body.firstChild); }
        }
        const mins = Math.round((st.age_s||0)/60);
        el.innerHTML = '<span>🖼️ <b>Imagen-contexto ACTIVA</b> — el relay la adjunta como referencia (IMAGEN 1) a TODAS las preguntas nuevas'
            + (st.size_kb ? ' · ' + st.size_kb + ' KB' : '')
            + ' · marcada hace ' + mins + ' min</span>'
            + '<button onclick="clearContext()" style="background:#5a1a1a;color:#fff;border:0;'
            + 'border-radius:5px;padding:4px 10px;cursor:pointer;font-size:11px;white-space:nowrap">✕ Quitar contexto</button>';
        el.style.display = '';
    } else if(el){
        el.style.display = 'none';
    }
}
async function clearContext(){
    try{
        const r = await fetch('/api/context_clear?key=' + KEY, {method:'POST', headers:{'X-Api-Key':KEY}});
        if(r.ok){ toast('🧹 Contexto de imagen quitado'); renderContextBanner({active:false}); }
        else { toast('❌ No se pudo quitar el contexto'); }
    }catch(e){ toast('❌ ' + e.message); }
}

// A10: helper para escapar valores que se interpolan dentro de `onclick="fn('${x}')"`.
// Si `x` contiene una comilla simple, sin escape rompe el atributo y permite
// inyectar HTML/JS. Aplica también a job_id (UUID hex normalmente seguro, pero
// si la persistencia se corrompe podría llegar cualquier cosa).
function jsAttr(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function applyJobs(jobs) {
    const area = document.getElementById('cards-area');
    const active = jobs.filter(j => j.status !== 'done' && j.status !== 'error');
    const hist   = jobs.filter(j => j.status === 'done'  || j.status === 'error');

    if(active.length > 0) {
        const emp = area.querySelector('.empty'); if(emp) emp.remove();
    } else if(!area.querySelector('.card')) {
        if(!area.querySelector('.empty'))
            area.innerHTML = '<div class="empty">Sin peticiones activas</div>';
    }

    // Stats
    const nRev  = jobs.filter(j=>j.status==='awaiting_review').length;
    const nPend = jobs.filter(j=>j.status==='pending').length;
    const nDone = jobs.filter(j=>j.status==='done').length;
    const nErr  = jobs.filter(j=>j.status==='error').length;
    document.getElementById('stat-rev').textContent = '✏️ ' + nRev + ' revisión';
    document.getElementById('stat-pen').textContent = '⏳ ' + nPend + ' procesando';
    document.getElementById('stat-ok').textContent  = '✅ ' + nDone + ' hechos';
    document.getElementById('stat-err').textContent = '❌ ' + nErr + ' errores';

    // Polling adaptativo: jobs activos (pending/awaiting_review) reducen el
    // intervalo a 1.5s para feedback rápido; sin nada activo subimos a 5s
    // para no martillear al servidor con polls que devuelven lo mismo.
    _activeJobsCount = nRev + nPend;

    // Active cards: crear las nuevas, actualizar las existentes
    const seen = new Set();
    // Aislamiento por job: si UN job malformado tira un error en buildCard/
    // updateCard, los demás siguen renderizándose. Antes un job con campos raros
    // (responses=null, _providers=string, fecha inválida) tumbaba el poll entero
    // y la lista quedaba congelada hasta el siguiente refresh manual.
    for(const job of active) {
        seen.add(job.id);
        try {
            const el = document.getElementById('card-' + job.id);
            if(!el) {
                area.insertAdjacentHTML('afterbegin', buildCard(job));
                ticks();
            } else {
                updateCard(el, job);
            }
            _syncReview(job);
        } catch(jobErr) {
            console.error('[panel] Error rendering active job ' + (job && job.id),
                          jobErr);
        }
    }
    // Retirar las que ya no están activas
    area.querySelectorAll('.card[data-jid]').forEach(el => {
        if(!seen.has(el.dataset.jid)) {
            try { delete window._REVIEW[el.dataset.jid]; } catch(e){}  // evita fuga de estado de revisión
            el.style.transition = 'opacity .4s'; el.style.opacity = '0';
            setTimeout(() => el.remove(), 420);
        }
    });

    // Historial (con botón Corregir + ver respuestas individuales + vídeo)
    if(hist.length > 0) {
        const tb = document.getElementById('hist-body');

        // Preservar estado entre polls (cada 1.5s applyJobs re-render el tbody
        // entero; sin esto las filas "Ver/Editar" abiertas se cierran solas y
        // los inputs que el usuario esté rellenando se borran).
        const expandedIds = new Set();
        tb.querySelectorAll('tr[id^="corrrow-"]').forEach(tr => {
            if(tr.style.display !== 'none') expandedIds.add(tr.id.substring(8));
        });

        // A11: ANTES si había UNA fila expandida, se congelaba el historial
        // entero — nuevos jobs no aparecían hasta cerrar la fila. Ahora hacemos
        // re-render selectivo: actualizamos las filas que NO están expandidas
        // y preservamos las expandidas con su DOM intacto (sin tocar el <video>
        // ni el contenido cargado vía /api/partial).
        //
        // Estrategia:
        //   1) Si NO hay filas expandidas → fast path: re-render todo el tbody
        //      (igual que antes, conserva el comportamiento)
        //   2) Si HAY filas expandidas → re-render por fila:
        //        - para cada job del top-30, busca su <tr id="hist-{jid}"> existente
        //        - si jid está en expandedIds → no toques la fila ni su corrrow
        //        - si no → reemplaza solo esa fila + su corrrow vacía
        //        - los jobs nuevos se insertan al principio
        //        - los jobs viejos (fuera del top-30) se quitan SI no están expandidos
        const savedInputs = {};   // id → value (inputs/textareas dentro de filas expandidas)
        const savedFocusId = (document.activeElement && document.activeElement.id) || null;
        const savedSelStart = (document.activeElement && 'selectionStart' in document.activeElement)
                              ? document.activeElement.selectionStart : null;
        const savedSelEnd   = (document.activeElement && 'selectionEnd' in document.activeElement)
                              ? document.activeElement.selectionEnd : null;
        expandedIds.forEach(jid => {
            const row = document.getElementById('corrrow-' + jid);
            if(!row) return;
            row.querySelectorAll('input, textarea, select').forEach(el => {
                if(el.id) savedInputs[el.id] = el.value;
            });
        });

        // Helper: construye el par <tr hist> + <tr corrrow vacío> para UN job.
        // try/catch interno → un job malformado no tira el render entero.
        function _renderHistRowPair(j) {
            try {
                const hora = new Date((j.created || 0) * 1000).toLocaleTimeString('es');
                const bg   = j.status === 'done' ? '#198754' : '#dc3545';
                const ans  = (j.answer || '-').toUpperCase();
                const m    = (j.merged_answer || '').toUpperCase();
                const auto = j.auto_approved ? ' 🤖' : '';
                const canCorrect = (j.status === 'done' || j.status === 'error');
                const jidEsc = jsAttr(j.id);
                const ansEsc = jsAttr(ans);
                const correctBtn = canCorrect
                    ? `<button class="hist-corr" onclick="toggleCorr('${jidEsc}','${ansEsc}')">✏️ Ver/Editar</button>`
                    : '';
                const tStats = j.tavily_stats || null;
                const hasTavQ = tStats && Array.isArray(tStats.queries) && tStats.queries.length > 0;
                if(hasTavQ) {
                    window._TAVILY_STATS = window._TAVILY_STATS || {};
                    window._TAVILY_STATS[j.id] = tStats;
                }
                const tavBtn = hasTavQ
                    ? `<button class="hist-corr" onclick="showTavilyQueries('${jidEsc}')"
                               style="background:#0ea5e9;border:none;color:#fff;margin-right:4px"
                               title="Ver búsquedas Tavily inyectadas al razonador">🔎 ${tStats.queries.length}</button>`
                    : '';
                const statusUpper = jsAttr(String(j.status || 'unknown').toUpperCase());
                return `<tr id="hist-${jidEsc}">
                    <td class="gray">${jsAttr(hora)}</td>
                    <td><span style="padding:2px 8px;border-radius:12px;background:${bg};
                        font-size:10px;font-weight:700;color:#fff">${statusUpper}${auto}</span></td>
                    <td class="mono">${jsAttr(ans)}</td>
                    <td class="mono gray">${m && m!==ans ? jsAttr(m) : ''}</td>
                    <td style="text-align:right">${tavBtn}${correctBtn}</td>
                </tr>
                <tr id="corrrow-${jidEsc}" style="display:none" data-loaded="0">
                    <td colspan="5" style="background:#0d0d0d;padding:14px">
                        <div style="color:#888;text-align:center;padding:14px;font-size:12px">
                            ⏳ Pulsa "✏️ Ver/Editar" para cargar el expediente completo
                        </div>
                    </td>
                </tr>`;
            } catch(histErr) {
                console.error('[panel] Error rendering history job ' + (j && j.id),
                              histErr);
                return `<tr><td colspan="5" style="color:#dc3545;font-size:11px">
                    ⚠ Fila inválida (job ${j && j.id ? j.id.slice(0,8) : '?'})
                    </td></tr>`;
            }
        }

        const top30 = hist.slice(0, 30);
        if(expandedIds.size === 0) {
            // FAST PATH: ninguna fila expandida → re-render limpio del tbody.
            tb.innerHTML = top30.map(_renderHistRowPair).join('');
        } else {
            // A11: SELECTIVE PATH. Hay filas expandidas → no destruir el DOM
            // existente. Por cada job del top-30:
            //   - si su hist-row ya existe → actualiza SOLO los <td> de la fila
            //     principal (no toca el corrrow expandido)
            //   - si no existe → la creamos (inserta al principio)
            // Y quitamos las filas históricas que ya no estén en el top-30
            // (salvo las que estén expandidas — esas se quedan hasta cerrarlas).
            const seenIds = new Set();
            let prevRow = null;   // para inserción ordenada
            for(const j of top30) {
                seenIds.add(j.id);
                const existing = document.getElementById('hist-' + j.id);
                if(existing) {
                    // Actualiza solo la fila hist; el corrrow ya tiene su contenido.
                    // Reemplazamos el <tr id="hist-..."> con el nuevo, manteniendo
                    // el corrrow intacto.
                    const tmp = document.createElement('tbody');
                    tmp.innerHTML = _renderHistRowPair(j);
                    const newHistRow = tmp.querySelector('tr[id^="hist-"]');
                    if(newHistRow) {
                        existing.replaceWith(newHistRow);
                        // El corrrow queda donde estaba (después del histRow nuevo
                        // gracias a su posición previa).
                    }
                    prevRow = document.getElementById('corrrow-' + j.id) || existing;
                } else {
                    // Insertar nuevo pair al principio (o tras prevRow si hay)
                    const tmp = document.createElement('tbody');
                    tmp.innerHTML = _renderHistRowPair(j);
                    const newRows = Array.from(tmp.children);
                    if(prevRow && prevRow.parentNode === tb) {
                        for(const nr of newRows) prevRow.after(nr);
                    } else {
                        for(let i = newRows.length - 1; i >= 0; i--) {
                            tb.insertAdjacentElement('afterbegin', newRows[i]);
                        }
                    }
                    prevRow = newRows[newRows.length - 1];
                }
            }
            // Quitar filas viejas que ya no están en top30, EXCEPTO expandidas
            Array.from(tb.querySelectorAll('tr[id^="hist-"]')).forEach(tr => {
                const jid = tr.id.substring(5);
                if(!seenIds.has(jid) && !expandedIds.has(jid)) {
                    const corrrow = document.getElementById('corrrow-' + jid);
                    tr.remove();
                    if(corrrow) corrrow.remove();
                }
            });
        }

        // Restaurar visibilidad de las filas que estaban abiertas + valores que
        // el usuario haya tipeado dentro. Si el job ya no existe (cayó del top
        // 30), se ignora silenciosamente.
        expandedIds.forEach(jid => {
            const row = document.getElementById('corrrow-' + jid);
            if(row) row.style.display = '';
        });
        Object.keys(savedInputs).forEach(id => {
            const el = document.getElementById(id);
            if(el) el.value = savedInputs[id];
        });
        // Restaurar foco (con caret/selección si era input/textarea) para que
        // el usuario no pierda el cursor en medio de una corrección.
        if(savedFocusId) {
            const el = document.getElementById(savedFocusId);
            if(el) {
                el.focus();
                if(savedSelStart != null && 'setSelectionRange' in el) {
                    try { el.setSelectionRange(savedSelStart, savedSelEnd ?? savedSelStart); }
                    catch(_) {}
                }
            }
        }
    }
}

// ── Detalle de un job histórico ──────────────────────────────────────────────
// EXPEDIENTE COMPLETO: media + OCRs (si vídeo) + respuestas IA (con raw, modelo,
// error, edición de letras) + matriz de votos de la fusión + editor de
// corrección retroactiva. Todos los datos vienen ya en el job (responses,
// ocr_results, ocr_fused_text, etc.) — solo los renderizamos.
function buildHistoryDetail(j, ans) {
    const jid        = j.id;
    const responses  = j.responses || {};
    const ocrResults = j.ocr_results || {};
    const merged     = (j.merged_answer || '').toUpperCase();
    const expected   = j.expected_questions || Math.max(merged.length, ans.length, 1);
    // A10: añadimos escape de comilla simple (&#39;) — antes solo se escapaban
    // &<>" — así un dato corrupto con `'` que llegue a un `onclick="fn('${x}')"`
    // no rompe el atributo ni permite XSS. Coste cero, defensa universal.
    const escape     = s => String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')
                                          .replace(/>/g,'&gt;').replace(/"/g,'&quot;')
                                          .replace(/'/g,'&#39;');

    // 0) Cabecera SIM (qué SIM se usó + dBm). El card-head no se muestra en el
    //    histórico, así que repetimos el dato aquí para debugar problemas de cobertura.
    //    Se prepende al final (las ramas de has_video/has_img sobreescriben mediaHtml).
    let simBannerHtml = '';
    if(j.sim_operator != null || j.sim_dbm != null) {
        simBannerHtml = `<div style="margin-bottom:8px;padding:6px 8px;background:#0d1620;
                                      border:1px solid #2a2a2a;border-radius:5px;font-size:11px;
                                      color:#9ad;font-family:monospace">
            ${renderSimInfo(j)}
            ${j.sim_summary ? `<div style="color:#888;margin-top:4px;font-size:10px">${escape(j.sim_summary)}</div>` : ''}
        </div>`;
    }

    // 1) Vídeo / imagen del job (si todavía está en RAM)
    let mediaHtml = '';
    // Confirmación: este job marcó 🖼️ "Imagen en pregunta" → el relay cacheó la
    // mejor imagen y la devolvió al móvil como diagrama de contexto del caso
    // práctico. El móvil la adjunta a las SIGUIENTES solicitudes.
    if(j.context_image_provided) {
        mediaHtml += `<div style="border:1px solid #2e7d32;background:#0e1f10;border-radius:8px;margin-bottom:8px;padding:6px 8px;font-size:11px;color:#7fd98a">
            🖼️ <b>Imagen-contexto enviada al móvil</b> — se adjuntará como referencia (IMAGEN 1) a las próximas preguntas.
        </div>`;
    }
    // Diagrama de contexto (si el cliente lo adjuntó). Aparece ARRIBA — orden
    // visual coincide con el orden con que se envió al modelo (IMAGEN 1).
    const ctxSrc = j.context_source || 'none';
    const ctxSrcLabel = ctxSrc === 'mobile'
        ? 'móvil'
        : (ctxSrc === 'sticky' ? 'sticky relay' : 'sin contexto');
    if(j.has_context_image) {
        mediaHtml += `<div style="border:1px solid #4a3a1a;background:#1a1408;border-radius:8px;margin-bottom:8px;padding:4px">
            <div style="font-size:10px;color:#d4a017;text-transform:uppercase;letter-spacing:1px;padding:2px 4px 4px">
                🗂️ Diagrama de contexto · ✅ se envía a las IAs <small style="color:#856404;text-transform:none;letter-spacing:0">(IMAGEN 1, junto a la foto nueva · origen: ${ctxSrcLabel})</small>
            </div>
            <img src="/api/context_image/${jid}?key=${KEY}"
                 style="max-width:100%;max-height:240px;border-radius:6px;display:block"
                 onerror="this.parentNode.innerHTML='<div style=padding:8px;color:#888;font-size:11px>⚠ Diagrama no disponible</div>'">
        </div>`;
    }
    if(j.has_video && !j.video_pruned) {
        mediaHtml = `<div style="margin-bottom:10px">
            <video src="/api/video/${jid}?key=${KEY}" controls muted playsinline
                   style="width:100%;max-width:480px;max-height:240px;border-radius:6px;background:#000"
                   onerror="this.parentNode.innerHTML='<div style=color:#888;font-size:11px>⚠ Vídeo expirado (>30min)</div>'">
            </video>
        </div>`;
    } else if(j.has_video && j.video_pruned) {
        mediaHtml = `<div style="color:#888;font-size:11px;margin-bottom:8px">
            ⏳ Vídeo expirado (purgado tras 30min para liberar RAM)
        </div>`;
    } else if(j.has_img && !j.img_pruned) {
        mediaHtml = `<div style="margin-bottom:10px">
            <img src="/api/image/${jid}?key=${KEY}" style="max-width:100%;max-height:240px;border-radius:6px">
        </div>`;
    }

    // Tira de frames extraídos + imagen fusionada — sólo si el job tenía
    // video y el relay aún los conserva en RAM (purgados a los 30 min vía
    // VIDEO_PRUNE_AFTER). Mismo helper que la card en vivo.
    if(j.has_video && ((j.extracted_frames_count || 0) > 0 || j.has_fused_image)) {
        mediaHtml += buildFramesStrip(jid, j);
    }

    // 2) OCRs por IA + fusión OCR (sólo en jobs de vídeo)
    let ocrHtml = '';
    if(j.has_video) {
        ocrHtml = '<div style="margin-bottom:12px">'
                + '<div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;'
                + 'padding:4px 0">📝 OCRs del vídeo (fase 1)</div>';
        const ocrProvs = ['qwen_ocr','gemini_ocr','kimi_ocr','mimo_ocr','anthropic_ocr','openai_ocr'];
        for(const prov of ocrProvs) {
            const r = ocrResults[prov];
            const label = PROVIDER_LABELS[prov] || prov;
            if(!r) {
                ocrHtml += `<div style="display:flex;gap:8px;padding:3px 0;font-family:monospace;font-size:12px;color:#555">
                    <span style="min-width:120px">${label}</span><span>— sin datos</span></div>`;
            } else if(r.ok) {
                // Mismo patrón defensivo que buildOcrRow: si backend viejo no
                // envía ni chars ni text en el slim, mostramos "OK" en lugar
                // de "0 chars" para no inducir a error.
                let chars, charsLabel;
                if(r.chars != null) {
                    chars = r.chars;
                    charsLabel = `${chars} chars`;
                } else if(r.text != null) {
                    chars = r.text.length;
                    charsLabel = `${chars} chars`;
                } else {
                    chars = null;
                    charsLabel = 'OK';
                }
                if(r.text) {
                    window._OCR_TEXTS = window._OCR_TEXTS || {};
                    window._OCR_TEXTS[`${jid}|${prov}`] = r.text;
                }
                const previewSrc = r.text || r.preview || '';
                const preview = escape(previewSrc.slice(0, 50));
                const ms = r.ms ? `<small style="color:#666">${ms_fmt(r.ms)}</small>` : '';
                ocrHtml += `<div style="display:flex;gap:8px;align-items:center;padding:3px 0;font-family:monospace;font-size:12px">
                    <span style="min-width:120px;color:#9cf">${label} ${ms}</span>
                    <span style="color:#5d8;min-width:80px">✓ ${charsLabel}</span>
                    <span style="color:#888;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${preview}${(chars != null && chars > 50) ? '…' : ''}</span>
                    <button onclick="showOcrText('${jid}','${prov}','${label}')"
                            style="background:#0d6efd;color:#fff;border:none;border-radius:4px;
                                   padding:3px 8px;font-size:10px;cursor:pointer" title="Ver transcripción completa">📜</button>
                </div>`;
            } else {
                const err = escape((r.error || 'error').slice(0, 100));
                const ms = r.ms ? `<small style="color:#666">${ms_fmt(r.ms)}</small>` : '';
                ocrHtml += `<div style="display:flex;gap:8px;padding:3px 0;font-family:monospace;font-size:12px">
                    <span style="min-width:120px;color:#9cf">${label} ${ms}</span>
                    <span style="color:#dc3545" title="${err}">❌ ${err}</span></div>`;
            }
        }
        // Fila de la fusión OCR (texto canónico que se envió a los analyzers)
        if(j.ocr_fused_text) {
            window._OCR_TEXTS = window._OCR_TEXTS || {};
            // Sólo cacheamos la versión COMPLETA. En el listado slim el backend
            // trunca ocr_fused_text a 200 chars y marca _ocr_fused_truncated=true;
            // si cacheamos eso, el popup mostraría 200 chars en lugar de los 4000+
            // reales. Mejor dejar vacío y que showOcrText haga lazy-fetch desde
            // /api/partial cuando el usuario abra el modal.
            if(!j._ocr_fused_truncated) {
                window._OCR_TEXTS[`${jid}|fusion`] = j.ocr_fused_text;
            }
            const used = j.ocr_fusion_used === true;
            const stats = j.ocr_fusion_stats || {};
            const k = stats.clusters || 0;
            const color = used ? '#5d8' : '#e0c060';
            const tag   = used ? '✓ usada' : '⚠ no usada (fallback)';
            // chars REAL desde ocr_fused_text_full_chars; .length daría el truncado.
            const fusedChars = (j.ocr_fused_text_full_chars != null)
                                 ? j.ocr_fused_text_full_chars
                                 : j.ocr_fused_text.length;
            ocrHtml += `<div style="display:flex;gap:8px;align-items:center;padding:5px 0 3px;
                                     border-top:1px dashed #2a2a2a;margin-top:4px;font-family:monospace;font-size:12px">
                <span style="min-width:120px;color:${color};font-weight:700">🧬 Fusión OCR</span>
                <span style="color:${color};flex:1">${tag} · ${k} preguntas · ${fusedChars} chars</span>
                <button onclick="showOcrText('${jid}','fusion','Fusión OCR')"
                        style="background:${used?'#198754':'#d4a017'};color:#fff;border:none;border-radius:4px;
                               padding:3px 8px;font-size:10px;cursor:pointer">📜</button>
            </div>`;
        }
        // Fila Tavily (paso intermedio OCR → razonamiento). Mostramos qué se buscó,
        // si llegó a tiempo, cuántas fuentes salieron, y permitimos ver el XML
        // exacto que se inyectó al prompt del razonador.
        const tavStats = j.tavily_stats || {};
        const tavEnabled = (tavStats.enabled === true) || j.tavily_used === true;
        if(tavEnabled || tavStats.skipped) {
            const nQ        = tavStats.n_questions   || 0;
            const nSources  = tavStats.n_with_sources || 0;
            const elapsed   = tavStats.elapsed_ms    || 0;
            const skipped   = tavStats.skipped       || '';
            const blockLen  = (j.tavily_block_full_chars != null)
                                ? j.tavily_block_full_chars
                                : (j.tavily_block ? j.tavily_block.length : 0);
            const usedTav   = j.tavily_used === true;
            const tCol      = skipped ? '#888' : (usedTav ? '#0ea5e9' : '#e0c060');
            let tDesc;
            if(skipped) {
                tDesc = `⏭ omitido · motivo=${escape(skipped)}`;
            } else if(usedTav) {
                tDesc = `✓ inyectado · ${nSources}/${nQ} preguntas con fuentes · `
                      + `${blockLen} chars · ${ms_fmt(elapsed)}`;
            } else {
                tDesc = `⚠ sin resultados utilizables · ${nQ} preguntas · ${ms_fmt(elapsed)}`;
            }
            // Cachea el bloque para que el modal no haga lazy-fetch si ya lo tenemos.
            if(j.tavily_block && !j._tavily_block_truncated) {
                window._TAVILY_BLOCKS = window._TAVILY_BLOCKS || {};
                window._TAVILY_BLOCKS[jid] = j.tavily_block;
            }
            window._TAVILY_STATS = window._TAVILY_STATS || {};
            if(Array.isArray(tavStats.queries) && tavStats.queries.length) {
                window._TAVILY_STATS[jid] = tavStats;
            }
            const btns = [];
            if(Array.isArray(tavStats.queries) && tavStats.queries.length) {
                btns.push(`<button onclick="showTavilyQueries('${jid}')"
                                    style="background:#0ea5e9;color:#fff;border:none;border-radius:4px;
                                           padding:3px 8px;font-size:10px;cursor:pointer"
                                    title="Ver qué buscó Tavily por cada pregunta">🔎 ${tavStats.queries.length}</button>`);
            }
            if(blockLen > 0) {
                btns.push(`<button onclick="showTavilyBlock('${jid}')"
                                    style="background:#0d6efd;color:#fff;border:none;border-radius:4px;
                                           padding:3px 8px;font-size:10px;cursor:pointer"
                                    title="Ver el XML <internet_context> EXACTO inyectado al prompt del razonador">📜</button>`);
            }
            ocrHtml += `<div style="display:flex;gap:8px;align-items:center;padding:5px 0 3px;
                                     border-top:1px dashed #2a2a2a;margin-top:4px;font-family:monospace;font-size:12px">
                <span style="min-width:120px;color:${tCol};font-weight:700">🌐 Tavily</span>
                <span style="color:${tCol};flex:1">${tDesc}</span>
                ${btns.join('')}
            </div>`;
        }
        ocrHtml += '</div>';
    }

    // 3) Respuestas individuales por IA (editables · con raw / modelo / error)
    let respsHtml = '<div style="margin-bottom:12px">'
                  + '<div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;'
                  + 'padding:4px 0">🧠 Respuestas por IA '
                  + '<small style="text-transform:none;letter-spacing:0;color:#666">(💾 guarda letras editadas · 📜 ve la respuesta completa)</small>'
                  + '</div>';
    const provs = (j._providers && j._providers.length) ? j._providers
                 : Object.keys(responses);
    if(provs.length === 0) {
        respsHtml += '<div style="color:#666;font-size:11px">(sin datos de IAs)</div>';
    } else {
        for(const prov of provs) {
            const r     = responses[prov] || {};
            const label = (typeof PROVIDER_LABELS!=='undefined' && PROVIDER_LABELS[prov])
                         || prov.toUpperCase();
            const a     = (r.answer || '').toUpperCase();
            const okIco = r.ok ? '✓' : (r.status==='no_key' ? '🔑' : '❌');
            const ms    = r.ms ? `<small style="color:#666">${ms_fmt(r.ms)}</small>` : '';
            const edited = r.edited ? '<small style="color:#ffc107"> · ✎ editado</small>' : '';
            const modelTag = r.model
                ? `<small style="color:#557;font-size:10px" title="modelo usado">[${escape(r.model)}]</small>`
                : '';

            // Botón "📜" para abrir el raw completo de esta IA en modal.
            // Cacheamos en window._RAW_RESPONSES para no meter 5000+ chars al DOM por
            // cada IA × cada job × top-30 del historial.
            let rawBtn = '';
            if(r.raw && r.raw.length > 0) {
                window._RAW_RESPONSES = window._RAW_RESPONSES || {};
                window._RAW_RESPONSES[`${jid}|${prov}`] = r.raw;
                rawBtn = `<button onclick="showRawResponse('${jid}','${prov}','${label}')"
                                  style="background:#6c757d;color:#fff;border:none;border-radius:4px;
                                         padding:4px 8px;font-size:10px;cursor:pointer"
                                  title="Ver razonamiento/respuesta completa de ${label} (${r.raw.length} chars)">📜</button>`;
            }

            // Mensaje de error inline si la IA falló (status_no_key no es error real)
            const errInline = (!r.ok && r.status !== 'no_key' && r.error)
                ? `<div style="color:#dc3545;font-size:11px;padding:2px 0 0 98px;font-family:monospace"
                        title="${escape(r.error)}">⚠ ${escape((r.error||'').slice(0,180))}</div>`
                : '';
            // Mensaje no_key (gris, no es error)
            const noKeyInline = (r.status === 'no_key')
                ? `<div style="color:#888;font-size:11px;padding:2px 0 0 98px;font-style:italic">
                       🔑 sin key configurada — no participa en la fusión</div>`
                : '';

            respsHtml += `<div>
                <div style="display:flex;gap:8px;align-items:center;padding:4px 0;
                            font-family:monospace;font-size:13px;flex-wrap:wrap">
                    <span style="min-width:90px;color:#9cf">${label} ${okIco}</span>
                    <input id="resp-${jid}-${prov}" type="text" maxlength="50"
                           value="${a}"
                           style="flex:1;min-width:80px;background:#1a1a1a;color:#fff;
                                  border:1px solid #333;border-radius:4px;padding:4px 8px;
                                  font-family:monospace;font-weight:700;letter-spacing:2px;text-transform:uppercase"
                           oninput="this.value=this.value.toUpperCase().replace(/[^ABCDX]/g,'')">
                    <button onclick="saveProviderResponse('${jid}','${prov}')"
                            style="background:#0d6efd;color:#fff;border:none;border-radius:4px;
                                   padding:4px 10px;font-size:11px;font-weight:700;cursor:pointer"
                            title="Guardar letras editadas (recalcula la fusión)">💾</button>
                    ${rawBtn}
                    ${ms}${edited}${modelTag}
                </div>
                ${errInline}${noKeyInline}
            </div>`;
        }
        // Fila fusión + botón para mostrar matriz de votos
        respsHtml += `<div style="display:flex;gap:8px;align-items:center;padding:6px 0;
                                   font-family:monospace;font-size:13px;border-top:1px dashed #2a2a2a;margin-top:4px">
            <span style="min-width:90px;color:#5d8;font-weight:700">🔀 Fusión</span>
            <span id="merged-${jid}" style="flex:1;font-family:monospace;letter-spacing:2px;font-weight:700;color:#fff">${merged || '—'}</span>
            <button onclick="toggleFusionMatrix('${jid}')"
                    style="background:#5d8;color:#000;border:none;border-radius:4px;
                           padding:3px 8px;font-size:10px;font-weight:700;cursor:pointer"
                    title="Ver / ocultar la matriz de votos pregunta-a-pregunta">📊 Votos</button>
            <small style="color:#666;font-size:10px">(auto-recalcula)</small>
        </div>
        <div id="fusion-matrix-${jid}" style="display:none;margin-top:6px">${buildFusionMatrix(j)}</div>`;
    }
    respsHtml += '</div>';

    // 4) Editor de corrección retroactiva (vibra al móvil)
    const corrHtml = `<div style="border-top:1px solid #2a2a2a;padding-top:10px">
        <div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;padding-bottom:6px">
            📤 Enviar corrección al móvil (vibrará la nueva respuesta)
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <input id="corra-${jid}" type="text" maxlength="30" value="${ans}"
                   style="flex:1;min-width:160px;font-family:monospace;font-weight:700;
                          font-size:18px;letter-spacing:3px;text-transform:uppercase;
                          padding:8px;background:#1a1a1a;color:#fff;border:2px solid #fd7e14;border-radius:6px"
                   oninput="this.value=this.value.toUpperCase().replace(/[^ABCDX]/g,'')">
            <input id="corrp-${jid}" type="number" min="1" max="99" placeholder="Pág"
                   style="width:70px;padding:8px;background:#1a1a1a;color:#fff;
                          border:2px solid #2a2a2a;border-radius:6px;text-align:center">
            <button onclick="sendCorrection('${jid}')"
                    style="background:#fd7e14;color:#fff;border:none;border-radius:6px;
                           padding:9px 14px;font-weight:700;cursor:pointer">📤 Enviar corrección</button>
            <span id="corrs-${jid}" style="font-size:11px;color:#888"></span>
        </div>
    </div>`;

    return simBannerHtml + mediaHtml + ocrHtml + respsHtml + corrHtml;
}

// Toggle de visibilidad de la matriz de votos en el detalle del job.
function toggleFusionMatrix(jid) {
    const el = document.getElementById(`fusion-matrix-${jid}`);
    if(!el) return;
    el.style.display = (el.style.display === 'none' || !el.style.display) ? 'block' : 'none';
}

// Matriz de votos pregunta-a-pregunta: filas = IAs, columnas = preguntas.
// Resalta en verde las letras que coinciden con la fusión, en rojo las que disienten,
// en gris las X / sin opinión. La fila inferior es la fusión final.
function buildFusionMatrix(j) {
    const merged    = (j.merged_answer || '').toUpperCase();
    const responses = j.responses || {};
    const provs     = (j._providers && j._providers.length)
                     ? j._providers : Object.keys(responses);
    if(provs.length === 0) {
        return '<div style="color:#666;font-size:11px;padding:6px">Sin datos de votos.</div>';
    }
    // Longitud objetivo: la mayor entre fusión, expected, y la respuesta más larga.
    let maxLen = j.expected_questions || merged.length || 0;
    for(const prov of provs) {
        const a = (responses[prov] && responses[prov].answer) || '';
        if(a.length > maxLen) maxLen = a.length;
    }
    if(maxLen === 0) {
        return '<div style="color:#666;font-size:11px;padding:6px">Aún sin respuestas para votar.</div>';
    }

    let html = '<div style="overflow-x:auto;border:1px solid #2a2a2a;border-radius:6px;'
             + 'padding:8px;background:#0d0d0d">'
             + '<div style="font-size:10px;color:#888;padding-bottom:6px">'
             + '📊 Matriz de votos · una columna por pregunta. '
             + '<span style="color:#5d8">verde</span>=coincide con fusión · '
             + '<span style="color:#dc3545">rojo</span>=disiente · '
             + '<span style="color:#888">·</span>=sin opinión (X o no respondió).'
             + '</div>'
             + '<table style="border-collapse:collapse;font-family:monospace;font-size:11px">'
             + '<thead><tr><th style="text-align:left;padding:3px 8px;color:#888">IA</th>';
    for(let i = 0; i < maxLen; i++) {
        html += `<th style="padding:3px 5px;color:#888;min-width:18px;text-align:center">${i+1}</th>`;
    }
    html += '<th style="padding:3px 8px;color:#888;text-align:right">estado</th></tr></thead><tbody>';

    for(const prov of provs) {
        const r = responses[prov] || {};
        const label = (typeof PROVIDER_LABELS!=='undefined' && PROVIDER_LABELS[prov])
                     || prov.toUpperCase();
        const ans = (r.answer || '').toUpperCase();
        let estado = '—';
        if(r.status === 'no_key') estado = '🔑';
        else if(r.ok)            estado = '✓';
        else if(r.status === 'error') estado = '❌';
        html += `<tr><td style="padding:3px 8px;color:#9cf;white-space:nowrap">${label}</td>`;
        for(let i = 0; i < maxLen; i++) {
            const ch = (i < ans.length) ? ans[i] : '';
            const m  = (i < merged.length) ? merged[i] : '';
            let color = '#666';
            let bg    = 'transparent';
            let disp  = ch || '·';
            if(!ch || ch === 'X') { color = '#666'; disp = ch || '·'; }
            else if(!m)            { color = '#aaa'; }
            else if(ch === m)      { color = '#5d8'; bg = '#0d1a0d'; }
            else                   { color = '#dc3545'; bg = '#1a0d0d'; }
            html += `<td style="padding:3px 5px;text-align:center;color:${color};background:${bg};font-weight:700">${disp}</td>`;
        }
        html += `<td style="padding:3px 8px;text-align:right;color:#888">${estado}</td></tr>`;
    }
    // Fila de la fusión (resultado final)
    html += '<tr style="border-top:1px dashed #2a2a2a">'
          + '<td style="padding:4px 8px;color:#5d8;font-weight:700">🔀 FUSIÓN</td>';
    for(let i = 0; i < maxLen; i++) {
        const m = (i < merged.length) ? merged[i] : '·';
        html += `<td style="padding:4px 5px;text-align:center;color:#fff;font-weight:700;background:#1a3a1a">${m}</td>`;
    }
    html += '<td></td></tr></tbody></table></div>';
    return html;
}

// ── Guardar la respuesta editada de UN provider ──────────────────────────────
async function saveProviderResponse(jid, prov) {
    const el = document.getElementById(`resp-${jid}-${prov}`);
    if(!el) return;
    const ans = (el.value || '').toUpperCase().replace(/[^ABCDX]/g, '');
    if(!ans) { toast('⚠ Vacío — escribe letras A/B/C/D/X'); return; }
    try {
        const r = await fetch(`/api/response/${jid}/${prov}`, {
            method:  'PATCH',
            headers: {'Content-Type':'application/json','X-Api-Key':KEY},
            body:    JSON.stringify({answer: ans}),
        });
        let data = {}; try { data = await r.json(); } catch(_) {}
        if(!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
        // Actualizar la celda de fusión en sitio (sin recargar todo)
        const mEl = document.getElementById(`merged-${jid}`);
        if(mEl && data.merged_answer) mEl.textContent = data.merged_answer.toUpperCase();
        toast(`✓ ${prov}: ${data.old||'?'} → ${data.new}`);
    } catch(e) {
        toast('❌ ' + e.message);
    }
}

// ── Editor de correcciones en historial ──────────────────────────────────────
// Toggle de la fila expandida del historial. La primera vez que se abre carga
// el expediente completo del job via /api/partial/JID (que sí incluye raws de
// IA + transcripciones OCR completas, a diferencia del /api/jobs ligero).
// Hits sucesivos solo togglean visibilidad (cacheado en el DOM).
async function toggleCorr(jid, defaultAns) {
    const row = document.getElementById('corrrow-' + jid);
    if(!row) return;
    const opening = row.style.display === 'none' || !row.style.display;
    row.style.display = opening ? '' : 'none';
    if(!opening) return;            // cierre: no hace falta cargar nada
    if(row.dataset.loaded === '1') return;  // ya cargado en una apertura previa
    const td = row.querySelector('td');
    if(!td) return;
    td.innerHTML = '<div style="color:#888;text-align:center;padding:18px;font-size:12px">'
                 + '⏳ Cargando expediente del job...</div>';
    try {
        const r = await fetch('/api/partial/' + jid + '?key=' + encodeURIComponent(KEY),
                              { headers: { 'X-Api-Key': KEY } });
        if(!r.ok) {
            let detail = '';
            try { detail = (await r.json()).detail || ''; } catch(_) {}
            // 404 es un caso esperado: el job expiró (>1h) o el servidor reinició
            // y el JOBS dict en RAM se vació. Lanzamos un error con código para
            // que el catch lo distinga del resto y muestre un mensaje claro.
            const err = new Error('HTTP ' + r.status + (detail ? ' · ' + detail : ''));
            err.httpStatus = r.status;
            throw err;
        }
        const fullJob = await r.json();
        td.innerHTML = buildHistoryDetail(fullJob, defaultAns);
        row.dataset.loaded = '1';
    } catch(e) {
        // Render defensivo: si falla la carga, mostrar error con botón reintentar
        // que limpia el cache y vuelve a llamar a toggleCorr.
        const safeAns = String(defaultAns||'').replace(/'/g, '');
        // Si es 404 (job purgado / expirado), explicamos qué pasó en vez del HTTP
        // crudo. Reintentar no va a recuperar el job; sugerimos los datos básicos
        // que sí están en el listado del historial (answer, estado, IA fusión).
        if(e.httpStatus === 404) {
            td.innerHTML =
                '<div style="color:#e0c060;padding:14px;font-size:12px;line-height:1.5">'
              + '<div style="font-weight:700;margin-bottom:6px">📭 Expediente ya no disponible</div>'
              + '<div style="color:#aaa">Este job ya no está en memoria del servidor. Causas habituales:</div>'
              + '<ul style="color:#aaa;margin:6px 0 8px 18px;padding:0">'
              + '<li>Tiene más de 1h (TTL) — se purgó automáticamente.</li>'
              + '<li>El servicio se reinició (re-deploy de Render) y el dyno free perdió el disco.</li>'
              + '<li>Superó el cap de 150 jobs y se descartó por ser de los más antiguos.</li>'
              + '</ul>'
              + '<div style="color:#888;font-size:11px">Los datos básicos (respuesta, estado, IA fusión) siguen visibles en la fila del historial. '
              + 'El detalle completo (raws de IA, OCR, bloque Tavily) se perdió.</div>'
              + '</div>';
            return;
        }
        td.innerHTML =
            '<div style="color:#dc3545;padding:14px;font-size:12px">'
          + '⚠ No se pudo cargar el expediente: ' + (e.message || 'error desconocido') + '. '
          // FIX: usar &quot; HTML entities en lugar de \' — el JS exterior usa
          // comillas simples como delimitador, así que cada \' literal dentro
          // CERRABA el string JS y rompía el parseo de TODO el <script>. Sin
          // parsear, autoLoadConfig() ni loadConfig() corrían → panel sin keys.
          // Con &quot;, el navegador parsea HTML → JS recibe r.style.display="none"
          // (comillas dobles dentro de string ' es legal y no choca).
          + '<a onclick="(function(r){r.dataset.loaded=0;r.style.display=&quot;none&quot;;'
          +     'toggleCorr(&quot;' + jid + '&quot;,&quot;' + safeAns + '&quot;);})(this.closest(&quot;tr&quot;))" '
          +    'style="color:#4d9eff;cursor:pointer;text-decoration:underline">↻ Reintentar</a>'
          + '</div>';
    }
}

async function sendCorrection(jid) {
    const ansEl = document.getElementById('corra-' + jid);
    const pgEl  = document.getElementById('corrp-' + jid);
    const stEl  = document.getElementById('corrs-' + jid);
    const ans = (ansEl?.value || '').toUpperCase().replace(/[^ABCDX]/g, '');
    const page = pgEl?.value ? parseInt(pgEl.value) : null;
    if(!ans) { if(stEl) stEl.textContent = '⚠ Escribe la corrección'; return; }
    try {
        const r = await fetch('/correction/' + jid, {
            method:  'POST',
            headers: {'Content-Type': 'application/json', 'X-Api-Key': KEY},
            body:    JSON.stringify({answer: ans, page: page})
        });
        if(!r.ok) throw new Error((await r.json()).detail || r.status);
        const data = await r.json();
        if(stEl) stEl.textContent = '✅ Encolada · ' + data.answer + (page ? ' · pág ' + page : '');
        toast('📤 Corrección enviada');
        setTimeout(() => { const row = document.getElementById('corrrow-' + jid);
                           if(row) row.style.display = 'none'; }, 2000);
    } catch(e) {
        if(stEl) stEl.textContent = '❌ ' + e.message;
    }
}

// ── Tira de frames extraídos del vídeo ───────────────────────────────────────
// Helper común usado por buildCard (card en vivo) y buildHistoryDetail (histórico).
// Renderiza los top-K frames más nítidos que el relay sacó del MP4 ANTES de
// mandarlos a los OCR de imagen. Se muestran DEBAJO del vídeo para que el
// usuario pueda compararlos visualmente con el clip original. Cada thumb es un
// <a> que abre el JPEG completo en nueva pestaña.
//
// Diseño:
//   - max-width 220px por thumb, height auto (mantenemos aspect ratio del JPEG
//     original → sin object-fit:cover que recortaba los bordes y perdía texto).
//   - flex-wrap → en pantallas estrechas hace dos filas.
//   - Si nFrames == 0 (job pre-extracción, modo imagen, video purgado),
//     devuelve string vacío (no contamina el layout).
function buildFramesStrip(jid, job) {
    const nFrames = job.extracted_frames_count || 0;
    const hasFused = !!job.has_fused_image;
    if(!nFrames && !hasFused) return '';

    let html = '';

    // ── Imagen fusionada (stacking + opcional Topaz Wonder 3) ────────────
    // Es la "foto casi-perfecta" del folio. Se muestra ARRIBA del strip de
    // frames porque es la que reciben Mistral/DeepSeek/GLM (los OCRs de
    // página única). Badge distinto según el origen: ✨ Topaz cuando subió
    // por la API, 🔧 stacking cuando es solo el pipeline local.
    if(hasFused) {
        const src = job.fused_image_source || '';
        const isTopaz = src === 'topaz_wonder3';
        const badge = isTopaz ? '✨ Imagen fusionada (Topaz Wonder 3)'
                              : '🔧 Imagen fusionada (stacking local)';
        const badgeColor = isTopaz ? '#d4a017' : '#9ad';
        const ms = (job.fused_image_topaz_ms || 0) + (job.fused_image_stack_ms || 0);
        const msLabel = ms > 0 ? ` · ${ms} ms` : '';
        html += `<div style="font-size:10px;color:${badgeColor};text-transform:uppercase;
                              letter-spacing:1px;padding:6px 2px 4px;margin-top:4px;
                              border-top:1px dashed #2a2a2a">
                    ${badge}<small style="color:#666;text-transform:none;letter-spacing:0">${msLabel} · va a Mistral/DeepSeek/GLM</small>
                 </div>
                 <div style="padding:4px 0 8px">
                    <a href="/api/fused_image/${jid}?key=${KEY}" target="_blank"
                       title="Click para abrir la imagen fusionada a tamaño completo">
                        <img src="/api/fused_image/${jid}?key=${KEY}"
                             style="max-width:100%;width:100%;height:auto;border-radius:6px;
                                    border:1px solid ${isTopaz ? '#4a3a1a' : '#2a4a3a'};
                                    background:#0a0a0a;display:block"
                             onerror="this.style.display='none'">
                    </a>
                 </div>`;
    }

    // ── Frames crudos (top-3 por Laplacian) ──────────────────────────────
    if(nFrames > 0) {
        html += `<div style="font-size:10px;color:#888;text-transform:uppercase;letter-spacing:1px;
                              padding:6px 2px 4px;border-top:1px dashed #2a2a2a">
                    📸 Frames crudos <small style="color:#666;text-transform:none;letter-spacing:0">
                    (${nFrames} · top sharpness · van a Claude/GPT multi-frame)</small>
                 </div>
                 <div style="display:flex;gap:8px;padding:4px 0 8px;flex-wrap:wrap">`;
        for(let i = 0; i < nFrames; i++) {
            const label = (i === 0) ? '🏆 #1 (top)' : '#' + (i + 1);
            html += `<a href="/api/frame/${jid}/${i}?key=${KEY}" target="_blank"
                        title="Click para abrir frame ${i+1} a tamaño completo"
                        style="display:flex;flex-direction:column;align-items:center;gap:3px;
                               text-decoration:none;color:#aaa;font-size:11px">
                        <img src="/api/frame/${jid}/${i}?key=${KEY}"
                             style="max-width:220px;width:100%;height:auto;border-radius:6px;
                                    border:1px solid #2a2a2a;background:#0a0a0a;display:block"
                             onerror="this.style.display='none';this.nextElementSibling.style.color='#666'">
                        <span>${label}</span>
                    </a>`;
        }
        html += `</div>`;
    }
    return html;
}

// ── Construir tarjeta (primera vez) ──────────────────────────────────────────
function buildCard(job) {
    const jid      = job.id;
    const merged   = job.merged_answer || '';
    const expected = job.expected_questions || 0;
    const cols     = Math.max(expected, merged.length, 1);
    const hora     = new Date(job.created * 1000).toLocaleTimeString('es');
    const dl       = job.review_deadline || 0;
    const tout     = job.review_timeout_seconds || 90;
    _STATE[jid]    = {dirty: false};

    const imgHtml = job.has_img
        ? `<div class="img-wrap">
               <img src="/api/image/${jid}?key=${KEY}" class="foto"
                    onclick="this.classList.toggle('big')" alt="Foto">
           </div>`
        : '';

    // Video MP4 reproducible — sólo si el job lo trae (mode VIDEO). Se carga
    // desde /api/video. Permite ver EXACTAMENTE lo que se envió a OCR.
    const videoHtml = job.has_video
        ? `<div class="img-wrap" style="background:#000;border:1px solid #2a2a2a;border-radius:8px;margin:6px 0">
               <video src="/api/video/${jid}?key=${KEY}" controls muted playsinline class="card-video"
                      onerror="this.parentNode.innerHTML='<div style=padding:10px;color:#888>⚠ Video no disponible (probablemente perdido tras reinicio)</div>'">
               </video>
               <div style="font-size:10px;color:#888;padding:4px 6px;text-align:right">
                   📹 ${Math.round((job.video_size_b64||0)*3/4/1024)} KB · MP4
               </div>
           </div>`
        : '';

    // Diagrama de contexto (si el cliente lo adjuntó). Se muestra ENCIMA del
    // vídeo/foto para hacer claro el orden con el que se envió al modelo:
    // IMAGEN 1 (referencia, este diagrama) + IMAGEN 2 (la "buena": vídeo/foto).
    // Si el cliente no envió diagrama (caso normal en exámenes sin caso práctico),
    // ni siquiera renderizamos la sección.
    const ctxSrc = job.context_source || 'none';
    const ctxSrcLabel = ctxSrc === 'mobile'
        ? 'móvil'
        : (ctxSrc === 'sticky' ? 'sticky relay' : 'sin contexto');
    const ctxHtml = job.has_context_image
        ? `<div class="img-wrap" style="border:1px solid #4a3a1a;background:#1a1408;border-radius:8px;margin:6px 0;padding:4px">
               <div style="font-size:10px;color:#d4a017;text-transform:uppercase;letter-spacing:1px;padding:2px 4px 4px">
                   🗂️ Diagrama de contexto · ✅ enviándose a TODAS las IAs <small style="color:#856404;text-transform:none;letter-spacing:0">(IMAGEN 1 — referencia del caso práctico, junto a la foto nueva · origen: ${ctxSrcLabel})</small>
               </div>
               <img src="/api/context_image/${jid}?key=${KEY}" class="foto"
                    style="max-width:100%;border-radius:6px;display:block;cursor:pointer"
                    onclick="this.classList.toggle('big')"
                    onerror="this.parentNode.innerHTML='<div style=padding:8px;color:#888;font-size:11px>⚠ Diagrama no disponible</div>'">
           </div>`
        : '';

    return `
    <div class="card card-pend" id="card-${jid}" data-jid="${jid}"
         data-status="${job.status}" data-merged="${merged}" data-cols="${cols}">
        <div class="card-head">
            <span class="badge b-pend" id="badge-${jid}">⏳ PROCESANDO</span>
            <span class="hora">${hora}</span>
            <span class="cd" id="cd-${jid}" data-deadline="${dl}" data-total="${tout}">⏱</span>
            <span class="sim-info" id="sim-${jid}">${renderSimInfo(job)}</span>
        </div>
        <div class="phase-bar" id="phase-${jid}" style="font-size:11px;color:#9ad;padding:4px 8px;background:#0d1620;border-radius:5px;margin:4px 0">
            ${renderPhase(job)}
        </div>
        <div class="card-split">
            <div class="media-col">
                ${ctxHtml}${videoHtml}${imgHtml}
                <div class="frames-strip" id="frames-${jid}">${buildFramesStrip(jid, job)}</div>
            </div>
            <div class="ia-col">
                <div class="review-area" id="review-${jid}"></div>
                <div class="ia-section" id="ias-${jid}">${buildIaSection(jid, job)}</div>
            </div>
        </div>
        <div class="complice-wrap">
            <div class="complice-label">${job.has_video ? '✋ Respuesta final — resumen editable · corrige pregunta a pregunta arriba ☝️' : '✋ Tu respuesta (Cómplice) — toca para cambiar'}</div>
            <div class="keys-row" id="keys-${jid}">${buildKeys(jid, merged, cols)}</div>
            <input class="complice-text" id="txt-${jid}" type="text"
                   value="${merged}" placeholder="O escribe aquí: ABCDA..."
                   oninput="onTextInput('${jid}', this)"
                   maxlength="50" autocomplete="off" spellcheck="false">
        </div>
        <div class="actions-row">
            <label class="check-mini" title="Marca si este examen lleva diagrama / caso práctico: el relay enviará la mejor imagen al móvil para que la adjunte como contexto (IMAGEN 1) a las PRÓXIMAS preguntas.">
                <input type="checkbox" id="img-${jid}" ${job.has_image ? 'checked' : ''}>
                <span>🖼️ Imagen en pregunta (enviar como contexto)</span>
            </label>
            <button class="btn-sm" onclick="resetToFusion('${jid}')" type="button">↺ Fusión</button>
            <button class="btn-sm btn-sm-red" onclick="clearAll('${jid}')" type="button">✕ Limpiar</button>
        </div>
        <button class="btn-send" id="send-${jid}" onclick="sendToMobile('${jid}')" type="button">
            <span class="send-label">
                <span class="send-label-main">✅ ENVIAR AL MÓVIL</span>
                <span class="send-label-sub">🔀 Fusión IA · edita arriba para cambiar</span>
            </span>
            <span class="send-prev" id="prev-${jid}">${merged || '---'}</span>
        </button>
    </div>`;
}

// Renderiza la fase actual del job en una sola línea legible.
// Para videos hace doble servicio: además de la fase, muestra cuántos
// OCRs llevamos y cuántos analyzers están corriendo.
// Pinta la SIM elegida + dBm que envió el móvil con la foto/video. El móvil mide
// la señal de ambas SIM con la radio encendida (SimCoverageSelector.kt) y manda
// la elegida en /ask. Si el campo no viene (cliente viejo, debugSinAvion, no hay
// READ_PHONE_STATE), no pintamos nada (string vacío → el <span> queda invisible).
function renderSimInfo(job) {
    const op = job.sim_operator;
    const dbm = job.sim_dbm;
    if(op == null && dbm == null) return '';
    // Colorimetría rápida por dBm: -85 o mejor verde, -85..-100 amarillo, peor rojo.
    let color = '#9ad';
    if(typeof dbm === 'number') {
        if(dbm >= -85) color = '#7dd87d';
        else if(dbm >= -100) color = '#e8c547';
        else color = '#e87d7d';
    }
    const escape = s => String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')
                                      .replace(/>/g,'&gt;').replace(/"/g,'&quot;')
                                      .replace(/'/g,'&#39;');
    const opTxt  = escape(op || '?');
    const dbmTxt = (typeof dbm === 'number') ? `${dbm} dBm` : '— dBm';
    const slotTxt = (typeof job.sim_slot === 'number') ? ` · slot${job.sim_slot}` : '';
    const summary = job.sim_summary ? escape(job.sim_summary) : '';
    const title = summary ? `title="${summary}"` : '';
    return `<span ${title} style="font-size:11px;color:${color};margin-left:6px;padding:2px 6px;
                   border:1px solid #2a2a2a;border-radius:4px;background:#0d1620;
                   font-family:monospace">📶 ${opTxt} ${dbmTxt}${slotTxt}</span>`;
}

function renderPhase(job) {
    const phase = job.phase || (job.has_video ? 'received' : 'analysis_running');
    const PHASE_LABELS = {
        'received':         '📥 RECIBIDO (video)',
        'ocr_running':      '🔍 OCR DEL VIDEO',
        'ocr_done':         '✅ OCR COMPLETO',
        'analysis_running': '🧠 ANALIZANDO (texto)',
        'analysis_done':    '✅ ANÁLISIS COMPLETO',
        'awaiting_review':  '✏️ ESPERANDO REVISIÓN',
        'done':             '✅ DONE',
        'error_ocr':        '❌ OCR FALLÓ',
        'error_pipeline':   '❌ PIPELINE FALLÓ',
    };
    let label = PHASE_LABELS[phase] || phase;

    // Detalle adicional según fase
    let extra = '';
    if(phase === 'ocr_running') {
        // Buscar la última entrada de phase_history para conocer progreso
        const hist = job.phase_history || [];
        for(let i = hist.length - 1; i >= 0; i--) {
            const h = hist[i];
            if(h.phase === 'ocr_running' && h.ocr_done !== undefined) {
                extra = ` (${h.ocr_done}/${h.ocr_total} listos)`;
                break;
            }
        }
    } else if(phase === 'ocr_done') {
        const hist = job.phase_history || [];
        for(let i = hist.length - 1; i >= 0; i--) {
            const h = hist[i];
            if(h.phase === 'ocr_done') {
                extra = ` · ${h.ocr_chars || 0} chars · ${h.ocr_ok_count}/${h.ocr_total} OK`;
                break;
            }
        }
    } else if(phase === 'analysis_running') {
        const ana = (job._providers || []).join(', ');
        if(ana) extra = ` [${ana}]`;
    }
    return `<b>FASE:</b> ${label}${extra}`;
}

// Catálogo de proveedores conocidos y su etiqueta humana en el panel.
// Mantener sincronizado con _PROVIDERS y _OCR_PROVIDERS en main.py.
const PROVIDER_LABELS = {
    // Analyzers (fase 2 en video, fase única en imagen)
    'gpt':        'GPT',
    'claude':     'Claude',
    'gemini':     'Gemini',
    'deepseek':   'DeepSeek',
    'mistral':    'Mistral',
    'nvidia':     '🟢 NVIDIA Nemotron',
    // OCRs (fase 1, sólo video)
    'qwen_ocr':      '🎬 Qwen OCR',
    'gemini_ocr':    '🎬 Gemini OCR',
    'kimi_ocr':      '🎬 Kimi OCR',
    'mimo_ocr':      '🎬 MiMo OCR',
    'anthropic_ocr': '🎬 Claude OCR',
    'openai_ocr':    '🎬 GPT-4o OCR',
    'mistral_ocr':   '📄 Mistral OCR',     // OCR puro (page-based, top-1 frame)
    'deepseek_ocr':  '📄 DeepSeek OCR',    // OCR puro
    'glm_ocr':       '📄 GLM-OCR (Z.AI)',  // OCR puro
};

function buildIaSection(jid, job) {
    const merged   = job.merged_answer || '';
    const expected = job.expected_questions || 0;
    const cols     = Math.max(expected, merged.length, 1);
    const resp     = job.responses || {};
    let html = '';

    // ── Fase 1: OCRs (sólo videos) ──────────────────────────────────────
    if(job.has_video) {
        const ocrResults = job.ocr_results || {};

        // Los frames extraídos NO se renderizan aquí — el helper buildFramesStrip(jid, job)
        // los pinta debajo del <video> dentro de media-col en buildCard / buildHistoryDetail.
        // Así el usuario los compara visualmente con el vídeo, no a la derecha pegados a OCRs.

        // Los OCRs que corren en paralelo en fase 1 (ver _OCR_PROVIDERS).
        // MiMo (Xiaomi) se añadió en el flujo full-modal; antes solo estaban los 3
        // primeros y MiMo no aparecía aunque hubiera key configurada. Claude OCR
        // se añadió posteriormente — usa la API de visión sobre un frame del video.
        const ocrProvs = ['qwen_ocr', 'gemini_ocr', 'kimi_ocr', 'mimo_ocr', 'anthropic_ocr', 'openai_ocr', 'mistral_ocr', 'deepseek_ocr', 'glm_ocr'];
        // Mostrar header sólo si hay al menos info de un OCR esperable
        html += `<div class="ocr-header" style="font-size:10px;color:#888;
                       text-transform:uppercase;letter-spacing:1px;padding:6px 4px 2px">
                    📝 OCR del video (fase 1)
                 </div>`;
        for(const prov of ocrProvs) {
            html += buildOcrRow(jid, prov, PROVIDER_LABELS[prov] || prov, ocrResults[prov]);
        }
        // Fila especial: el OCR fusionado por consenso (server-side). Muestra
        // K preguntas únicas y permite ver el texto canónico que se envió a los
        // analyzers. Si la fusión falló, aparece en amarillo indicando fallback.
        html += buildOcrFusionRow(jid, job);
        html += `<div class="ocr-header" style="font-size:10px;color:#888;
                       text-transform:uppercase;letter-spacing:1px;padding:8px 4px 2px">
                    🧠 Analyzers (fase 2 · sobre OCR fusionado)
                 </div>`;
    }

    // ── Fase 2 (o única): analyzers ────────────────────────────────────
    // Usamos job._providers para conocer EXACTAMENTE qué analyzers corren
    // en este job (modo IMAGEN excluye deepseek/mistral, modo VIDEO los incluye).
    // Si por algún motivo falta, hacemos fallback al catálogo completo.
    const provs = (job._providers && job._providers.length)
        ? job._providers
        : ['gpt','claude','gemini','deepseek','mistral','nvidia'];
    for(const prov of provs) {
        const label = PROVIDER_LABELS[prov] || prov.toUpperCase();
        html += buildIaRow(jid, prov, label, resp[prov], merged, cols);
    }
    // NB: la fila de Fusión ya NO va aquí — es la respuesta que se envía por
    // defecto si el cómplice no edita, así que está integrada visualmente en
    // el propio botón "ENVIAR AL MÓVIL" (mira buildCard).
    return html;
}

// Fila para un OCR provider. Estados: waiting (aún no corrió), ok (con chars),
// error (con motivo). NO tiene botones de envío — es solo informativo.
function buildOcrRow(jid, prov, label, r) {
    const id = `ocr-${prov}-${jid}`;
    if(!r) {
        return `<div class="ia-row" id="${id}">
                    <span class="ia-name">${label}</span>
                    <span class="ia-wait">⏳ esperando OCR...</span>
                </div>`;
    }
    const ms = r.ms ? `<small>${ms_fmt(r.ms)}</small>` : '';
    if(r.ok) {
        // chars/preview: el backend NUEVO precomputa r.chars + r.preview en el
        // slim (/api/jobs strippea r.text para no inflar payload). Si el backend
        // está corriendo una versión vieja (pre-precompute), `r.chars` no llega
        // y `r.text` está strippeado → ambos undefined.
        let charsLabel, charsVal;
        if(r.chars != null) {
            charsVal = r.chars;
            charsLabel = `${charsVal} chars`;
        } else if(r.text != null) {
            charsVal = r.text.length;
            charsLabel = `${charsVal} chars`;
        } else {
            charsVal = null;
            charsLabel = 'OK';
        }
        if(r.text) {
            window._OCR_TEXTS = window._OCR_TEXTS || {};
            window._OCR_TEXTS[`${jid}|${prov}`] = r.text;
        }
        const previewSrc = r.text || r.preview || r.preview_diag || '';
        const preview = previewSrc.slice(0, 60).replace(/[<>&"]/g, '');
        // Calidad: el backend marca quality=empty | no_legible | ok según el
        // texto devuelto. Si no llega quality (backend viejo o cubrir caso),
        // inferimos por chars/preview. NUNCA mostramos ✅ verde brillante para
        // 0 chars o NO_LEGIBLE — eso era engañoso (parecía éxito real).
        let quality = r.quality;
        if(!quality) {
            const previewUpper = (previewSrc || '').trim().toUpperCase();
            if(charsVal === 0) quality = 'empty';
            else if(previewUpper === 'NO_LEGIBLE' || previewUpper.startsWith('NO_LEGIBLE')) quality = 'no_legible';
            else quality = 'ok';
        }
        let badge, badgeColor;
        if(quality === 'empty') {
            badge = `⚠️ vacío`;
            badgeColor = '#e0c060';   // ámbar — preocupante
        } else if(quality === 'no_legible') {
            badge = `⓵ NO_LEGIBLE`;
            badgeColor = '#888';      // gris — informativo
        } else {
            badge = `✅ ${charsLabel}`;
            badgeColor = '#5d8';      // verde — éxito real
        }
        return `<div class="ia-row" id="${id}" style="align-items:center">
                    <span class="ia-name">${label}${ms}</span>
                    <span style="color:${badgeColor};font-weight:600">${badge}</span>
                    <span style="color:#aaa;font-size:11px;font-family:monospace;
                                 overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                                 max-width:280px;flex:1">
                        ${preview}${(charsVal != null && charsVal > 60) ? '…' : ''}
                    </span>
                    <button onclick="showOcrText('${jid}','${prov}','${label.replace(/[\\'`]/g,'')}')"
                            style="background:#0d6efd;color:#fff;border:none;border-radius:4px;
                                   padding:3px 10px;font-size:11px;font-weight:700;cursor:pointer"
                            title="Ver transcripción completa">📜 Ver</button>
                </div>`;
    } else {
        const err = (r.error || 'error').slice(0, 80);
        return `<div class="ia-row" id="${id}">
                    <span class="ia-name">${label}${ms}</span>
                    <span class="ia-err" title="${err}">❌ ${err}</span>
                </div>`;
    }
}

// Fila para el OCR FUSIONADO (resultado del consenso server-side de las N OCRs).
// Estados posibles:
//   - waiting:   aún no hay fusión calculada (fase OCR sin terminar)
//   - ok+used:   fusión exitosa y se usó para los analyzers (verde)
//   - ok+fallback: fusión calculada pero no se usó (fallback a <readings> crudo, amarillo)
//   - empty:     ningún OCR parseable, fusión vacía (gris)
function buildOcrFusionRow(jid, job) {
    const id = `ocr-fusion-${jid}`;
    const fusedText = job.ocr_fused_text || '';
    const stats     = job.ocr_fusion_stats || {};
    const used      = job.ocr_fusion_used === true;
    const ocrResults = job.ocr_results || {};
    // Si ningún OCR ha terminado aún, mostramos estado "esperando"
    const anyOcrDone = Object.values(ocrResults).some(r => r && (r.ok || r.error));
    if(!anyOcrDone) {
        return `<div class="ia-row" id="${id}" style="background:#16181c;border-left:3px solid #444">
                    <span class="ia-name">🧬 Fusión OCR</span>
                    <span class="ia-wait">⏳ esperando OCRs...</span>
                </div>`;
    }
    // Caso: fusión hecha y usada → verde brillante
    if(fusedText && used) {
        // chars REAL: el backend trunca ocr_fused_text a 200 chars en el listado
        // y manda el len original en ocr_fused_text_full_chars. Usar este último
        // para no mostrar siempre "200 chars" cuando el original era más largo.
        const chars = (job.ocr_fused_text_full_chars != null)
                        ? job.ocr_fused_text_full_chars
                        : fusedText.length;
        const k = stats.clusters || 0;
        const parsedMap = stats.parsed_per_ocr || {};
        const parsedList = Object.entries(parsedMap)
            .map(([p,n]) => `${p.replace('_ocr','')}:${n}`).join(' ');
        // Cachear SOLO la versión completa (el listado slim trunca a 200 chars y
        // marca _ocr_fused_truncated=true). Si cacheamos el truncado, showOcrText
        // lo lee de cache y NO lazy-fetchea el completo → popup queda cortado.
        // El bug histórico: aquí escribía sin guard → popup mostraba "200 chars".
        window._OCR_TEXTS = window._OCR_TEXTS || {};
        if(!job._ocr_fused_truncated) {
            window._OCR_TEXTS[`${jid}|fusion`] = fusedText;
        }
        const preview = fusedText.slice(0, 60).replace(/[<>&"]/g, '');
        return `<div class="ia-row" id="${id}" style="background:#0d1a0d;border-left:3px solid #5d8;align-items:center">
                    <span class="ia-name" style="color:#7df097">🧬 Fusión OCR <small style="color:#888;font-weight:400">→ analyzers</small></span>
                    <span style="color:#5d8;font-weight:600">✅ ${k} preguntas · ${chars} chars</span>
                    <span style="color:#aaa;font-size:11px;font-family:monospace;
                                 overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                                 max-width:240px;flex:1" title="parsed: ${parsedList}">
                        ${preview}${chars > 60 ? '…' : ''}
                    </span>
                    <button onclick="showOcrText('${jid}','fusion','Fusión OCR (consenso)')"
                            style="background:#198754;color:#fff;border:none;border-radius:4px;
                                   padding:3px 10px;font-size:11px;font-weight:700;cursor:pointer"
                            title="Ver transcripción fusionada completa">📜 Ver</button>
                </div>`;
    }
    // Caso: fusión calculada pero NO usada → fallback a <readings>, color amarillo
    if(fusedText && !used) {
        const chars = (job.ocr_fused_text_full_chars != null)
                        ? job.ocr_fused_text_full_chars
                        : fusedText.length;
        const k = stats.clusters || 0;
        // Mismo guard que la rama "used": NO cachear texto truncado o el popup se
        // queda con la versión cortada y NO hace lazy-fetch.
        window._OCR_TEXTS = window._OCR_TEXTS || {};
        if(!job._ocr_fused_truncated) {
            window._OCR_TEXTS[`${jid}|fusion`] = fusedText;
        }
        return `<div class="ia-row" id="${id}" style="background:#1a1a0d;border-left:3px solid #d4a017;align-items:center">
                    <span class="ia-name" style="color:#e0c060">🧬 Fusión OCR</span>
                    <span style="color:#e0c060;font-weight:600">⚠️ ${k} preguntas · no usada (fallback)</span>
                    <button onclick="showOcrText('${jid}','fusion','Fusión OCR (calculada pero no usada)')"
                            style="background:#d4a017;color:#000;border:none;border-radius:4px;
                                   padding:3px 10px;font-size:11px;font-weight:700;cursor:pointer">📜 Ver</button>
                </div>`;
    }
    // Caso: fusión vacía o con error → gris
    const errInfo = stats.error ? `: ${stats.error}` : '';
    return `<div class="ia-row" id="${id}" style="background:#1a0d0d;border-left:3px solid #666">
                <span class="ia-name" style="color:#999">🧬 Fusión OCR</span>
                <span style="color:#aaa">⚪ sin fusión (analyzers reciben &lt;readings&gt; crudo)${errInfo}</span>
            </div>`;
}

// Modal genérico para mostrar texto completo (OCR transcrito o raw de IA).
// Se monta lazy la primera vez; lo reusan showOcrText y showRawResponse.
function showFullText(title, meta, txt) {
    let modal = document.getElementById('ocr-modal');
    if(!modal) {
        modal = document.createElement('div');
        modal.id = 'ocr-modal';
        modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.85);'
                            + 'z-index:9999;display:none;align-items:center;justify-content:center;padding:24px';
        modal.innerHTML = `
            <div style="background:#1a1a1a;border:1px solid #333;border-radius:10px;
                        max-width:900px;max-height:85vh;width:100%;display:flex;flex-direction:column;overflow:hidden">
                <div style="display:flex;align-items:center;justify-content:space-between;
                            padding:12px 16px;background:#0d0d0d;border-bottom:1px solid #2a2a2a">
                    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
                        <span id="ocr-modal-title" style="font-weight:700;color:#fff">📜 Transcripción</span>
                        <span id="ocr-modal-meta" style="font-size:11px;color:#888"></span>
                    </div>
                    <div style="display:flex;gap:6px">
                        <button onclick="copyOcrText()" style="background:#0d6efd;color:#fff;border:none;
                                border-radius:4px;padding:5px 10px;font-size:11px;cursor:pointer">📋 Copiar</button>
                        <button onclick="document.getElementById('ocr-modal').style.display='none'"
                                style="background:#dc3545;color:#fff;border:none;border-radius:4px;
                                       padding:5px 10px;font-size:11px;cursor:pointer">✕ Cerrar</button>
                    </div>
                </div>
                <textarea id="ocr-modal-text" readonly
                          style="flex:1;background:#0d0d0d;color:#e7e7e7;border:none;padding:14px;
                                 font-family:'SF Mono','Consolas',monospace;font-size:12px;line-height:1.5;
                                 resize:none;outline:none;min-height:300px"></textarea>
            </div>`;
        document.body.appendChild(modal);
        // Cerrar al pulsar fondo (no el contenido)
        modal.addEventListener('click', e => { if(e.target === modal) modal.style.display='none'; });
    }
    document.getElementById('ocr-modal-title').textContent = title;
    document.getElementById('ocr-modal-meta').textContent  = meta;
    document.getElementById('ocr-modal-text').value        = txt || '(vacío)';
    modal.style.display = 'flex';
}

// Variante HTML del modal (en lugar de textarea, un div con innerHTML). Usado
// para vistas estructuradas con enlaces clickables (búsquedas Tavily, etc.).
// Modal separado para no compartir DOM con showFullText (que usa textarea).
function showFullTextHtml(title, meta, html) {
    let modal = document.getElementById('html-modal');
    if(!modal) {
        modal = document.createElement('div');
        modal.id = 'html-modal';
        modal.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.85);'
                            + 'z-index:9999;display:none;align-items:center;justify-content:center;padding:24px';
        modal.innerHTML = `
            <div style="background:#1a1a1a;border:1px solid #333;border-radius:10px;
                        max-width:900px;max-height:85vh;width:100%;display:flex;flex-direction:column;overflow:hidden">
                <div style="display:flex;align-items:center;justify-content:space-between;
                            padding:12px 16px;background:#0d0d0d;border-bottom:1px solid #2a2a2a">
                    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
                        <span id="html-modal-title" style="font-weight:700;color:#fff"></span>
                        <span id="html-modal-meta" style="font-size:11px;color:#888"></span>
                    </div>
                    <button onclick="document.getElementById('html-modal').style.display='none'"
                            style="background:#dc3545;color:#fff;border:none;border-radius:4px;
                                   padding:5px 10px;font-size:11px;cursor:pointer">✕ Cerrar</button>
                </div>
                <div id="html-modal-body" style="flex:1;overflow:auto;padding:14px;background:#0d0d0d;
                                                  color:#e7e7e7;font-family:'Segoe UI',system-ui,sans-serif;
                                                  font-size:13px;line-height:1.5"></div>
            </div>`;
        document.body.appendChild(modal);
        modal.addEventListener('click', e => { if(e.target === modal) modal.style.display='none'; });
    }
    document.getElementById('html-modal-title').textContent = title;
    document.getElementById('html-modal-meta').textContent  = meta;
    document.getElementById('html-modal-body').innerHTML    = html || '(vacío)';
    modal.style.display = 'flex';
}

// Cache helper: pide /api/partial/{jid} y devuelve el job completo.
// _PARTIAL_FETCHES dedupea peticiones concurrentes (si el usuario abre dos
// popups del mismo job en rápida sucesión, la 2ª espera al promise de la 1ª).
window._PARTIAL_FETCHES = window._PARTIAL_FETCHES || {};
async function fetchPartialJob(jid) {
    if(window._PARTIAL_FETCHES[jid]) return window._PARTIAL_FETCHES[jid];
    const p = (async () => {
        const r = await fetch('/api/partial/' + encodeURIComponent(jid)
                              + '?key=' + encodeURIComponent(KEY),
                              { headers: { 'X-Api-Key': KEY } });
        if(!r.ok) {
            let detail = '';
            try { detail = (await r.json()).detail || ''; } catch(_) {}
            throw new Error('HTTP ' + r.status + (detail ? ' · ' + detail : ''));
        }
        return await r.json();
    })();
    window._PARTIAL_FETCHES[jid] = p;
    // Liberamos el slot tras unos segundos para futuras llamadas (no cacheamos
    // el JSON entero — solo el promise en vuelo). Las extracciones individuales
    // sí se cachean en _OCR_TEXTS / _RAW_RESPONSES.
    p.finally(() => { setTimeout(() => { delete window._PARTIAL_FETCHES[jid]; }, 3000); });
    return p;
}

// Modal con la transcripción OCR completa. Si la cache local no la tiene
// (caso normal en el listado slim — el backend strippea ocr_results[*].text
// y trunca ocr_fused_text a 200 chars para no inflar el polling), hacemos
// lazy-fetch a /api/partial/{jid} y la cacheamos para próximos clicks.
async function showOcrText(jid, prov, label) {
    const key = `${jid}|${prov}`;
    let txt = (window._OCR_TEXTS && window._OCR_TEXTS[key]) || '';
    if(!txt) {
        showFullText(`📜 ${label}`,
                     `cargando... · job ${jid.slice(0,8)}`,
                     '⏳ Cargando transcripción completa desde el servidor...');
        try {
            const j = await fetchPartialJob(jid);
            if(prov === 'fusion') {
                txt = j.ocr_fused_text || '';
            } else {
                txt = ((j.ocr_results || {})[prov] || {}).text || '';
            }
            if(txt) {
                window._OCR_TEXTS = window._OCR_TEXTS || {};
                window._OCR_TEXTS[key] = txt;
            }
        } catch(e) {
            showFullText(`📜 ${label}`,
                         `error · job ${jid.slice(0,8)}`,
                         '⚠ No se pudo cargar la transcripción: ' + (e.message || 'error desconocido'));
            return;
        }
    }
    showFullText(`📜 ${label}`,
                 `${txt.length} chars · job ${jid.slice(0,8)}`,
                 txt);
}

// Modal con el bloque <internet_context> EXACTO que se inyectó al prompt del
// razonador. Si en el listado slim solo llegó truncado (>200 chars), hace
// lazy-fetch a /api/partial para conseguir el bloque completo.
async function showTavilyBlock(jid) {
    let txt = (window._TAVILY_BLOCKS && window._TAVILY_BLOCKS[jid]) || '';
    if(!txt) {
        showFullText(`🌐 Tavily · bloque inyectado`,
                     `cargando... · job ${jid.slice(0,8)}`,
                     '⏳ Cargando bloque <internet_context> completo desde el servidor...');
        try {
            const j = await fetchPartialJob(jid);
            txt = j.tavily_block || '';
            if(txt) {
                window._TAVILY_BLOCKS = window._TAVILY_BLOCKS || {};
                window._TAVILY_BLOCKS[jid] = txt;
            }
        } catch(e) {
            showFullText(`🌐 Tavily · bloque inyectado`,
                         `error · job ${jid.slice(0,8)}`,
                         '⚠ No se pudo cargar el bloque: ' + (e.message || 'error'));
            return;
        }
    }
    if(!txt) {
        showFullText(`🌐 Tavily · bloque inyectado`,
                     `vacío · job ${jid.slice(0,8)}`,
                     '(Tavily no inyectó nada — sin resultados utilizables o paso omitido)');
        return;
    }
    showFullText(`🌐 Tavily · bloque inyectado al prompt`,
                 `${txt.length} chars · job ${jid.slice(0,8)}`,
                 txt);
}

// Modal con la lista detallada de búsquedas Tavily: una sección por pregunta,
// query enviada, fuentes encontradas (title + url + snippet recortado).
function showTavilyQueries(jid) {
    const stats = (window._TAVILY_STATS && window._TAVILY_STATS[jid]) || null;
    if(!stats || !Array.isArray(stats.queries) || !stats.queries.length) {
        showFullText('🔎 Tavily · búsquedas',
                     `vacío · job ${jid.slice(0,8)}`,
                     '(no hay queries guardadas para este job)');
        return;
    }
    const escapeHtml = s => String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');  // A10: incluir comilla simple
    const blocks = stats.queries.map((q, i) => {
        const tag = q.ok
            ? `<span style="color:#0ea5e9">✓ ${q.n_sources} fuentes · ${ms_fmt(q.ms||0)}</span>`
            : `<span style="color:#dc3545">❌ ${escapeHtml(q.error || 'sin respuesta')}</span>`;
        const numLabel = (q.num != null) ? `P${q.num}` : `P?${i+1}`;
        const sources = (q.sources || []).map(s => {
            const title = escapeHtml((s.title || '(sin título)'));
            const url   = escapeHtml(s.url || '');
            const snip  = escapeHtml(s.snippet || '');
            return `<div style="margin:4px 0 4px 12px;padding:4px 8px;background:#0a0a0a;
                                 border-left:2px solid #0ea5e9;border-radius:3px">
                        <div style="color:#9cf;font-weight:600">${title}</div>
                        <div><a href="${url}" target="_blank" rel="noopener"
                                style="color:#0ea5e9;font-size:11px;text-decoration:none">${url}</a></div>
                        ${snip ? `<div style="color:#bbb;font-size:11px;margin-top:3px">${snip}</div>` : ''}
                    </div>`;
        }).join('');
        return `<div style="margin-bottom:10px;padding:6px 8px;background:#101010;border:1px solid #222;border-radius:5px">
                    <div style="display:flex;gap:8px;align-items:center;margin-bottom:4px">
                        <span style="background:#0ea5e9;color:#000;font-weight:700;padding:1px 6px;
                                     border-radius:3px;font-size:11px">${numLabel}</span>
                        ${tag}
                    </div>
                    <div style="color:#aaa;font-size:11px;font-family:monospace;margin-bottom:4px">
                        🔎 ${escapeHtml(q.query || '(query vacía)')}
                    </div>
                    ${sources || '<div style="color:#666;font-size:11px;font-style:italic">(sin fuentes)</div>'}
                </div>`;
    }).join('');
    showFullTextHtml('🔎 Tavily · búsquedas por pregunta',
                     `${stats.queries.length} preguntas · ${stats.n_with_sources || 0} con fuentes · ${ms_fmt(stats.elapsed_ms||0)}`,
                     blocks);
}

// Modal con la respuesta cruda completa de UN analyzer (incluye razonamiento,
// citas inline, texto antes de la extracción de letras, etc.). Cargado desde
// window._RAW_RESPONSES para no inflar el DOM con cada raw de cada IA del top-30.
// Igual que showOcrText: lazy-fetch desde /api/partial si la cache está vacía
// (el listado slim strippea responses[*].raw).
async function showRawResponse(jid, prov, label) {
    const key = `${jid}|${prov}`;
    let txt = (window._RAW_RESPONSES && window._RAW_RESPONSES[key]) || '';
    if(!txt) {
        showFullText(`💬 ${label} · respuesta completa`,
                     `cargando... · job ${jid.slice(0,8)}`,
                     '⏳ Cargando respuesta completa desde el servidor...');
        try {
            const j = await fetchPartialJob(jid);
            txt = ((j.responses || {})[prov] || {}).raw || '';
            if(txt) {
                window._RAW_RESPONSES = window._RAW_RESPONSES || {};
                window._RAW_RESPONSES[key] = txt;
            }
        } catch(e) {
            showFullText(`💬 ${label} · respuesta completa`,
                         `error · job ${jid.slice(0,8)}`,
                         '⚠ No se pudo cargar la respuesta: ' + (e.message || 'error desconocido'));
            return;
        }
    }
    showFullText(`💬 ${label} · respuesta completa`,
                 `${txt.length} chars · job ${jid.slice(0,8)}`,
                 txt);
}

function copyOcrText() {
    const ta = document.getElementById('ocr-modal-text');
    if(!ta) return;
    ta.select();
    try { navigator.clipboard.writeText(ta.value); toast('📋 Copiado'); }
    catch(_) { document.execCommand('copy'); toast('📋 Copiado'); }
}

function buildIaRow(jid, prov, label, r, merged, cols) {
    // Las filas individuales de IA NO envían — solo cargan al editor del
    // cómplice (botón 📋). El único botón de envío REAL es el verde grande
    // "✅ ENVIAR AL MÓVIL" al fondo del card.
    const id = `ia-${prov}-${jid}`;
    const ms = r && r.ms ? `<small>${ms_fmt(r.ms)}</small>` : '';
    let body = '';
    if(!r || r.status === 'waiting' || r.status === 'pending') {
        body = `<span class="ia-wait">⏳ esperando...</span>
                <button class="btn-use" disabled>📋</button>`;
    } else if(r.status === 'processing') {
        body = `<span class="ia-proc">⚡ procesando...</span>
                <button class="btn-use" disabled>📋</button>`;
    } else if(r.status === 'no_key') {
        // Sin key configurada: NO es un fallo, simplemente no participa.
        // Estilo gris/neutro para no confundir con un error real.
        body = `<span style="color:#888;font-size:11px;font-style:italic">
                    🔑 sin key configurada
                </span>
                <button class="btn-use" disabled>📋</button>`;
    } else if(r.ok && r.answer) {
        const ans = r.answer.replace(/'/g, "\\'");
        body = `<div class="cells">${buildCells(r.answer, cols, merged)}</div>
                <button class="btn-use" title="Cargar esta respuesta al editor"
                        onclick="useAns('${jid}','${ans}')">📋</button>`;
    } else {
        const err = ((r && r.error) || 'error').slice(0, 60);
        body = `<span class="ia-err" title="${err}">❌ ${err}</span>
                <button class="btn-use" disabled>📋</button>`;
    }
    return `<div class="ia-row" id="${id}">
                <span class="ia-name">${label}${ms}</span>
                ${body}
            </div>`;
}

function buildFusionRow(jid, merged, cols) {
    let body;
    if(merged) {
        const m = merged.replace(/'/g, "\\'");
        body = `<div class="cells">${buildCells(merged, cols, merged)}</div>
                <button class="btn-use"   title="Copiar al editor"
                        onclick="useAns('${jid}','${m}')">📋</button>
                <button class="btn-quick" title="Enviar fusión"
                        onclick="sendDirect('${jid}','${m}','Fusión')">📤</button>`;
    } else {
        body = `<span class="ia-wait">Esperando IAs...</span>
                <button class="btn-use"   disabled>📋</button>
                <button class="btn-quick" disabled>📤</button>`;
    }
    return `<div class="ia-row ia-fusion" id="ia-fusion-${jid}">
                <span class="ia-name">🔀 Fusión</span>
                ${body}
            </div>`;
}

function ms_fmt(ms) {
    if(ms < 1000) return ms + 'ms';
    return (ms/1000).toFixed(1) + 's';
}

function buildCells(ans, cols, merged) {
    const n = Math.max(cols, ans ? ans.length : 0);
    let html = '';
    for(let i = 0; i < n; i++) {
        const ch = (ans && i < ans.length) ? ans[i] : '?';
        const cls = 'c-' + (ch.match(/[ABCDX]/) ? ch : 'q');
        const diff = (merged && i < merged.length && merged[i] !== ch) ? ' diff' : '';
        html += `<span class="cell ${cls}${diff}">${ch}</span>`;
    }
    return html;
}

function buildKeys(jid, ans, cols) {
    let html = '';
    for(let i = 0; i < cols; i++) {
        const ch = (ans && i < ans.length) ? ans[i] : 'X';
        html += `<button type="button" class="key k-${ch}" data-i="${i}"
                         onclick="rotKey(this,'${jid}')">
                     <span class="key-num">${i+1}</span>
                     <span class="key-letter">${ch}</span>
                 </button>`;
    }
    return html;
}

// ── Actualizar tarjeta en sitio (sin destruir inputs) ────────────────────────
function updateCard(card, job) {
    const jid    = job.id;
    const merged = job.merged_answer || '';
    const expq   = job.expected_questions || 0;
    const cols   = Math.max(expq, merged.length, 1);

    // Cambio de estado
    if(card.dataset.status !== job.status) {
        card.dataset.status = job.status;
        const badge = document.getElementById('badge-' + jid);
        if(job.status === 'awaiting_review') {
            card.className = 'card card-rev';
            if(badge) { badge.className = 'badge b-rev'; badge.textContent = '✏️ REVISAR'; }
            // Actualizar deadline en contador
            const cd = document.getElementById('cd-' + jid);
            if(cd && job.review_deadline) {
                cd.dataset.deadline = job.review_deadline;
                cd.dataset.total    = job.review_timeout_seconds || 90;
            }
        }
    }

    // Actualizar la fase (siempre, incluso si status no cambió: phase es más granular)
    const phaseEl = document.getElementById('phase-' + jid);
    if(phaseEl) {
        const newPhase = renderPhase(job);
        if(phaseEl.innerHTML !== newPhase) phaseEl.innerHTML = newPhase;
    }

    // SIM + dBm — el móvil podría mandar la telemetría con un poco de delay si el
    // job se creó en una versión antigua sin el campo y se rellenó después; por
    // si acaso, re-renderizamos si cambió.
    const simEl = document.getElementById('sim-' + jid);
    if(simEl) {
        const newSim = renderSimInfo(job);
        if(simEl.innerHTML !== newSim) simEl.innerHTML = newSim;
    }

    // Actualizar la tira de frames del video — aparece cuando el relay pasa
    // de "received" a "ocr_running" (la pre-extracción rellena
    // extracted_frames_count). Y se actualiza otra vez cuando llega la
    // imagen fusionada (stacking → opcional Topaz). Hash simple para no
    // re-renderizar si nada cambió.
    const framesEl = document.getElementById('frames-' + jid);
    if(framesEl) {
        const want = (job.extracted_frames_count || 0);
        const fusedTag = (job.has_fused_image ? 'F:' + (job.fused_image_source || 'local') : 'F:none');
        const stamp = `${want}|${fusedTag}`;
        if(framesEl.dataset.stamp !== stamp) {
            framesEl.innerHTML = buildFramesStrip(jid, job);
            framesEl.dataset.stamp = stamp;
        }
    }

    // Actualizar filas de IA en sitio. Estrategia: si el conjunto de providers
    // efectivos ya está renderizado, actualizamos cada fila individualmente
    // (no destruimos los inputs). Si NO coincide (caso raro: server cambia
    // la lista por reinicio o cambio de modo), re-render completo de la sección.
    const resp = job.responses || {};
    const provs = (job._providers && job._providers.length)
        ? job._providers
        : ['gpt','claude','gemini','deepseek','mistral','nvidia'];

    // ¿Las filas existentes coinciden con los providers actuales?
    let needFullRebuild = false;
    for(const prov of provs) {
        if(!document.getElementById(`ia-${prov}-${jid}`)) {
            needFullRebuild = true; break;
        }
    }
    if(needFullRebuild) {
        const iasEl = document.getElementById('ias-' + jid);
        if(iasEl) iasEl.innerHTML = buildIaSection(jid, job);
    } else {
        // OCR rows (solo videos): actualizar si existen
        if(job.has_video) {
            const ocrResults = job.ocr_results || {};
            for(const prov of ['qwen_ocr', 'gemini_ocr', 'kimi_ocr', 'mimo_ocr', 'anthropic_ocr', 'openai_ocr', 'mistral_ocr', 'deepseek_ocr', 'glm_ocr']) {
                const rowEl = document.getElementById(`ocr-${prov}-${jid}`);
                if(!rowEl) continue;
                rowEl.outerHTML = buildOcrRow(jid, prov, PROVIDER_LABELS[prov] || prov, ocrResults[prov]);
            }
            // Refrescar también la fila de OCR fusionado (puede haber llegado
            // después de las OCRs individuales).
            const fusRowEl = document.getElementById(`ocr-fusion-${jid}`);
            if(fusRowEl) fusRowEl.outerHTML = buildOcrFusionRow(jid, job);
        }
        // Analyzer rows
        for(const prov of provs) {
            const rowEl = document.getElementById(`ia-${prov}-${jid}`);
            if(!rowEl) continue;
            const label = PROVIDER_LABELS[prov] || prov.toUpperCase();
            rowEl.outerHTML = buildIaRow(jid, prov, label, resp[prov], merged, cols);
        }
        // (Fila de Fusión eliminada — el preview vive ahora en el botón de envío.)
    }

    // Si el Cómplice no ha editado → sincronizar teclas con fusión
    const state = _STATE[jid] || (_STATE[jid] = {dirty: false});
    if(!state.dirty && merged && merged !== card.dataset.merged) {
        card.dataset.merged = merged;
        _setKeys(jid, merged);
        const txt = document.getElementById('txt-' + jid);
        if(txt && document.activeElement !== txt) txt.value = merged;
        _updPreview(jid);
    }

    // Ampliar teclas si llegaron más preguntas que las que había
    const currentCols = document.querySelectorAll(`#keys-${jid} .key`).length;
    if(cols > currentCols) {
        const keysEl = document.getElementById('keys-' + jid);
        if(keysEl) {
            const cur = _getAns(jid);
            keysEl.innerHTML = buildKeys(jid, cur.padEnd(cols, 'X'), cols);
            card.dataset.cols = cols;
        }
    }
}

// ── Countdown ─────────────────────────────────────────────────────────────────
function clsP(p) { return p < 0.25 ? 'cd-urg' : p < 0.5 ? 'cd-warn' : 'cd-ok'; }

function ticks() {
    const now = Date.now() / 1000;
    document.querySelectorAll('.cd[data-deadline]').forEach(el => {
        const dl  = parseFloat(el.dataset.deadline);
        const tot = parseFloat(el.dataset.total) || 90;
        if(!dl || dl <= 0) return;
        const left = Math.round(dl - now);
        if(left <= 0) { el.textContent = '⚡ auto'; el.className = 'cd cd-urg'; }
        else { el.textContent = '⏱ ' + left + 's'; el.className = 'cd ' + clsP(left/tot); }
    });
}

// ── Keys (tap para rotar A→B→C→D→X) ─────────────────────────────────────────
function rotKey(btn, jid) {
    const lEl = btn.querySelector('.key-letter');
    const next = _LET[(_LET.indexOf(lEl.textContent) + 1) % _LET.length];
    lEl.textContent = next;
    btn.className = btn.className.replace(/k-[ABCDX]/, 'k-' + next);
    _markDirty(jid);
    _syncTxt(jid);
    _updPreview(jid);
}

function _getAns(jid) {
    return Array.from(document.querySelectorAll(`#keys-${jid} .key .key-letter`))
               .map(k => k.textContent).join('');
}

function _setKeys(jid, ans) {
    const keys = document.querySelectorAll(`#keys-${jid} .key`);
    const n    = keys.length;
    const s    = (ans || '').padEnd(n, 'X').slice(0, n);
    keys.forEach((k, i) => {
        const ch = s[i] || 'X';
        k.querySelector('.key-letter').textContent = ch;
        k.className = k.className.replace(/k-[ABCDX]/, 'k-' + ch);
    });
    _updPreview(jid);
}

function _syncTxt(jid) {
    const t = document.getElementById('txt-' + jid);
    if(t) t.value = _getAns(jid);
}

function onTextInput(jid, input) {
    const v = input.value.toUpperCase().replace(/[^ABCDX]/g, '');
    input.value = v;
    _setKeys(jid, v);
    _markDirty(jid);
    _updPreview(jid);
}

function _updPreview(jid) {
    const p = document.getElementById('prev-' + jid);
    if(p) p.textContent = _getAns(jid) || '---';
}

function _markDirty(jid) { (_STATE[jid] || (_STATE[jid] = {})).dirty = true; }

// ── Usar / Reset / Clear ──────────────────────────────────────────────────────
function useAns(jid, ans) {
    _setKeys(jid, ans);
    const t = document.getElementById('txt-' + jid);
    if(t) t.value = ans;
    _markDirty(jid);
    _updPreview(jid);
    toast('Copiado: ' + ans);
}

function resetToFusion(jid) {
    const card   = document.getElementById('card-' + jid);
    const merged = card ? card.dataset.merged : '';
    _setKeys(jid, merged);
    const t = document.getElementById('txt-' + jid);
    if(t) t.value = merged;
    (_STATE[jid] || (_STATE[jid] = {})).dirty = false;
    _updPreview(jid);
}

function clearAll(jid) {
    const card = document.getElementById('card-' + jid);
    const cols = parseInt(card?.dataset.cols || '1', 10);
    const blank = 'X'.repeat(Math.max(1, cols));
    _setKeys(jid, blank);
    const t = document.getElementById('txt-' + jid);
    if(t) t.value = blank;
    _markDirty(jid);
    _updPreview(jid);
}

// ── Enviar DIRECTO una respuesta concreta (sin pasar por el editor) ─────────
async function sendDirect(jid, ans, label) {
    if(!ans) return;
    const card  = document.getElementById('card-' + jid);
    const imgEl = document.getElementById('img-' + jid);
    try {
        const r = await fetch('/result/' + jid, {
            method:  'PATCH',
            headers: {'Content-Type': 'application/json', 'X-Api-Key': KEY},
            body:    JSON.stringify({answer: ans, has_image: imgEl ? imgEl.checked : false})
        });
        if(!r.ok) throw new Error((await r.json()).detail || r.status);
        toast('✅ ' + label + ': ' + ans + (imgEl?.checked ? ' · 🖼️ imagen-contexto al móvil' : ''));
        if(card) { card.style.transition='opacity .35s'; card.style.opacity='0';
                    setTimeout(()=>card.remove(), 380); }
    } catch(e) {
        toast('❌ ' + e.message);
    }
}

// ── Enviar al móvil ───────────────────────────────────────────────────────────
async function sendToMobile(jid) {
    const ans = _getAns(jid);
    if(!ans || ans === '' ) { toast('No hay respuesta'); return; }
    const imgEl = document.getElementById('img-' + jid);
    const btn   = document.getElementById('send-' + jid);
    btn.disabled = true;
    btn.querySelector('span').textContent = 'ENVIANDO...';
    try {
        const r = await fetch('/result/' + jid, {
            method:  'PATCH',
            headers: {'Content-Type': 'application/json', 'X-Api-Key': KEY},
            body:    JSON.stringify({answer: ans, has_image: imgEl ? imgEl.checked : false})
        });
        if(!r.ok) throw new Error((await r.json()).detail || r.status);
        toast('✅ ' + ans + ' enviado' + (imgEl?.checked ? ' · 🖼️ imagen-contexto al móvil' : ''));
        const card = document.getElementById('card-' + jid);
        if(card) { card.style.transition='opacity .35s'; card.style.opacity='0';
                    setTimeout(()=>card.remove(), 380); }
    } catch(e) {
        btn.disabled = false;
        btn.querySelector('span').textContent = '✅ ENVIAR AL MÓVIL';
        alert('Error: ' + e.message);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// REVISIÓN POR PREGUNTA — cards de corrección del OCR fusionado.
// El revisor marca OK / mal-leída (→X) / borra / reordena cada pregunta. Cada
// cambio: (1) renumera y re-pinta las cards, (2) recompone la cadena final y la
// vuelca en la tira de teclas (fuente de _getAns → sendToMobile) + el input,
// (3) autosalva en el servidor (PATCH /api/review), que recompone job["answer"]
// para que el auto-aprobado entregue la versión revisada aunque no se pulse ENVIAR.
// ════════════════════════════════════════════════════════════════════════════
window._REVIEW = window._REVIEW || {};
var _REV_LETTERS = ['A','B','C','D','X'];

function _revEsc(s){ return String(s==null?'':s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;'); }

// Las cards salen EN CUANTO la fusión OCR está lista (fase ocr_done), sin
// esperar a la 1ª IA de razonamiento → el compañero compara foto y pregunta
// cuanto antes. Luego, en cada poll, se refrescan los votos de las IAs conforme
// van respondiendo, SIN pisar lo que el compañero ya haya corregido.
function _ocrReady(job){
    if(job.status==='awaiting_review') return true;
    var p = job.phase || '';
    return ['ocr_done','tavily_running','tavily_done','analysis_running',
            'analysis_done','awaiting_review'].indexOf(p) >= 0;
}
function _syncReview(job){
    if(!job || !job.has_video || !_ocrReady(job)) return;
    var host = document.getElementById('review-'+job.id);
    if(!host) return;
    var st = window._REVIEW[job.id];
    if(st && st.loading) return;
    if(st && st.loaded) loadReview(job.id, true);   // refresca votos (preserva ediciones)
    else loadReview(job.id, false);                  // carga inicial (reintenta si la fusión aún no está)
}

async function loadReview(jid, isRefresh){
    var st = window._REVIEW[jid] || (window._REVIEW[jid] = {});
    if(st.loading) return;
    st.loading = true;
    try{
        var r = await fetch('/api/review/'+jid+'?key='+encodeURIComponent(KEY));
        if(!r.ok) throw new Error('HTTP '+r.status);
        var d = await r.json();
        var newQ = d.questions || [];
        if(isRefresh && st.loaded){
            // Refresco: actualiza SOLO las preguntas/votos del servidor; preserva
            // las ediciones del compañero (letras/marcas/borradas/orden). Re-render
            // únicamente si algo del servidor cambió (evita parpadeo en cada poll).
            var fp = JSON.stringify(newQ);
            if(fp !== st._qfp){
                st._qfp = fp;
                st.questions = newQ;
                var known = {}; st.order.forEach(function(r){ known[r]=true; });
                newQ.forEach(function(q){ if(!known[q.rid]) st.order.push(q.rid); });
                _revApply(jid, false);
            }
            st.loading = false;
            return;
        }
        // Carga inicial: preguntas + estado guardado (si lo había).
        var s = d.state || {};
        st.questions = newQ;
        st.letters = s.answer || {};
        st.marks   = s.state  || {};
        st.deleted = (s.deleted||[]).map(Number);
        st.order   = (s.order||[]).map(Number);
        if(!st.order.length) st.order = newQ.map(function(q){return q.rid;});
        st._qfp = JSON.stringify(newQ);
        st.loading = false;
        if(newQ.length){ st.loaded = true; _revApply(jid, false); }
        // Si aún no hay preguntas (fusión OCR no lista) NO marcamos loaded → se
        // reintenta en el siguiente poll hasta que la fusión esté disponible.
    }catch(e){
        st.loading = false;
        if(!st.loaded){
            var host = document.getElementById('review-'+jid);
            if(host) host.innerHTML = '<div class="rev-wrap" style="color:#888;font-size:11px">'
                + '⚠ Revisión por preguntas no disponible aún ('+_revEsc(e.message)+').</div>';
        }
    }
}

function renderReview(jid){
    var st = window._REVIEW[jid];
    var host = document.getElementById('review-'+jid);
    if(!st || !host) return;
    if(!st.questions || !st.questions.length){ host.innerHTML=''; return; }
    var byId = {}; st.questions.forEach(function(q){ byId[q.rid]=q; });
    var del = {}; st.deleted.forEach(function(r){ del[r]=true; });
    var aliveNum = 0, cards = '', aliveTotal = 0;
    st.order.forEach(function(rid){
        if(rid < 0){ aliveNum++; aliveTotal++; cards += _revGapCard(jid, rid, aliveNum); return; }
        var q = byId[rid]; if(!q) return;
        if(del[rid]){ cards += _revCard(jid, q, null); }
        else { aliveNum++; aliveTotal++; cards += _revCard(jid, q, aliveNum); }
    });
    host.innerHTML =
        '<div class="rev-wrap">'
      +   '<div class="rev-head-row">'
      +     '<span class="rev-title">📝 Revisar preguntas</span>'
      +     '<span class="rev-count">'+aliveTotal+' preguntas</span>'
      +     '<button class="rev-help" onclick="showRevHelp()" title="Cómo funciona / guía del revisor">ℹ️ ¿Cómo funciona?</button>'
      +     '<span class="rev-hint">✓ correcta · ✗ mal leída (X) · 🗑 borrar · ⊕ falta una · ▲▼ ordenar</span>'
      +   '</div>'
      +   '<button class="rev-gap-add" onclick="revInsertGapTop(\'' + jid + '\')" '
      +     'title="El OCR se saltó preguntas al principio: añade un hueco (X) para que las respuestas NO se desplacen">'
      +     '➕ Falta una pregunta al principio (pulsa otra vez si faltan más)</button>'
      +   cards
      + '</div>';
}

function _revChosen(st, q){
    var rid = q.rid;
    if(st.marks[rid]==='bad') return 'X';
    var L = st.letters[rid] || q.ia_answer || 'X';
    L = String(L).toUpperCase();
    return (_REV_LETTERS.indexOf(L)>=0) ? L : 'X';
}

function _revCard(jid, q, num){
    var st = window._REVIEW[jid];
    var rid = q.rid;
    var deleted = (num===null);
    var mark = st.marks[rid];
    var chosen = deleted ? 'X' : _revChosen(st, q);

    if(deleted){
        return '<div class="rev-card del" id="revcard-'+jid+'-'+rid+'">'
          + '<div class="rev-c-head">'
          +   '<span class="rev-num" style="color:#dc3545">🗑</span>'
          +   '<span class="rev-q" style="flex:1;margin:0;color:#888;text-decoration:line-through">'
          +       _revEsc((q.question||'(sin texto)').slice(0,90))+'</span>'
          +   '<button class="rev-restore" onclick="revRestore(\'' + jid + '\',' + rid + ')">↺ Restaurar</button>'
          + '</div></div>';
    }

    var opts='';
    var o = q.options||{};
    ['A','B','C','D'].forEach(function(L){
        if(o[L]!=null && o[L]!=='') opts += '<div class="rev-opt"><b>'+L+')</b> '+_revEsc(o[L])+'</div>';
    });

    var letters='';
    _REV_LETTERS.forEach(function(L){
        var sel = (L===chosen) ? ' sel l-'+L : '';
        letters += '<span class="rev-letter'+sel+'" onclick="revSetLetter(\'' + jid + '\',' + rid + ',\'' + L + '\')">'+L+'</span>';
    });

    var votes='';
    var v = q.votes||{};
    Object.keys(v).forEach(function(prov){
        var lt = v[prov];
        var agree = (lt===chosen);
        var lbl = (typeof PROVIDER_LABELS!=='undefined' && PROVIDER_LABELS[prov]) ? PROVIDER_LABELS[prov] : prov;
        votes += '<span class="rev-vote '+(agree?'agree':'differ')+'">'+_revEsc(lbl)+'·<b>'+_revEsc(lt)+'</b></span>';
    });

    return '<div class="rev-card'+(mark==='ok'?' ok':(mark==='bad'?' bad':''))+'" id="revcard-'+jid+'-'+rid+'">'
      + '<div class="rev-c-head">'
      +   '<span class="rev-move">'
      +     '<button class="rev-mv" onclick="revMove(\'' + jid + '\',' + rid + ',-1)" title="Subir">▲</button>'
      +     '<button class="rev-mv" onclick="revMove(\'' + jid + '\',' + rid + ',1)" title="Bajar">▼</button>'
      +   '</span>'
      +   '<span class="rev-num">'+num+'</span>'
      +   (q.num!=null ? '<span class="rev-orig">orig '+_revEsc(q.num)+'</span>' : '')
      +   (q.section ? '<span class="rev-sec">'+_revEsc(q.section)+'</span>' : '')
      +   '<span class="rev-tools">'
      +     '<button class="rev-tool" onclick="revInsertGapBefore(\'' + jid + '\',' + rid + ')" title="Falta una pregunta ANTES de esta → insertar hueco (X)">⊕</button>'
      +     '<button class="rev-tool'+(mark==='ok'?' act-ok':'')+'" onclick="revMarkOk(\'' + jid + '\',' + rid + ')" title="Marcar correcta">✓</button>'
      +     '<button class="rev-tool'+(mark==='bad'?' act-bad':'')+'" onclick="revMarkBad(\'' + jid + '\',' + rid + ')" title="Mal leída → X">✗</button>'
      +     '<button class="rev-tool" onclick="revDelete(\'' + jid + '\',' + rid + ')" title="Borrar (alucinada)">🗑</button>'
      +   '</span>'
      + '</div>'
      + (q.question ? '<div class="rev-q">'+_revEsc(q.question)+'</div>' : '')
      + (opts ? '<div class="rev-opts">'+opts+'</div>' : '')
      + '<div class="rev-ans-row">'
      +   '<span class="rev-ans-lbl">Resp:</span>'
      +   letters
      +   '<span class="rev-votes">'+votes+'</span>'
      + '</div>'
      + '</div>';
}

function revSetLetter(jid, rid, L){
    var st = window._REVIEW[jid]; if(!st) return;
    st.letters[rid] = L;
    if(st.marks[rid]==='bad') delete st.marks[rid];   // elegir letra rehabilita
    _revApply(jid, true);
}
function revMarkOk(jid, rid){
    var st = window._REVIEW[jid]; if(!st) return;
    if(st.marks[rid]==='ok') delete st.marks[rid]; else st.marks[rid]='ok';
    _revApply(jid, true);
}
function revMarkBad(jid, rid){
    var st = window._REVIEW[jid]; if(!st) return;
    if(st.marks[rid]==='bad') delete st.marks[rid]; else st.marks[rid]='bad';
    _revApply(jid, true);
}
function revDelete(jid, rid){
    var st = window._REVIEW[jid]; if(!st) return;
    if(st.deleted.indexOf(rid)<0) st.deleted.push(rid);
    _revApply(jid, true);
}
function revRestore(jid, rid){
    var st = window._REVIEW[jid]; if(!st) return;
    st.deleted = st.deleted.filter(function(x){ return x!==rid; });
    _revApply(jid, true);
}
function revMove(jid, rid, dir){
    var st = window._REVIEW[jid]; if(!st) return;
    var del = {}; st.deleted.forEach(function(r){ del[r]=true; });
    var i = st.order.indexOf(rid); if(i<0) return;
    var j = i + dir;
    while(j>=0 && j<st.order.length && del[st.order[j]]) j += dir;  // saltar borradas
    if(j<0 || j>=st.order.length) return;
    var t = st.order[i]; st.order[i]=st.order[j]; st.order[j]=t;
    _revApply(jid, true);
}

// ── Huecos: preguntas que el OCR se saltó. Se insertan como rids NEGATIVOS en
// `order` y recomponen como "X" → evitan que las respuestas se desplacen. ──
function _revNextGapId(st){
    var negs = (st.order||[]).filter(function(r){ return r<0; });
    return (negs.length ? Math.min.apply(null, negs) : 0) - 1;
}
function revInsertGapTop(jid){
    var st = window._REVIEW[jid]; if(!st) return;
    st.order.unshift(_revNextGapId(st));
    _revApply(jid, true);
}
function revInsertGapBefore(jid, rid){
    var st = window._REVIEW[jid]; if(!st) return;
    var i = st.order.indexOf(rid); if(i<0) i = 0;
    st.order.splice(i, 0, _revNextGapId(st));
    _revApply(jid, true);
}
function revGapRemove(jid, rid){
    var st = window._REVIEW[jid]; if(!st) return;
    st.order = st.order.filter(function(x){ return x!==rid; });
    _revApply(jid, true);
}
function _revGapCard(jid, rid, num){
    return '<div class="rev-card rev-gap" id="revcard-'+jid+'-'+rid+'">'
      + '<div class="rev-c-head">'
      +   '<span class="rev-move">'
      +     '<button class="rev-mv" onclick="revMove(\'' + jid + '\',' + rid + ',-1)" title="Subir">▲</button>'
      +     '<button class="rev-mv" onclick="revMove(\'' + jid + '\',' + rid + ',1)" title="Bajar">▼</button>'
      +   '</span>'
      +   '<span class="rev-num">'+num+'</span>'
      +   '<span class="rev-gap-lbl">⬚ Hueco — pregunta que el OCR no leyó (se envía X para no desplazar el resto)</span>'
      +   '<span class="rev-tools">'
      +     '<button class="rev-tool" onclick="revGapRemove(\'' + jid + '\',' + rid + ')" title="Quitar hueco">🗑</button>'
      +   '</span>'
      + '</div></div>';
}

// Recompone la cadena final en cliente (espejo de _recompose_review_answer).
function _revRecompose(jid){
    var st = window._REVIEW[jid]; if(!st) return '';
    var byId={}; st.questions.forEach(function(q){ byId[q.rid]=q; });
    var del={}; st.deleted.forEach(function(r){ del[r]=true; });
    var out='';
    st.order.forEach(function(rid){
        if(del[rid]) return;
        if(rid < 0){ out += 'X'; return; }   // hueco → X (realinear)
        var q=byId[rid]; if(!q) return;
        out += _revChosen(st, q);
    });
    return out;
}

// Re-pinta cards (renumera) + vuelca la cadena en teclas/input/preview + autosave.
function _revApply(jid, save){
    renderReview(jid);
    var ans = _revRecompose(jid);
    var keysEl = document.getElementById('keys-'+jid);
    if(keysEl) keysEl.innerHTML = buildKeys(jid, ans, Math.max(1, ans.length));
    var t = document.getElementById('txt-'+jid);
    if(t) t.value = ans;
    _updPreview(jid);
    if(save) _revSaveDebounced(jid);
}

function _revSaveDebounced(jid){
    var st = window._REVIEW[jid]; if(!st) return;
    if(st._saveT) clearTimeout(st._saveT);
    st._saveT = setTimeout(function(){ _revSave(jid); }, 350);
}
async function _revSave(jid){
    var st = window._REVIEW[jid]; if(!st) return;
    try{
        var r = await fetch('/api/review/'+jid, {
            method:'PATCH',
            headers:{'Content-Type':'application/json','X-Api-Key':KEY},
            body: JSON.stringify({answer: st.letters, state: st.marks,
                                  deleted: st.deleted, order: st.order})
        });
        var d = await r.json().catch(function(){ return {}; });
        if(d && d.closed) toast('⏱ Ventana cerrada — ya se auto-envió al móvil');
    }catch(e){ /* autosave best-effort; ENVIAR manual reintenta con la misma cadena */ }
}

// ── Modal de instrucciones para el revisor (botón "ℹ️ ¿Cómo funciona?") ──
function showRevHelp(){
    var id='revhelp-modal';
    var ex=document.getElementById(id);
    if(ex){ ex.style.display='flex'; return; }
    var m=document.createElement('div');
    m.id=id;
    m.style.cssText='position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:10000;display:flex;align-items:center;justify-content:center;padding:18px';
    m.onclick=function(e){ if(e.target===m) m.style.display='none'; };
    m.innerHTML=
        '<div style="background:#12121c;border:1px solid #2a2a3a;border-radius:12px;max-width:680px;max-height:88vh;overflow:auto;padding:22px 24px;color:#ddd;font-size:14px;line-height:1.5">'
      +   '<h2 style="margin:0 0 4px;color:#9b85ff;font-size:19px">📝 Cómo revisar — guía rápida</h2>'
      +   '<p style="color:#9ad;margin:0 0 14px">Tu trabajo: comprobar que la IA <b>no se ha inventado preguntas</b> y que las ha <b>leído bien</b>, comparando la <b>foto de la izquierda</b> con cada pregunta de la derecha. Pasa el ratón por la foto para ampliarla.</p>'
      +   '<div style="background:#0d1620;border-left:3px solid #9b85ff;border-radius:6px;padding:10px 12px;margin-bottom:14px">'
      +     '<b>Si no tocas nada</b>, se envía al móvil la respuesta por <b>consenso de todas las IAs</b>. Tú solo corriges lo que veas mal.'
      +   '</div>'
      +   '<h3 style="color:#fff;font-size:15px;margin:14px 0 6px">Botones de cada pregunta</h3>'
      +   '<ul style="margin:0 0 8px;padding-left:18px">'
      +     '<li><b style="color:#3ddc84">✓</b> — pregunta y respuesta correctas.</li>'
      +     '<li><b style="color:#e0c060">✗</b> — <b>mal leída</b>: se envía <b>X</b> en su lugar.</li>'
      +     '<li><b style="color:#ff6b6b">🗑</b> — la IA se la <b>inventó</b> o está duplicada: bórrala.</li>'
      +     '<li><b>A B C D X</b> — cambia la letra si la pregunta es buena pero la opción está mal.</li>'
      +     '<li><b>▲ ▼</b> — reordena si el OCR las puso en orden equivocado.</li>'
      +   '</ul>'
      +   '<h3 style="color:#fff;font-size:15px;margin:16px 0 6px">⚠️ Lo más importante: que NO se desplacen</h3>'
      +   '<p style="margin:0 0 8px">Las respuestas se mandan <b>por posición</b>: la 1ª letra es para la 1ª pregunta del examen, la 2ª para la 2ª… Si el OCR <b>se saltó preguntas</b> (p.ej. lo que aquí sale como "pregunta 1" es en realidad la <b>3ª</b> del examen porque arriba había 2 sin leer), <b>TODO se desplaza</b> y el alumno lo pondría una casilla antes.</p>'
      +   '<p style="margin:0 0 6px">Para arreglarlo, añade <b>huecos</b> (cuentan como X):</p>'
      +   '<ul style="margin:0 0 8px;padding-left:18px">'
      +     '<li><b style="color:#9ad">➕ Falta una pregunta al principio</b> — hueco arriba (púlsalo 2 veces si faltan 2).</li>'
      +     '<li><b style="color:#9ad">⊕</b> (en una pregunta) — inserta un hueco <b>justo antes</b>, si el OCR se saltó una en medio.</li>'
      +     '<li><b>🗑</b> en un hueco — lo quita si te pasaste.</li>'
      +   '</ul>'
      +   '<p style="color:#888;margin:0 0 16px;font-size:13px">Ejemplo: el examen tiene preguntas 1-2-3-4-5 pero el OCR solo leyó 3-4-5 → pon <b>2 huecos arriba</b>. La cadena pasa de <code style="color:#ccc">BCA</code> a <code style="color:#ccc">XXBCA</code> y todo cuadra.</p>'
      +   '<div style="text-align:right"><button onclick="document.getElementById(\'' + id + '\').style.display=\'none\'" style="background:#198754;color:#fff;border:none;border-radius:7px;padding:9px 18px;font-size:14px;font-weight:700;cursor:pointer">Entendido</button></div>'
      + '</div>';
    document.body.appendChild(m);
}

// ════════════════════════════════════════════════════════════════════════════
// ZOOM RÁPIDO (hover) sobre fotos y vídeo de la columna izquierda — el revisor
// lee el folio y lo compara con las cards sin clicar nada.
//   · Imágenes → lupa flotante (panel fixed con la zona ampliada bajo el cursor);
//     no la corta el overflow del media-col sticky.
//   · Vídeo → scale in-place dentro de su marco, siguiendo el cursor.
// ════════════════════════════════════════════════════════════════════════════
(function(){
    var lens = null;
    var ZOOM = 2.8;   // factor de ampliación de la lupa de imágenes
    function _lens(){
        if(lens) return lens;
        lens = document.createElement('div');
        lens.id = '_zoomlens';
        lens.style.cssText =
            'position:fixed;pointer-events:none;display:none;z-index:9998;'
          + 'width:min(46vw,560px);height:min(72vh,680px);'
          + 'border:2px solid #9b85ff;border-radius:10px;'
          + 'box-shadow:0 10px 44px rgba(0,0,0,.75);background:#000 no-repeat;';
        document.body.appendChild(lens);
        return lens;
    }
    function _hide(){ if(lens) lens.style.display='none'; }
    document.addEventListener('mousemove', function(e){
        var t = e.target;
        // Vídeo: scale in-place (origen = posición del cursor).
        if(t && t.tagName==='VIDEO' && t.closest('.media-col')){
            var rv = t.getBoundingClientRect();
            t.style.transformOrigin =
                ((e.clientX-rv.left)/rv.width*100)+'% '+((e.clientY-rv.top)/rv.height*100)+'%';
            t.classList.add('zoomed');
            _hide();
            return;
        }
        // Imagen: lupa flotante con la zona ampliada bajo el cursor.
        if(t && t.tagName==='IMG' && t.closest('.media-col')){
            var src = t.currentSrc || t.src;
            if(!src){ _hide(); return; }
            var r = t.getBoundingClientRect();
            if(r.width < 4 || r.height < 4){ _hide(); return; }
            var L = _lens();
            var bw = r.width*ZOOM, bh = r.height*ZOOM;
            var px = Math.min(1, Math.max(0, (e.clientX-r.left)/r.width));
            var py = Math.min(1, Math.max(0, (e.clientY-r.top)/r.height));
            var lw = L.offsetWidth || 480, lh = L.offsetHeight || 600;
            L.style.backgroundImage    = 'url("'+src+'")';
            L.style.backgroundSize     = bw+'px '+bh+'px';
            L.style.backgroundPosition = (-(px*bw - lw/2))+'px '+(-(py*bh - lh/2))+'px';
            var lx = e.clientX + 26, ly = e.clientY + 26;
            if(lx + lw > window.innerWidth)  lx = e.clientX - lw - 26;
            if(lx < 8) lx = 8;
            if(ly + lh > window.innerHeight) ly = window.innerHeight - lh - 8;
            if(ly < 8) ly = 8;
            L.style.left = lx+'px'; L.style.top = ly+'px';
            L.style.display = 'block';
            return;
        }
        _hide();
    }, true);
    // Quitar el zoom del vídeo al salir de él.
    document.addEventListener('mouseout', function(e){
        if(e.target && e.target.tagName==='VIDEO') e.target.classList.remove('zoomed');
    }, true);
    // Ocultar la lupa al scrollear o salir de la ventana (su posición cambia).
    window.addEventListener('scroll', _hide, true);
    document.addEventListener('mouseleave', _hide);
})();

// ── Config: cargar valores actuales al abrir ────────────────────────────────
// ── Pie chart de pesos en la fusión ─────────────────────────────────────────
// Dibuja un donut SVG donde cada porción es proporcional al peso de cada IA en
// la votación. Se llama al cargar la config (loadConfig) y al mover cualquier
// slider (oninput). Si todos los pesos son 0 muestra placeholder.
const _WEIGHT_ITEMS = [
    { id:'cfg_w_ant', name:'Anthropic', color:'#3ddc84', emoji:'🟢' },
    { id:'cfg_w_oai', name:'OpenAI',    color:'#4d9eff', emoji:'🔵' },
    { id:'cfg_w_gem', name:'Gemini',    color:'#ffb066', emoji:'🟠' },
    { id:'cfg_w_dsk', name:'DeepSeek',  color:'#a855f7', emoji:'🟣' },
    { id:'cfg_w_mst', name:'Mistral',   color:'#fb7185', emoji:'🌶️' },
    { id:'cfg_w_nv',  name:'NVIDIA',    color:'#76b900', emoji:'🟢' },
    { id:'cfg_w_mim', name:'MiMo',      color:'#ff6b00', emoji:'🤖' },
];
function renderWeightsPie() {
    const svg = document.getElementById('weights-pie-svg');
    const legend = document.getElementById('weights-pie-legend');
    if(!svg) return;
    const items = _WEIGHT_ITEMS.map(it => ({
        ...it,
        w: parseInt(document.getElementById(it.id)?.value) || 0
    }));
    const total = items.reduce((s, it) => s + it.w, 0);
    const cx = 70, cy = 70, rOuter = 60, rInner = 28;

    if(total === 0) {
        svg.innerHTML =
            `<circle cx="${cx}" cy="${cy}" r="${rOuter}" fill="#1a1a1a" stroke="#333"/>
             <text x="${cx}" y="${cy-4}" text-anchor="middle" font-size="10" fill="#666">sin pesos</text>
             <text x="${cx}" y="${cy+9}" text-anchor="middle" font-size="9" fill="#555">(todos a 0)</text>`;
        if(legend) legend.innerHTML = '<span style="color:#666">Sube algún slider para activar la votación.</span>';
        return;
    }

    // Construimos las porciones. Caso especial: una sola IA con peso → círculo completo.
    const active = items.filter(it => it.w > 0);
    let svgInner = '';
    if(active.length === 1) {
        svgInner += `<circle cx="${cx}" cy="${cy}" r="${rOuter}" fill="${active[0].color}"/>`;
    } else {
        let cumAngle = -90; // empieza en las 12 en punto
        for(const it of active) {
            const angle = (it.w / total) * 360;
            const a0 = cumAngle * Math.PI / 180;
            const a1 = (cumAngle + angle) * Math.PI / 180;
            const x0 = cx + rOuter * Math.cos(a0);
            const y0 = cy + rOuter * Math.sin(a0);
            const x1 = cx + rOuter * Math.cos(a1);
            const y1 = cy + rOuter * Math.sin(a1);
            const large = angle > 180 ? 1 : 0;
            svgInner += `<path d="M ${cx} ${cy} L ${x0} ${y0} `
                      + `A ${rOuter} ${rOuter} 0 ${large} 1 ${x1} ${y1} Z" `
                      + `fill="${it.color}" stroke="#0a0a0a" stroke-width="1"/>`;
            cumAngle += angle;
        }
    }
    // Agujero central (efecto donut) + total
    svgInner += `<circle cx="${cx}" cy="${cy}" r="${rInner}" fill="#0f0f0f"/>`;
    svgInner += `<text x="${cx}" y="${cy-3}" text-anchor="middle" font-size="8" fill="#888" letter-spacing="0.5">PESO TOTAL</text>`;
    svgInner += `<text x="${cx}" y="${cy+13}" text-anchor="middle" font-size="16" fill="#fff" font-weight="700">${total}</text>`;
    svg.innerHTML = svgInner;

    // Leyenda con porcentajes (sólo las IAs con peso > 0)
    if(legend) {
        const rows = active.map(it => {
            const pct = (it.w / total * 100).toFixed(1);
            return `<span style="display:inline-flex;align-items:center;gap:3px;margin:1px 4px;white-space:nowrap">
                        <span style="width:9px;height:9px;background:${it.color};border-radius:2px;display:inline-block"></span>
                        <span style="color:${it.color};font-weight:600">${it.name}</span>
                        <span style="color:#888">${pct}%</span>
                    </span>`;
        }).join('');
        legend.innerHTML = rows;
    }
}

async function loadConfig() {
    const status = document.getElementById('cfg-status');
    if(status) { status.textContent = '⏳ Cargando config…'; status.style.color = '#ffb066'; }
    console.log('[panel] loadConfig() fetching /api/config…');
    try {
        const r = await fetch('/api/config?key=' + encodeURIComponent(KEY));
        console.log('[panel] /api/config status=' + r.status);
        if(!r.ok) {
            const msg = '❌ No se pudo cargar config (HTTP ' + r.status + ')';
            if(status) { status.textContent = msg; status.style.color = '#dc3545'; }
            console.error('[panel] ' + msg);
            return;
        }
        let c = {};
        try { c = await r.json(); } catch(parseErr) {
            const msg = '❌ Respuesta inválida del relay';
            if(status) { status.textContent = msg; status.style.color = '#dc3545'; }
            console.error('[panel] JSON parse error:', parseErr);
            return;
        }
        console.log('[panel] config keys recibidas:', Object.keys(c).length);
        const set = (id, v) => { const el = document.getElementById(id);
                                  if(el) el.value = v || ''; };
        // Analyzers (text/image)
        set('cfg_ant',             c.anthropic_key);
        set('cfg_ant_bk',          c.anthropic_key_backup);
        set('cfg_ant_model',       c.anthropic_model);
        // Anthropic también participa en OCR-video (mismo key, modelo separado)
        set('cfg_ant_video_model', c.claude_video_ocr_model);
        set('cfg_oai',             c.openai_key);
        set('cfg_oai_bk',          c.openai_key_backup);
        set('cfg_oai_model',       c.openai_model);
        // OpenAI también participa en OCR-video (mismo key, modelo separado)
        set('cfg_oai_video_model', c.openai_video_ocr_model);
        // OpenAI tuning agéntico (Responses API): effort + max_tool_calls + allowed_domains
        const effortEl = document.getElementById('cfg_oai_effort');
        if(effortEl) effortEl.value = (c.openai_reasoning_effort || 'medium');
        const mtcEl = document.getElementById('cfg_oai_max_tools');
        if(mtcEl) mtcEl.value = String(c.openai_max_tool_calls || 4);
        const domsEl = document.getElementById('cfg_oai_domains');
        if(domsEl) {
            const arr = Array.isArray(c.openai_allowed_domains) ? c.openai_allowed_domains : [];
            domsEl.value = arr.join('\n');
        }
        set('cfg_gem',             c.gemini_key);
        set('cfg_gem_bk',          c.gemini_key_backup);
        set('cfg_gem_model',       c.gemini_model);
        set('cfg_gem_video_model', c.gemini_video_model);
        set('cfg_dsk',             c.deepseek_key);
        set('cfg_dsk_bk',          c.deepseek_key_backup);
        set('cfg_dsk_model',       c.deepseek_model);
        set('cfg_mst',             c.mistral_key);
        set('cfg_mst_bk',          c.mistral_key_backup);
        set('cfg_mst_model',       c.mistral_model);
        // OCRs de video
        set('cfg_qwn',             c.qwen_key);
        set('cfg_qwn_bk',          c.qwen_key_backup);
        set('cfg_qwn_model',       c.qwen_video_model);
        set('cfg_kmi',             c.kimi_key);
        set('cfg_kmi_bk',          c.kimi_key_backup);
        set('cfg_kmi_model',       c.kimi_video_model);
        // MiMo: analyzer + OCR video (mismo proveedor)
        set('cfg_mim',             c.mimo_key);
        set('cfg_mim_bk',          c.mimo_key_backup);
        set('cfg_mim_model',       c.mimo_model);
        set('cfg_mim_video_model', c.mimo_video_model);
        // NVIDIA Nemotron Super 49B (analyzer texto-only con su propia key)
        set('cfg_nv',              c.nvidia_key);
        set('cfg_nv_bk',           c.nvidia_key_backup);
        set('cfg_nv_model',        c.nvidia_model);
        // Z.AI / Zhipu (GLM-OCR — OCR-image puro)
        set('cfg_zai',             c.z_ai_key);
        set('cfg_zai_bk',          c.z_ai_key_backup);
        // Tavily (paso intermedio OCR → razonamiento)
        set('cfg_tav',             c.tavily_key);
        set('cfg_tav_bk',          c.tavily_key_backup);
        set('cfg_tav_depth',       c.tavily_search_depth || 'basic');
        if(c.tavily_max_results      != null) set('cfg_tav_max',      String(c.tavily_max_results));
        if(c.tavily_http_timeout_s   != null) set('cfg_tav_http_to',  String(c.tavily_http_timeout_s));
        if(c.tavily_total_deadline_s != null) set('cfg_tav_deadline', String(c.tavily_total_deadline_s));
        const tavEn = document.getElementById('cfg_tav_enabled');
        if(tavEn) tavEn.checked = (c.tavily_enabled !== false);
        // Meta-Judge (árbitro post-fusión, opción B + fallback)
        set('cfg_mj_model',  c.meta_judge_model  || 'gpt-5');
        set('cfg_mj_effort', c.meta_judge_reasoning_effort || 'high');
        if(c.meta_judge_timeout_s != null) set('cfg_mj_timeout', String(c.meta_judge_timeout_s));
        const mjEn = document.getElementById('cfg_mj_enabled');
        if(mjEn) mjEn.checked = (c.meta_judge_enabled !== false);
        // Topaz Labs (mejora de imagen post-stacking)
        set('cfg_tpz',        c.topaz_key);
        set('cfg_tpz_model',  c.topaz_model || 'Wonder 3');
        if(c.topaz_output_height != null) set('cfg_tpz_height',  String(c.topaz_output_height));
        if(c.topaz_timeout_s     != null) set('cfg_tpz_timeout', String(c.topaz_timeout_s));
        const tpzEn = document.getElementById('cfg_tpz_enabled');
        if(tpzEn) tpzEn.checked = (c.topaz_enabled !== false);
        // Pesos (range 0..10). El label de valor también se sincroniza.
        const setW = (id, v) => {
            const el = document.getElementById(id);
            if(!el) return;
            const n = (v == null) ? 1 : Math.max(0, Math.min(10, parseInt(v) || 0));
            el.value = String(n);
            const vlabel = document.getElementById(id + '_v');
            if(vlabel) vlabel.textContent = String(n);
        };
        setW('cfg_w_ant', c.anthropic_weight);
        setW('cfg_w_oai', c.openai_weight);
        setW('cfg_w_gem', c.gemini_weight);
        setW('cfg_w_dsk', c.deepseek_weight);
        setW('cfg_w_mst', c.mistral_weight);
        setW('cfg_w_nv',  c.nvidia_weight);
        setW('cfg_w_mim', c.mimo_weight);
        renderWeightsPie();
        // Pesos OCR de video (fase 1) — independientes
        setW('cfg_wocr_mst', c.mistral_ocr_weight);
        setW('cfg_wocr_gem', c.gemini_ocr_weight);
        setW('cfg_wocr_qwn', c.qwen_ocr_weight);
        setW('cfg_wocr_glm', c.glm_ocr_weight);
        setW('cfg_wocr_ant', c.anthropic_ocr_weight);
        setW('cfg_wocr_oai', c.openai_ocr_weight);
        setW('cfg_wocr_dsk', c.deepseek_ocr_weight);
        setW('cfg_wocr_kmi', c.kimi_ocr_weight);
        setW('cfg_wocr_mim', c.mimo_ocr_weight);
        renderOcrWeightsPie();
        if(status) {
            status.textContent = '✓ Config cargada (' + Object.keys(c).length + ' campos)';
            status.style.color = '#198754';
        }
        console.log('[panel] loadConfig() OK');
    } catch(e) {
        const msg = '❌ ' + (e.message || e);
        if(status) { status.textContent = msg; status.style.color = '#dc3545'; }
        console.error('[panel] loadConfig falló:', e);
    }
}

// ── Auto-guardado de pesos al mover slider ──────────────────────────────────
// Cuando el usuario suelta cualquier slider de peso, esto se dispara automática-
// mente. NO envía API keys ni modelos (esos siguen requiriendo "Guardar todo")
// para evitar pisar inputs que el usuario aún esté editando.
let _autoSaveTimer = null;
function autoSaveWeights() {
    const status = document.getElementById('weights-status');
    if(status) { status.textContent = '⏳ guardando…'; status.style.color = '#ffb066'; }
    clearTimeout(_autoSaveTimer);
    // Debounce 300ms: si el usuario mueve varios sliders rápido, agrupa en 1 POST.
    _autoSaveTimer = setTimeout(async () => {
        const v = id => (document.getElementById(id)?.value || '').trim();
        const body = {
            anthropic_weight: parseInt(v('cfg_w_ant')) || 0,
            openai_weight:    parseInt(v('cfg_w_oai')) || 0,
            gemini_weight:    parseInt(v('cfg_w_gem')) || 0,
            deepseek_weight:  parseInt(v('cfg_w_dsk')) || 0,
            mistral_weight:   parseInt(v('cfg_w_mst')) || 0,
            nvidia_weight:    parseInt(v('cfg_w_nv'))  || 0,
            mimo_weight:      parseInt(v('cfg_w_mim')) || 0,
        };
        try {
            const r = await fetch('/api/config', {
                method:  'POST',
                headers: {'Content-Type':'application/json','X-Api-Key':KEY},
                body:    JSON.stringify(body),
            });
            let data = {};
            try { data = await r.json(); } catch(_) { data = {}; }
            if(!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
            if(status) {
                if(data.persisted === false) {
                    status.textContent = '⚠ aplicado en RAM (DB caída)';
                    status.style.color = '#ffc107';
                } else {
                    status.textContent = '✓ guardado · ' + new Date().toLocaleTimeString('es');
                    status.style.color = '#198754';
                    // Borra el mensaje a los 3s para no acumular ruido visual.
                    setTimeout(() => {
                        if(status.textContent.startsWith('✓')) status.textContent = '';
                    }, 3000);
                }
            }
            toast('⚖️ Pesos actualizados');
        } catch(e) {
            if(status) { status.textContent = '❌ ' + e.message; status.style.color = '#dc3545'; }
        }
    }, 300);
}

// ── Pie chart de pesos OCR (fase 1, independiente del de analyzers) ────────
const _OCR_WEIGHT_ITEMS = [
    { id:'cfg_wocr_mst', name:'Mistral OCR',  color:'#fb7185', emoji:'📄' },
    { id:'cfg_wocr_gem', name:'Gemini OCR',   color:'#ffb066', emoji:'🎬' },
    { id:'cfg_wocr_qwn', name:'Qwen OCR',     color:'#22d3ee', emoji:'🎬' },
    { id:'cfg_wocr_glm', name:'GLM-OCR',      color:'#a78bfa', emoji:'📄' },
    { id:'cfg_wocr_ant', name:'Claude OCR',   color:'#3ddc84', emoji:'🎬' },
    { id:'cfg_wocr_oai', name:'GPT-4o OCR',   color:'#4d9eff', emoji:'🎬' },
    { id:'cfg_wocr_dsk', name:'DeepSeek OCR', color:'#a855f7', emoji:'📄' },
    { id:'cfg_wocr_kmi', name:'Kimi OCR',     color:'#facc15', emoji:'🎬' },
    { id:'cfg_wocr_mim', name:'MiMo OCR',     color:'#ff6b00', emoji:'🎬' },
];
function renderOcrWeightsPie() {
    const svg = document.getElementById('weights-ocr-pie-svg');
    const legend = document.getElementById('weights-ocr-pie-legend');
    if(!svg) return;
    const items = _OCR_WEIGHT_ITEMS.map(it => ({
        ...it,
        w: parseInt(document.getElementById(it.id)?.value) || 0
    }));
    const total = items.reduce((s, it) => s + it.w, 0);
    const cx = 70, cy = 70, rOuter = 60, rInner = 28;
    if(total === 0) {
        svg.innerHTML =
            `<circle cx="${cx}" cy="${cy}" r="${rOuter}" fill="#1a1a1a" stroke="#333"/>
             <text x="${cx}" y="${cy-4}" text-anchor="middle" font-size="10" fill="#666">sin pesos</text>
             <text x="${cx}" y="${cy+9}" text-anchor="middle" font-size="9" fill="#555">(todos a 0)</text>`;
        if(legend) legend.innerHTML = '<span style="color:#666">Sube algún slider para activar la votación OCR.</span>';
        return;
    }
    const active = items.filter(it => it.w > 0);
    let svgInner = '';
    if(active.length === 1) {
        svgInner += `<circle cx="${cx}" cy="${cy}" r="${rOuter}" fill="${active[0].color}"/>`;
    } else {
        let cumAngle = -90;
        for(const it of active) {
            const angle = (it.w / total) * 360;
            const a0 = cumAngle * Math.PI / 180;
            const a1 = (cumAngle + angle) * Math.PI / 180;
            const x0 = cx + rOuter * Math.cos(a0);
            const y0 = cy + rOuter * Math.sin(a0);
            const x1 = cx + rOuter * Math.cos(a1);
            const y1 = cy + rOuter * Math.sin(a1);
            const large = angle > 180 ? 1 : 0;
            svgInner += `<path d="M ${cx} ${cy} L ${x0} ${y0} `
                      + `A ${rOuter} ${rOuter} 0 ${large} 1 ${x1} ${y1} Z" `
                      + `fill="${it.color}" stroke="#0a0a0a" stroke-width="1"/>`;
            cumAngle += angle;
        }
    }
    svgInner += `<circle cx="${cx}" cy="${cy}" r="${rInner}" fill="#0f0f0f"/>`;
    svgInner += `<text x="${cx}" y="${cy-3}" text-anchor="middle" font-size="8" fill="#888" letter-spacing="0.5">OCR TOTAL</text>`;
    svgInner += `<text x="${cx}" y="${cy+13}" text-anchor="middle" font-size="16" fill="#fff" font-weight="700">${total}</text>`;
    svg.innerHTML = svgInner;
    if(legend) {
        const rows = active.map(it => {
            const pct = (it.w / total * 100).toFixed(1);
            return `<span style="display:inline-flex;align-items:center;gap:3px;margin:1px 4px;white-space:nowrap">
                        <span style="width:9px;height:9px;background:${it.color};border-radius:2px;display:inline-block"></span>
                        <span style="color:${it.color};font-weight:600">${it.name}</span>
                        <span style="color:#888">${pct}%</span>
                    </span>`;
        }).join('');
        legend.innerHTML = rows;
    }
}

// ── Auto-guardado de pesos OCR (debounced, independiente del de analyzers) ──
let _autoSaveOcrTimer = null;
function autoSaveOcrWeights() {
    const status = document.getElementById('weights-ocr-status');
    if(status) { status.textContent = '⏳ guardando…'; status.style.color = '#ffb066'; }
    clearTimeout(_autoSaveOcrTimer);
    _autoSaveOcrTimer = setTimeout(async () => {
        const v = id => (document.getElementById(id)?.value || '').trim();
        const body = {
            mistral_ocr_weight:   parseInt(v('cfg_wocr_mst')) || 0,
            gemini_ocr_weight:    parseInt(v('cfg_wocr_gem')) || 0,
            qwen_ocr_weight:      parseInt(v('cfg_wocr_qwn')) || 0,
            glm_ocr_weight:       parseInt(v('cfg_wocr_glm')) || 0,
            anthropic_ocr_weight: parseInt(v('cfg_wocr_ant')) || 0,
            openai_ocr_weight:    parseInt(v('cfg_wocr_oai')) || 0,
            deepseek_ocr_weight:  parseInt(v('cfg_wocr_dsk')) || 0,
            kimi_ocr_weight:      parseInt(v('cfg_wocr_kmi')) || 0,
            mimo_ocr_weight:      parseInt(v('cfg_wocr_mim')) || 0,
        };
        try {
            const r = await fetch('/api/config', {
                method:  'POST',
                headers: {'Content-Type':'application/json','X-Api-Key':KEY},
                body:    JSON.stringify(body),
            });
            let data = {};
            try { data = await r.json(); } catch(_) { data = {}; }
            if(!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
            if(status) {
                if(data.persisted === false) {
                    status.textContent = '⚠ aplicado en RAM (DB caída)';
                    status.style.color = '#ffc107';
                } else {
                    status.textContent = '✓ guardado · ' + new Date().toLocaleTimeString('es');
                    status.style.color = '#198754';
                    setTimeout(() => {
                        if(status.textContent.startsWith('✓')) status.textContent = '';
                    }, 3000);
                }
            }
            toast('🎬 Pesos OCR actualizados');
        } catch(e) {
            if(status) { status.textContent = '❌ ' + e.message; status.style.color = '#dc3545'; }
        }
    }, 300);
}

// ── Config: guardar TODO (envía siempre todos los campos, "" = limpiar) ─────
async function saveKeys() {
    const status = document.getElementById('cfg-status');
    const v = id => (document.getElementById(id)?.value || '').trim();
    const body = {
        // Analyzers
        anthropic_key:        v('cfg_ant'),
        anthropic_key_backup: v('cfg_ant_bk'),
        openai_key:           v('cfg_oai'),
        openai_key_backup:    v('cfg_oai_bk'),
        gemini_key:           v('cfg_gem'),
        gemini_key_backup:    v('cfg_gem_bk'),
        deepseek_key:         v('cfg_dsk'),
        deepseek_key_backup:  v('cfg_dsk_bk'),
        mistral_key:          v('cfg_mst'),
        mistral_key_backup:   v('cfg_mst_bk'),
        // OCRs video
        qwen_key:             v('cfg_qwn'),
        qwen_key_backup:      v('cfg_qwn_bk'),
        kimi_key:             v('cfg_kmi'),
        kimi_key_backup:      v('cfg_kmi_bk'),
        // MiMo (analyzer + OCR video, full-modal Xiaomi)
        mimo_key:             v('cfg_mim'),
        mimo_key_backup:      v('cfg_mim_bk'),
        // NVIDIA Nemotron OCR (key propia)
        nvidia_key:           v('cfg_nv'),
        nvidia_key_backup:    v('cfg_nv_bk'),
        // Z.AI / Zhipu GLM-OCR (key propia)
        z_ai_key:             v('cfg_zai'),
        z_ai_key_backup:      v('cfg_zai_bk'),
        // Tavily (paso intermedio OCR → razonamiento)
        tavily_key:           v('cfg_tav'),
        tavily_key_backup:    v('cfg_tav_bk'),
        tavily_enabled:       !!document.getElementById('cfg_tav_enabled')?.checked,
        tavily_search_depth:  v('cfg_tav_depth') || 'basic',
        tavily_max_results:   parseInt(v('cfg_tav_max'))      || 3,
        tavily_http_timeout_s:   parseFloat(v('cfg_tav_http_to'))  || 6.0,
        tavily_total_deadline_s: parseFloat(v('cfg_tav_deadline')) || 8.0,
        // Meta-Judge (árbitro post-fusión, opción B + fallback)
        meta_judge_enabled:          !!document.getElementById('cfg_mj_enabled')?.checked,
        meta_judge_model:            v('cfg_mj_model')  || 'gpt-5',
        meta_judge_reasoning_effort: v('cfg_mj_effort') || 'high',
        meta_judge_timeout_s:        parseFloat(v('cfg_mj_timeout')) || 60.0,
        // Topaz Labs (mejora de imagen post-stacking, opcional)
        topaz_key:           v('cfg_tpz'),
        topaz_enabled:       !!document.getElementById('cfg_tpz_enabled')?.checked,
        topaz_model:         v('cfg_tpz_model') || 'Wonder 3',
        topaz_output_height: parseInt(v('cfg_tpz_height')) || 0,
        topaz_timeout_s:     parseFloat(v('cfg_tpz_timeout')) || 60.0,
        // Modelos
        anthropic_model:      v('cfg_ant_model'),
        openai_model:         v('cfg_oai_model'),
        gemini_model:         v('cfg_gem_model'),
        gemini_video_model:   v('cfg_gem_video_model'),
        deepseek_model:       v('cfg_dsk_model'),
        mistral_model:        v('cfg_mst_model'),
        qwen_video_model:     v('cfg_qwn_model'),
        kimi_video_model:     v('cfg_kmi_model'),
        mimo_model:           v('cfg_mim_model'),
        mimo_video_model:     v('cfg_mim_video_model'),
        // Claude/OpenAI OCR-video: misma key que analyzer, modelo separado
        claude_video_ocr_model: v('cfg_ant_video_model'),
        openai_video_ocr_model: v('cfg_oai_video_model'),
        // NVIDIA Nemotron como analyzer texto (no OCR)
        nvidia_model:         v('cfg_nv_model'),
        // OpenAI tuning agéntico (Responses API · web_search GA)
        openai_reasoning_effort: v('cfg_oai_effort'),
        openai_max_tool_calls:   parseInt(v('cfg_oai_max_tools')) || 4,
        openai_allowed_domains:  v('cfg_oai_domains')
                                    .split(/[\n,]/)
                                    .map(s => s.trim())
                                    .filter(Boolean)
                                    .slice(0, 100),
        // Pesos en la votación de fusionar (0..10, int)
        anthropic_weight:     parseInt(v('cfg_w_ant')) || 0,
        openai_weight:        parseInt(v('cfg_w_oai')) || 0,
        gemini_weight:        parseInt(v('cfg_w_gem')) || 0,
        deepseek_weight:      parseInt(v('cfg_w_dsk')) || 0,
        mistral_weight:       parseInt(v('cfg_w_mst')) || 0,
        nvidia_weight:        parseInt(v('cfg_w_nv'))  || 0,
        mimo_weight:          parseInt(v('cfg_w_mim')) || 0,
    };
    try {
        const r = await fetch('/api/config', {
            method:  'POST',
            headers: {'Content-Type':'application/json','X-Api-Key':KEY},
            body:    JSON.stringify(body)
        });
        // Parsing defensivo: si el server devuelve un 500 con body no-JSON
        // (proxy error, etc.), JSON.parse falla con "Unexpected token". Caja-fuerte.
        let data = {};
        try { data = await r.json(); } catch(_) { data = {}; }
        if(!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
        const n = (data.changes || []).length;
        if(data.persisted === false) {
            // Caso especial: la config se aplicó en RAM pero ninguna DB la aceptó.
            // Server responde 503 → ya no entramos aquí (lanza throw arriba).
            // Pero por defensiva, también lo cubrimos por si el flujo cambia.
            if(status) status.textContent = '⚠️ Aplicado en RAM pero NO persistido (' + n + ' cambios)';
            toast('⚠️ Cambios aplicados temporalmente — DBs no disponibles');
        } else {
            if(status) status.textContent = '✅ Guardado: ' + n + ' cambios';
            toast('✅ Config actualizada');
        }
    } catch(e) {
        if(status) status.textContent = '❌ ' + e.message;
        toast('❌ ' + e.message);
    }
}

// ── Modal de errores ─────────────────────────────────────────────────────────
async function showErrors() {
    const body = document.getElementById('errors-body');
    const modal = document.getElementById('errors-modal');
    modal.style.display = '';
    body.textContent = 'Cargando...';
    try {
        const r = await fetch('/api/errors?key=' + encodeURIComponent(KEY) + '&limit=100');
        if(!r.ok) throw new Error('HTTP ' + r.status);
        const data = await r.json();
        if(data.errors.length === 0) { body.textContent = '✓ Sin errores recientes'; return; }
        body.textContent = data.errors.map(e => {
            const t = new Date(e.t * 1000).toLocaleString('es');
            // Si el mismo error se repitió (dedup), mostramos cuántas veces y la última.
            const n = e.count || 1;
            if(n > 1) {
                const lastT = e.last_t ? new Date(e.last_t * 1000).toLocaleString('es') : t;
                return `[${t} → ${lastT}] [${e.level}] [${e.thread}] (x${n})\n${e.msg}`;
            }
            return `[${t}] [${e.level}] [${e.thread}]\n${e.msg}`;
        }).join('\n\n');
    } catch(e) { body.textContent = '❌ ' + e.message; }
}

async function clearErrors() {
    if(!confirm('¿Borrar el log de errores?')) return;
    try {
        const r = await fetch('/api/errors', {
            method: 'DELETE',
            headers: {'X-Api-Key': KEY}
        });
        if(!r.ok) throw new Error('HTTP ' + r.status);
        document.getElementById('errors-body').textContent = '✓ Borrado';
        toast('🗑 Errores limpiados');
        pollErrorCount();
    } catch(e) { toast('❌ ' + e.message); }
}

async function pollErrorCount() {
    try {
        const r = await fetch('/health');
        if(!r.ok) return;
        const h = await r.json();
        const n = (h.errors && h.errors.in_log) || 0;
        const el = document.getElementById('stat-errlog');
        if(el) {
            el.textContent = '📋 ' + n + ' logs';
            el.style.background = n > 0 ? '#2a0a0a' : '';
        }
    } catch(e) {}
}

setInterval(pollErrorCount, 15000); pollErrorCount();

function toggleEye(id) {
    const el = document.getElementById(id);
    if(!el) return;
    el.type = el.type === 'password' ? 'text' : 'password';
}

// ── Exportar config a fichero local ──────────────────────────────────────────
async function exportConfig() {
    const status = document.getElementById('cfg-status');
    try {
        const url = '/api/config/export?key=' + encodeURIComponent(KEY);
        const r = await fetch(url);
        if(!r.ok) throw new Error('HTTP ' + r.status);
        const blob = await r.blob();
        const cd   = r.headers.get('Content-Disposition') || '';
        const fnMatch = /filename="([^"]+)"/.exec(cd);
        const fname = fnMatch ? fnMatch[1] : 'relay_config.json';
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = fname;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
        if(status) status.textContent = '✓ Descargado ' + fname;
        toast('📥 Backup descargado');
    } catch(e) {
        if(status) status.textContent = '❌ Export: ' + e.message;
        toast('❌ ' + e.message);
    }
}

// ── Importar config desde fichero ─────────────────────────────────────────────
async function importConfig(input) {
    const status = document.getElementById('cfg-status');
    const file = input.files && input.files[0];
    if(!file) return;
    try {
        const text = await file.text();
        const data = JSON.parse(text);
        const r = await fetch('/api/config/import', {
            method:  'POST',
            headers: {'Content-Type': 'application/json', 'X-Api-Key': KEY},
            body:    JSON.stringify(data),
        });
        if(!r.ok) throw new Error((await r.json()).detail || r.status);
        const res = await r.json();
        if(status) status.textContent = '✓ Importado ' + res.applied.length + ' claves';
        toast('📤 Importado: ' + res.applied.length + ' claves');
        await loadConfig();   // refrescar inputs con los nuevos valores
    } catch(e) {
        if(status) status.textContent = '❌ Import: ' + e.message;
        toast('❌ ' + e.message);
    } finally {
        input.value = '';     // permitir re-importar el mismo archivo
    }
}

// Auto-cargar config en TODOS los escenarios:
//  1) Cuando el usuario abre el <details> (toggle event).
//  2) Inmediatamente al cargar la página, así los campos están poblados incluso
//     antes de abrir el panel (evita "campos vacíos" si el usuario abre rápido).
//  3) Si el panel ya está abierto en el momento del page-load (browser state
//     restore, navegación back/forward), el toggle no dispara → cargamos aquí.
//
// _loaded actúa como guard contra dobles cargas — la primera que llegue gana.
(function autoLoadConfig() {
    const panel = document.getElementById('cfg-panel');
    const tryLoad = () => {
        if(panel && panel._loaded) return;
        if(panel) panel._loaded = true;
        loadConfig();
    };
    if(panel) {
        panel.addEventListener('toggle', function() { if(this.open) tryLoad(); });
        if(panel.open) tryLoad();
    }
    // Failsafe: si por alguna razón el panel aún no existe (timing), cargar igualmente.
    if(document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', tryLoad, { once: true });
    } else {
        tryLoad();
    }
})();

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg; t.style.display = 'block';
    setTimeout(() => t.style.display = 'none', 2200);
}

// ── Atajos teclado (cuando hay 1 sola tarjeta activa) ────────────────────────
let _ki = 0;
document.addEventListener('keydown', ev => {
    const cards = document.querySelectorAll('.card[data-status="awaiting_review"]');
    if(cards.length !== 1) return;
    const jid = cards[0].dataset.jid;
    if(ev.key === 'Enter') { ev.preventDefault(); sendToMobile(jid); return; }
    if(/^[1-9]$/.test(ev.key)) {
        _ki = parseInt(ev.key, 10) - 1; ev.preventDefault();
        const keys = document.querySelectorAll(`#keys-${jid} .key`);
        keys.forEach((k,i) => k.style.outline = i===_ki ? '3px solid #9b85ff' : '');
        return;
    }
    const k = ev.key.toUpperCase();
    if(_LET.includes(k)) {
        const keys = document.querySelectorAll(`#keys-${jid} .key`);
        if(keys[_ki]) {
            const lEl = keys[_ki].querySelector('.key-letter');
            lEl.textContent = k;
            keys[_ki].className = keys[_ki].className.replace(/k-[ABCDX]/, 'k-' + k);
            _markDirty(jid); _syncTxt(jid); _updPreview(jid);
            _ki = Math.min(_ki + 1, keys.length - 1);
            keys.forEach((kk,i) => kk.style.outline = i===_ki ? '3px solid #9b85ff' : '');
        }
        ev.preventDefault();
    }
});

// Polling adaptativo: chain de setTimeout en vez de setInterval fijo.
// - Si hay jobs activos (pending/awaiting_review) → 1 s (latencia rápida).
// - Si todo está done/error/idle → 5 s (reduce carga del servidor 5×).
// _activeJobsCount lo actualiza applyJobs() después de cada poll exitoso.
// Si el poll falla, mantiene el intervalo activo para reintentar agresivo.
let _activeJobsCount = 0;
const POLL_FAST_MS = 1000;
const POLL_IDLE_MS = 5000;
function scheduleNextPoll() {
    const next = (_activeJobsCount > 0 || _pollFails > 0) ? POLL_FAST_MS : POLL_IDLE_MS;
    setTimeout(async () => {
        try { await poll(); } catch(_) {}
        scheduleNextPoll();
    }, next);
}
setInterval(ticks, 1000);
poll().finally(scheduleNextPoll); ticks();
</script>
</body>
</html>"""


def _render_panel_html(key: str) -> str:
    """Sustituye los placeholders __VAR__ con los valores actuales y devuelve HTML."""
    return (
        _PANEL_HTML_TEMPLATE
        .replace('__KEY_REPR__', repr(key))
        .replace('__CLAUDE_MODEL__', CLAUDE_MODEL)
        .replace('__OPENAI_MODEL__', OPENAI_MODEL)
        .replace('__GEMINI_MODEL__', GEMINI_MODEL)
        .replace('__DEEPSEEK_MODEL__', DEEPSEEK_MODEL)
        .replace('__MISTRAL_MODEL__', MISTRAL_MODEL)
        .replace('__MIMO_MODEL__', MIMO_MODEL)
        .replace('__QWEN_VIDEO_MODEL__', QWEN_VIDEO_MODEL)
        .replace('__KIMI_VIDEO_MODEL__', KIMI_VIDEO_MODEL)
        .replace('__CLAUDE_VIDEO_OCR_MODEL__', CLAUDE_VIDEO_OCR_MODEL)
        .replace('__GEMINI_VIDEO_MODEL__', GEMINI_VIDEO_MODEL)
        .replace('__MIMO_VIDEO_MODEL__', MIMO_VIDEO_MODEL)
        .replace('__OPENAI_VIDEO_OCR_MODEL__', OPENAI_VIDEO_OCR_MODEL)
        .replace('__NVIDIA_MODEL__', NVIDIA_MODEL)
    )



# ─── Smoke-test de arranque: validar que el panel renderiza ──────────────────
# Tras el refactor a string normal + .replace() (en lugar de f-string), Python
# YA NO interpreta llaves del template para nada — el bug original (`{x}` en
# comentario JS → NameError) es IMPOSIBLE de reintroducir por accidente.
#
# Este smoke-test sigue valiendo como defensa contra:
#   1. Que alguien añada un marcador `__NUEVA_VAR__` al template pero olvide
#      el .replace correspondiente → quedaría literal en el HTML y este test lo
#      detecta antes de que llegue al navegador.
#   2. Cualquier otro fallo de runtime al construir el HTML.
# Si lanza, loguea ERROR crítico (visible en /api/errors). NO aborta el arranque:
# la API sigue funcionando aunque el panel esté roto.
try:
    _panel_test_html = _render_panel_html("_smoke_test_")
    if len(_panel_test_html) < 1000:
        logger.error("💥 Panel renderizó solo %d chars — sospechoso (esperado >100KB)",
                     len(_panel_test_html))
    else:
        # Detectar marcadores `__VAR__` huérfanos (sin .replace correspondiente).
        # Si el operador ve esto en logs, sabe que tiene que añadir un .replace.
        import re as _re_panel
        _leftover = set(_re_panel.findall(r'__[A-Z][A-Z0-9_]+__', _panel_test_html))
        if _leftover:
            logger.error("💥 Panel HTML con marcadores SIN sustituir: %s — añade el .replace en _render_panel_html",
                         sorted(_leftover))
        else:
            logger.info("✅ Panel HTML smoke-test OK (%d chars, 0 marcadores residuales)",
                        len(_panel_test_html))
    del _panel_test_html
except Exception as _panel_exc:
    logger.error("💥 CRÍTICO: _render_panel_html() lanza al renderizar — el /panel devolverá fallback HTML "
                 "(API sigue funcional). Causa: %s: %s", type(_panel_exc).__name__, _panel_exc)
    traceback.print_exc(file=sys.stdout)


# ─── Shutdown hook: flush final de JOBS al recibir SIGTERM/SIGINT ────────────
import atexit
import signal as _signal

def _flush_on_shutdown(*_a):
    """Vuelca JOBS a disco antes de morir y cierra HTTP clients limpiamente.
    Cubre el caso de un shutdown limpio (Render/Railway envían SIGTERM antes
    de matar el contenedor)."""
    try:
        if _JOBS_DIRTY.is_set() or JOBS:
            _save_jobs_to_file()
            # El disco de Render/Railway es EFÍMERO: tras el SIGTERM de un deploy
            # se borra, así que el save a disco NO sobrevive al redeploy. Volcamos
            # también a la NUBE (fuente de verdad) para no perder los jobs en vuelo
            # (awaiting_review/pending) entre deploys.
            # C14b: ambas DBs EN PARALELO con join acotado. Antes eran SECUENCIALES:
            # si Supabase colgaba hasta su timeout httpx (~15s), a Appwrite podía
            # no darle tiempo a ejecutarse antes del SIGKILL de Render (la ventana
            # de gracia tras SIGTERM es limitada). En paralelo ambas tienen su
            # oportunidad dentro de esa ventana. Daemon + join acotado = nunca
            # bloquea el cierre indefinidamente.
            def _safe_supa_sd():
                try: _supabase_save_jobs()
                except Exception: pass
            def _safe_appw_sd():
                try: _appwrite_save_jobs()
                except Exception: pass
            _t_supa = threading.Thread(target=_safe_supa_sd, daemon=True, name="shutdown-supa")
            _t_appw = threading.Thread(target=_safe_appw_sd, daemon=True, name="shutdown-appw")
            _t_supa.start(); _t_appw.start()
            _t_supa.join(timeout=8.0)
            _t_appw.join(timeout=8.0)
            logger.info("💾 Flush final de JOBS antes de salir (disco + nube, paralelo)")
    except Exception:
        pass
    # Cerrar HTTP clients para liberar sockets y evitar warnings de ResourceWarning.
    for _cli, _name in ((HTTP_CLIENT, "HTTP_CLIENT"), (DB_HTTP_CLIENT, "DB_HTTP_CLIENT"),
                        (TAVILY_HTTP_CLIENT, "TAVILY_HTTP_CLIENT")):
        try:
            _cli.close()
        except Exception:
            pass

atexit.register(_flush_on_shutdown)
for _sig in (_signal.SIGTERM, _signal.SIGINT):
    try:
        _orig = _signal.getsignal(_sig)
        def _handler(s, f, _o=_orig):
            _flush_on_shutdown()
            if callable(_o): _o(s, f)
            else: sys.exit(0)
        _signal.signal(_sig, _handler)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8123"))
    print(f"[bolsillo-ia-relay] http://0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
