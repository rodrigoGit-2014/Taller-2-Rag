# Taller 2 · RAG paso a paso

RAG con LangChain, Redis y LangGraph usando modelos de Gemini.

## Requisitos

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Una API key de Google AI Studio
- Una base Redis (Upstash o Redis Cloud)

## Instalación

```bash
git clone <url-del-repo>
cd taller_2_rag_v2
uv sync            # crea .venv e instala las dependencias fijadas en uv.lock
```

## Configuración

```bash
cp .env.example .env
```

Completa en `.env` las dos variables obligatorias:

| Variable         | Descripción                                   |
|------------------|-----------------------------------------------|
| `GOOGLE_API_KEY` | API key de Gemini (Google AI Studio)          |
| `REDIS_URL`      | URL de conexión a Redis (`redis://` o `rediss://`) |

Las demás variables de `.env.example` son opcionales. Si falta alguna de las obligatorias, el programa se detiene al iniciar.

## Ejecución

Deja tu PDF (y el TXT opcional) junto a `Taller2.py` y ejecuta:

```bash
uv run Taller2.py
```

## Sin uv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python Taller2.py
```

## Actualizar dependencias

```bash
uv add <paquete>
uv export --format requirements-txt --no-hashes --no-dev --no-emit-project -o requirements.txt
```
