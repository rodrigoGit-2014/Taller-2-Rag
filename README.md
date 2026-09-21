# Taller 2 · Tu propio RAG paso a paso

RAG con LangChain, Redis y LangGraph usando modelos de Gemini.

## Cómo ejecutarlo

### 1. Clonar el repositorio

```bash
git clone git@github.com:rodrigoGit-2014/Taller-2-Rag.git
cd Taller-2-Rag
```

> Si no tienes llave SSH configurada en GitHub, usa `git clone https://github.com/rodrigoGit-2014/Taller-2-Rag.git`.

### 2. Instalar las dependencias

```bash
uv sync
```

Crea el entorno `.venv` con Python 3.12 e instala las versiones fijadas en `uv.lock`.

### 3. Crear el archivo de variables de entorno

```bash
cp .env.example .env
```

### 4. Completar las variables obligatorias en `.env`

```
GOOGLE_API_KEY=tu_api_key_de_google_ai_studio
REDIS_URL=rediss://default:TOKEN@host:6379
```

- `GOOGLE_API_KEY`: se obtiene en https://aistudio.google.com/apikey
- `REDIS_URL`: Upstash usa `rediss://` (TLS); Redis Cloud usa `redis://default:PASS@host:puerto`

El resto de las variables de `.env.example` son opcionales. Si falta alguna de las dos obligatorias, el programa se detiene al iniciar.

### 5. Copiar tus documentos a `docs/`

Los PDF no se suben al repositorio, así que la carpeta `docs/` viene vacía:

```bash
cp /ruta/a/tu/documento.pdf docs/
```

Se aceptan archivos PDF o TXT.

### 6. Ejecutar

```bash
uv run Taller2.py
```

## Notas

- **Si no tienes uv**, instálalo con:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Si aparece un aviso sobre `VIRTUAL_ENV`**, tienes otro entorno virtual activo. Es solo un aviso: puedes ignorarlo o ejecutar `deactivate` antes.
- **Sin uv**, también funciona con pip:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  python Taller2.py
  ```
