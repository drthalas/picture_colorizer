# Picture Colorizer

Local Telegram bot and CLI tool for improving black-and-white images of old paper documents, notes, handwriting, stamps, printed letters, and archival fragments.

This is intentionally not a portrait colorizer. The processing keeps ink dark and readable, avoids rewriting letters, and favors document readability over heavy stylization.

## Requirements

- macOS
- Python 3.11
- A Telegram bot token from BotFather

## Setup

```bash
cd ~/Projects/picture-colorizer
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
```

Edit `.env` and set:

```bash
TELEGRAM_BOT_TOKEN=your_token_here
ADMIN_USER_ID=your_telegram_user_id
AUTHORIZED_USERS_PATH=data/authorized_users.json
DEBUG_IMAGE_PIPELINE=false
IMAGE_MAX_LONG_SIDE=2800
LOCAL_VLM_ENABLED=false
LOCAL_VLM_PROVIDER=ollama
LOCAL_VLM_BASE_URL=http://127.0.0.1:11434
LOCAL_VLM_MODEL=qwen2.5vl:7b
LOCAL_VLM_FALLBACK_MODEL=qwen2.5vl:3b
LOCAL_VLM_TIMEOUT_SECONDS=180
LOCAL_OCR_ENABLED=false
LOCAL_OCR_PROVIDER=paddleocr
LOCAL_OCR_LANG=ru
LOCAL_OCR_MIN_CONFIDENCE=0.35
LOCAL_OCR_MIN_SIMILARITY=0.58
LOCAL_OCR_MAX_TEXT_DROP_RATIO=0.35
DOCUMENT_RESTORER_PROVIDER=opencv
DOCRES_REPO_DIR=.external/DocRes
DOCRES_PYTHON=
DOCRES_TASK=appearance
DOCRES_TIMEOUT_SECONDS=900
DOCRES_SAVE_DTSPROMPT=false
```

## Run the Telegram Bot

```bash
source .venv/bin/activate
python -m app.main
```

Then send the bot an image. The bot saves the source image locally and shows mode buttons before processing:

- 🤖 Авто-анализ локальной моделью
- 🖼 Фото/портрет
- ⚙️ Стандартная обработка

Access is controlled by the bot administrator. New users see a `Подключить` button. When they press it, the administrator receives a Telegram request with `Подтвердить` and `Отклонить` buttons. Approved user ids are saved locally in `data/authorized_users.json`.

## Run Locally Without Telegram

```bash
source .venv/bin/activate
picture-colorize path/to/input.jpg data/output/colorized.jpg --mode vintage
```

Or:

```bash
python -m app.cli path/to/input.jpg data/output/colorized.jpg --mode vintage
```

## Processing Modes

- `vintage` is the default mode. It adds warm aged paper, dark brown-black ink, and a soft archival look without making the document look overprocessed.
- `clean` improves readability with stronger contrast, nearly no added color, and only a minimal warm paper tone.
- `strong_1940s` applies a stronger 1940s-style look with warmer paper, more visible sepia, and a light vignette while keeping text readable.
- `stamp_focus` gently emphasizes existing red-brown or blue stamp-like areas when they are already present. It does not invent stamps or redraw missing marks.
- `archive_document_4050` is optimized for old documents, award sheets, certificates, letters, and pages with handwritten or printed text. It builds a readability layer, text/line/stamp mask, controlled vintage color layer, then blends text back over the color so readability stays first.
- `archive_document` is a short alias for `archive_document_4050`.
- `document_readability` and `standard` are conservative readability aliases for the current clean pipeline.
- `auto_vlm` uses local Qwen2.5-VL through Ollama to analyze the image, select a local OpenCV/Pillow pipeline, then compare before/after for readability risk. The model does not edit pixels directly.

Examples:

```bash
python -m app.cli input.jpg output.jpg --mode vintage
python -m app.cli input.jpg output.jpg --mode clean
python -m app.cli input.jpg output.jpg --mode strong_1940s
python -m app.cli input.jpg output.jpg --mode stamp_focus
python -m app.cli input.jpg output.jpg --mode archive_document_4050
python -m app.cli input.jpg output.jpg --mode archive_document
python -m app.cli input.jpg output.jpg --mode document_readability
python -m app.cli input.jpg output.jpg --mode standard
python -m app.cli input.jpg output.jpg --mode auto_vlm
```

## Local Qwen2.5-VL Analysis

The optional local VLM mode uses Qwen2.5-VL as a visual analysis and control layer. It classifies the input as document/photo/portrait/mixed, detects text/table/stamp/handwriting risk, recommends conservative processing settings, and checks whether the processed result preserved readability. Pixel editing still happens in the local OpenCV/Pillow pipelines.

Install and start Ollama:

```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5vl:7b
ollama pull qwen2.5vl:3b
```

Enable the mode in `.env`:

```bash
LOCAL_VLM_ENABLED=true
LOCAL_VLM_PROVIDER=ollama
LOCAL_VLM_BASE_URL=http://127.0.0.1:11434
LOCAL_VLM_MODEL=qwen2.5vl:7b
LOCAL_VLM_FALLBACK_MODEL=qwen2.5vl:3b
LOCAL_VLM_TIMEOUT_SECONDS=180
```

Diagnostic check:

```bash
python -m app.vlm_check input.jpg
```

For `auto_vlm`, analysis and comparison JSON files are saved next to the output image as `*.vlm_analysis.json` and `*.vlm_compare.json`. If Qwen2.5-VL reports that readability became worse, the bot sends a warning instead of silently presenting the result as clean.

Troubleshooting:

- If Ollama is not running, start it with `brew services start ollama` or `ollama serve`.
- If the model is missing, run `ollama pull qwen2.5vl:7b`; use `qwen2.5vl:3b` as a lighter fallback.
- If memory is tight or responses are slow, use `LOCAL_VLM_MODEL=qwen2.5vl:3b`.
- If the model returns invalid JSON, the request fails gracefully and the old non-VLM modes still work.
- If `LOCAL_VLM_ENABLED=false`, the bot and CLI keep using the existing local pipelines without calling Ollama.

## Optional PaddleOCR Text Control

PaddleOCR is optional and disabled by default. When enabled, `auto_vlm` runs OCR before and after processing, saves sidecar JSON files, and rejects a result if recognized text drops too much. This is a control layer only; it does not rewrite letters.

Install the optional package:

```bash
source .venv/bin/activate
python -m pip install -e ".[ocr]"
```

The `ocr` extra installs PaddleOCR and the local PaddlePaddle runtime. The first OCR run can download recognition models into `~/.paddlex`.

Enable OCR control in `.env`:

```bash
LOCAL_OCR_ENABLED=true
LOCAL_OCR_PROVIDER=paddleocr
LOCAL_OCR_LANG=ru
LOCAL_OCR_MIN_CONFIDENCE=0.35
LOCAL_OCR_MIN_SIMILARITY=0.58
LOCAL_OCR_MAX_TEXT_DROP_RATIO=0.35
```

Diagnostic check:

```bash
python -m app.ocr_check input.jpg
```

When enabled, `auto_vlm` writes:

- `*.ocr_before.json`
- `*.ocr_after.json`
- `*.ocr_compare.json`

If OCR is unavailable or fails, the bot logs/saves that state and keeps the non-OCR path working. If OCR succeeds and detects likely text loss, the bot returns the original image with a warning instead of silently sending a degraded result.

## Archive Document Pipeline

`archive_document_4050` is a lightweight OpenCV/Pillow pipeline. It does not install or require DocRes, DDColor, Real-ESRGAN, or other neural backends yet.

The pipeline is structured for later extension:

- document restoration backend placeholder for DocRes-style restoration;
- colorizer backend placeholder for DDColor-style colorization;
- deterministic local OpenCV implementation used by default.

`DOCUMENT_RESTORER_PROVIDER=opencv` currently keeps restoration as a local passthrough backend. `docres` runs the external official DocRes `inference.py` script through a subprocess and stores the restored intermediate image next to the bot output as `*.docres_restored.jpg`.

DocRes setup:

```bash
source .venv/bin/activate
python -m pip install -e ".[docres]"
mkdir -p .external
git clone https://github.com/ZZZHANG-jx/DocRes .external/DocRes
cd .external/DocRes
mkdir -p data/MBD/checkpoint checkpoints
```

Then place weights:

- `mbd.pkl` at `.external/DocRes/data/MBD/checkpoint/mbd.pkl`
- `docres.pkl` at `.external/DocRes/checkpoints/docres.pkl`

The official DocRes README links the required weights and documents inference tasks such as `appearance`, `dewarping`, `deshadowing`, `deblurring`, `binarization`, and `end2end`.

On Apple Silicon, do not use DocRes' original CUDA-pinned `requirements.txt`; use this project's `docres` extra instead. CPU inference can take several minutes per image, so keep `DOCUMENT_RESTORER_PROVIDER=opencv` for the Telegram bot until the diagnostic result and latency are acceptable.

Diagnostic check:

```bash
python -m app.docres_check input.jpg data/output/docres-restored.jpg --task appearance
```

Enable only after the diagnostic succeeds:

```bash
DOCUMENT_RESTORER_PROVIDER=docres
DOCRES_REPO_DIR=.external/DocRes
DOCRES_PYTHON=
DOCRES_TASK=appearance
DOCRES_TIMEOUT_SECONDS=900
DOCRES_SAVE_DTSPROMPT=false
```

If DocRes is enabled but the repo or weights are missing, `auto_vlm` fails clearly instead of silently changing the image.

Environment variables:

```bash
DEBUG_IMAGE_PIPELINE=false
IMAGE_MAX_LONG_SIDE=2800
```

When `DEBUG_IMAGE_PIPELINE=true`, intermediate masks and layers are saved locally next to the output file. They are not sent to Telegram users.

This mode is optimized for documents, not portraits. If a document looks prettier but text is harder to read, that is a bad result and the pipeline should be tuned for readability.

## Privacy and Git Hygiene

The repository ignores `.env`, `.venv`, local data folders, and generated images. Do not commit bot tokens, local input images, private documents, or generated outputs.
