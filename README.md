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
```

## Run the Telegram Bot

```bash
source .venv/bin/activate
python -m app.main
```

Then send the bot an image. The bot saves the source image locally and shows mode buttons before processing:

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

Examples:

```bash
python -m app.cli input.jpg output.jpg --mode vintage
python -m app.cli input.jpg output.jpg --mode clean
python -m app.cli input.jpg output.jpg --mode strong_1940s
python -m app.cli input.jpg output.jpg --mode stamp_focus
python -m app.cli input.jpg output.jpg --mode archive_document_4050
```

## Archive Document Pipeline

`archive_document_4050` is a lightweight OpenCV/Pillow pipeline. It does not install or require DocRes, DDColor, Real-ESRGAN, or other neural backends yet.

The pipeline is structured for later extension:

- document restoration backend placeholder for DocRes-style restoration;
- colorizer backend placeholder for DDColor-style colorization;
- deterministic local OpenCV implementation used by default.

Environment variables:

```bash
DEBUG_IMAGE_PIPELINE=false
IMAGE_MAX_LONG_SIDE=2800
```

When `DEBUG_IMAGE_PIPELINE=true`, intermediate masks and layers are saved locally next to the output file. They are not sent to Telegram users.

This mode is optimized for documents, not portraits. If a document looks prettier but text is harder to read, that is a bad result and the pipeline should be tuned for readability.

## Privacy and Git Hygiene

The repository ignores `.env`, `.venv`, local data folders, and generated images. Do not commit bot tokens, local input images, private documents, or generated outputs.
