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
```

## Run the Telegram Bot

```bash
source .venv/bin/activate
python -m app.main
```

Then send the bot a black-and-white image of a paper document, note, stamp, letter, or archival fragment.

The Telegram bot currently processes images automatically with the default `vintage` mode.

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

Examples:

```bash
python -m app.cli input.jpg output.jpg --mode vintage
python -m app.cli input.jpg output.jpg --mode clean
python -m app.cli input.jpg output.jpg --mode strong_1940s
python -m app.cli input.jpg output.jpg --mode stamp_focus
```

## Privacy and Git Hygiene

The repository ignores `.env`, `.venv`, local data folders, and generated images. Do not commit bot tokens, local input images, private documents, or generated outputs.
