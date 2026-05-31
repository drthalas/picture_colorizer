# Picture Colorizer

Local Telegram bot for softly colorizing black-and-white images of old paper documents, notes, handwriting, stamps, printed letters, and archival fragments.

This is intentionally not a portrait colorizer. The MVP keeps ink dark and readable, avoids rewriting letters, and adds a warm 1940s-inspired aged-paper tone.

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

## Run Locally Without Telegram

```bash
source .venv/bin/activate
picture-colorize path/to/input.jpg data/output/colorized.jpg
```

Or:

```bash
python -m app.cli path/to/input.jpg data/output/colorized.jpg
```

## Privacy and Git Hygiene

The repository ignores `.env`, `.venv`, local data folders, and generated images. Do not commit bot tokens, private input images, or generated outputs.
