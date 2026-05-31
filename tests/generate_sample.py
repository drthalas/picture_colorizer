from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    output = Path("data/input/sample-document.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("L", (1000, 700), 232)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.rectangle((60, 50, 940, 650), outline=40, width=3)
    draw.text((110, 110), "ARCHIVAL NOTE - 1943", fill=25, font=font)
    draw.text((110, 170), "Dear friend,", fill=35, font=font)
    draw.text((110, 220), "These lines should remain readable after colorization.", fill=35, font=font)
    draw.text((110, 270), "No invented letters. No distorted handwriting.", fill=35, font=font)
    draw.ellipse((720, 110, 870, 260), outline=55, width=5)
    draw.text((755, 175), "STAMP", fill=45, font=font)

    image.save(output)
    print(output)


if __name__ == "__main__":
    main()
