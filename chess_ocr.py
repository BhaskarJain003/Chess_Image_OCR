# chess_ocr.py -- Convert a chess board image to FEN notation using Gemini Vision.
#
# CLI usage:
#   python chess_ocr.py path/to/board.png --turn white
#   python chess_ocr.py path/to/board.jpg --turn black --verbose
#   python chess_ocr.py board.jpg --model gemini-2.5-flash
#
# Module usage:
#   from chess_ocr import image_to_fen
#   result = image_to_fen("board.png", turn="w")
#   print(result["fen"])

import argparse
import os
import sys
import urllib.parse
from pathlib import Path

import chess
from google import genai
from PIL import Image


PROMPT = """You are an expert chess analyst. Your job is to read a chess board image and output the FEN piece-placement string for the position shown.

Step 1 - Determine orientation.
Look at which color's pieces are closest to the bottom of the image. State: "White is at the bottom" or "Black is at the bottom."

Step 2 - Map the board.
Always output FEN from White's perspective: rank 8 (Black's back rank) at the top, rank 1 (White's back rank) at the bottom, files a-h left to right.
If Black is at the bottom in the image, mentally flip the board before reading it.

Step 3 - Read each rank.
For ranks 8 down to 1, list every square a-h and identify the piece or empty square.
Use standard FEN piece letters:
  White: K Q R B N P
  Black: k q r b n p

Step 4 - Build the FEN piece-placement string.
Consecutive empty squares on a rank are written as a digit (1-8).
Ranks are separated by "/".

Important notes for physical board photos:
- Shadows and angles can make pieces hard to identify -- use context clues.
- Pawns on the back ranks are illegal; if you see one there, reconsider.
- Both sides must have exactly one king.

Show your reasoning for each rank, then on the very last line write:

FEN: <piece_placement_only>

Do not include turn, castling, en passant, or move counters -- just the piece placement field.
"""

DEFAULT_MODEL = "gemini-2.5-flash"


def image_to_fen(image_path, turn="w", model=DEFAULT_MODEL):
    turn = turn.strip().lower()
    if turn in ("white", "w"):
        turn = "w"
    elif turn in ("black", "b"):
        turn = "b"
    else:
        return {"fen": None, "valid": False,
                "error": "Invalid turn '{}'. Use w/white or b/black.".format(turn)}

    path = Path(image_path)
    if not path.exists():
        return {"fen": None, "valid": False,
                "error": "File not found: {}".format(image_path)}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"fen": None, "valid": False,
                "error": "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com"}

    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        return {"fen": None, "valid": False,
                "error": "Failed to open image: {}".format(exc)}

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=[PROMPT, image],
        )
        raw = response.text
    except Exception as exc:
        return {"fen": None, "valid": False,
                "error": "Gemini API error: {}".format(exc)}

    placement = None
    for line in reversed(raw.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("FEN:"):
            placement = stripped.split(":", 1)[1].strip()
            break

    if not placement:
        return {"fen": None, "valid": False,
                "error": "Could not find 'FEN: ...' in Gemini response.",
                "raw_response": raw}

    full_fen = "{} {} - - 0 1".format(placement, turn)

    try:
        chess.Board(full_fen)
        valid = True
        error = None
    except ValueError as exc:
        valid = False
        error = str(exc)

    lichess_url = "https://lichess.org/analysis/" + urllib.parse.quote(full_fen, safe="")

    return {"fen": full_fen, "valid": valid, "error": error,
            "raw_response": raw, "lichess_url": lichess_url}


def main():
    parser = argparse.ArgumentParser(
        description="Convert a chess board image to FEN using Gemini Vision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python chess_ocr.py board.png --turn white\n"
            "  python chess_ocr.py board.jpg --turn black --verbose\n"
            "  python chess_ocr.py board.jpg --model gemini-2.0-flash\n\n"
            "Set your Gemini API key (free at https://aistudio.google.com):\n"
            "  PowerShell: $env:GEMINI_API_KEY='your_key'\n"
            "  Linux/Mac:  export GEMINI_API_KEY=your_key"
        ),
    )
    parser.add_argument("image", help="Path to the chess board image")
    parser.add_argument("--turn", default="white", metavar="COLOR",
                        help="Whose turn: 'white' or 'black' (default: white)")
    parser.add_argument("--model", default=DEFAULT_MODEL, metavar="MODEL",
                        help="Gemini model name (default: {})".format(DEFAULT_MODEL))
    parser.add_argument("--verbose", action="store_true",
                        help="Print Gemini's full reasoning")
    args = parser.parse_args()

    print("Image : " + args.image)
    print("Turn  : " + args.turn)
    print("Model : " + args.model)
    print("Querying Gemini...\n")

    result = image_to_fen(args.image, turn=args.turn, model=args.model)

    if args.verbose and result.get("raw_response"):
        print("-" * 60)
        print("Gemini reasoning:")
        print("-" * 60)
        print(result["raw_response"])
        print("-" * 60)
        print("")

    if result["fen"] is None:
        print("Error: " + str(result["error"]), file=sys.stderr)
        sys.exit(1)

    print("FEN: " + result["fen"])

    if result["valid"]:
        print("[OK] Validated by python-chess")
    else:
        print("[WARN] Validation failed: " + str(result["error"]))
        print("The FEN may still work -- paste it manually into Lichess.")

    print("\nLichess analysis URL:")
    print("  " + result["lichess_url"])


if __name__ == "__main__":
    main()
