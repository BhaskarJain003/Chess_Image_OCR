# benchmark.py -- Compare Gemini models on chess FEN generation across multiple images.
#
# Usage: uv run python benchmark.py
# Randomly samples N images from the chesscog dataset (test + val splits),
# runs each model, and reports per-image and aggregate scores.

import json
import os
import random
import sys
import time
from pathlib import Path

import chess
from google import genai
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_DIRS = [
    "Data_ChessCog/test_extracted/test",
    "Data_ChessCog/val_extracted/val",
]

SAMPLE_SIZE = 5
RANDOM_SEED = 42

MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
]

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

Notes:
- Shadows and angles can make pieces hard to identify -- use context clues.
- Pawns on the back ranks are illegal; reconsider if you see one.
- Both sides must have exactly one king.

Show your reasoning for each rank, then on the very last line write:

FEN: <piece_placement_only>

Do not include turn, castling, en passant, or move counters -- just the piece placement field.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_pairs(dataset_dirs):
    pairs = []
    for d in dataset_dirs:
        folder = Path(d)
        if not folder.exists():
            continue
        for png in folder.glob("*.png"):
            json_file = png.with_suffix(".json")
            if json_file.exists():
                try:
                    meta = json.loads(json_file.read_text())
                    fen = meta.get("fen")
                    white_turn = meta.get("white_turn", True)
                    if fen:
                        pairs.append({
                            "image": str(png),
                            "fen": fen,
                            "turn": "w" if white_turn else "b",
                            "name": png.stem,
                        })
                except Exception:
                    pass
    return pairs

def score_placement(predicted, truth):
    try:
        pred_board = chess.Board(predicted + " w - - 0 1")
        true_board = chess.Board(truth + " w - - 0 1")
        return sum(
            pred_board.piece_at(sq) == true_board.piece_at(sq)
            for sq in chess.SQUARES
        )
    except Exception:
        return 0

def parse_placement(raw):
    if not raw:
        return None
    for line in reversed(raw.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("FEN:"):
            return stripped.split(":", 1)[1].strip()
    return None

def query_gemini(model_id, pil_image, api_key):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_id,
        contents=[PROMPT, pil_image],
    )
    return response.text

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    # Collect all paired samples
    all_pairs = collect_pairs(DATASET_DIRS)
    if not all_pairs:
        print("Error: no paired image+JSON samples found in dataset dirs")
        sys.exit(1)

    # Random sample
    random.seed(RANDOM_SEED)
    sample = random.sample(all_pairs, min(SAMPLE_SIZE, len(all_pairs)))

    print("Dataset pairs available : " + str(len(all_pairs)))
    print("Sample size             : " + str(len(sample)))
    print("Models                  : " + str(len(MODELS)))
    print("Estimated runtime       : ~" + str(len(sample) * len(MODELS) * 40 // 60) + " mins")
    print("")

    # Results: model -> list of per-image scores
    model_scores = {m: [] for m in MODELS}
    model_errors = {m: 0 for m in MODELS}

    # Print header
    img_col = 8
    score_col = 16
    header = "{:<{ic}}  {:<{sc}}  {:<{sc}}  {:<{sc}}".format(
        "Image",
        MODELS[0][:score_col], MODELS[1][:score_col], MODELS[2][:score_col],
        ic=img_col, sc=score_col
    )
    print(header)
    print("-" * len(header))

    for item in sample:
        img_path = item["image"]
        truth = item["fen"]
        name = item["name"]

        pil_image = Image.open(img_path)
        pil_image.load()

        row_scores = []
        for model_id in MODELS:
            try:
                raw = query_gemini(model_id, pil_image, api_key)
                placement = parse_placement(raw)
                if placement:
                    score = score_placement(placement, truth)
                else:
                    score = -1  # no FEN found
                    model_errors[model_id] += 1
            except Exception as e:
                score = -1
                model_errors[model_id] += 1

            model_scores[model_id].append(max(score, 0))
            row_scores.append(score)
            time.sleep(1)

        def fmt(s):
            return "ERR" if s < 0 else "{}/64".format(s)

        print("{:<{ic}}  {:<{sc}}  {:<{sc}}  {:<{sc}}  | truth: {}".format(
            name,
            fmt(row_scores[0]), fmt(row_scores[1]), fmt(row_scores[2]),
            truth,
            ic=img_col, sc=score_col
        ))

    # Summary
    print("")
    print("=" * 80)
    print("SUMMARY  (avg squares correct out of 64, across {} images)".format(len(sample)))
    print("=" * 80)
    for model_id in MODELS:
        scores = model_scores[model_id]
        avg = sum(scores) / len(scores) if scores else 0
        errs = model_errors[model_id]
        print("  {:<35}  avg {:.1f}/64  ({:.0f}%)  errors: {}".format(
            model_id, avg, avg / 64 * 100, errs
        ))
    print("=" * 80)

if __name__ == "__main__":
    main()
