# Chess Image to FEN

A weekend experiment using a vision language model (VLM) to solve a problem I kept running into over the board: I'd reach an interesting position, want to analyze it later on Lichess, and have no easy way to get it into the engine from a photo.

This script takes an image of a chess board (photo of a physical board or a digital screenshot) and outputs a FEN string you can drop straight into Lichess analysis.

## How it works

The image gets sent to Gemini, which reads the board rank by rank, figures out orientation (white or black at the bottom), and returns the piece placement. The script wraps that into a full FEN string, validates it with `python-chess`, and spits out a ready-to-use Lichess URL.

The interesting design constraint: FEN only captures a static position, so move history is lost. That's fine for analysis purposes — Lichess just needs the position.

## Setup

Get a free API key at [aistudio.google.com](https://aistudio.google.com), then:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY='your_key_here'   # or $env:GEMINI_API_KEY on PowerShell
```

## Usage

```bash
python chess_ocr.py board.jpg --turn white
python chess_ocr.py board.jpg --turn black --verbose   # shows Gemini's reasoning
python chess_ocr.py board.jpg --model gemini-2.5-flash  # specify model
```

Output looks like:
```
FEN: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w - - 0 1
[OK] Validated by python-chess

Lichess analysis URL:
  https://lichess.org/analysis/r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R%20w%20-%20-%200%201
```

## Current status

Digital board screenshots work fine. Physical boards are a different story — piece misidentification is common enough that the output FEN often doesn't match the actual position. Going to keep playing around with this: better prompting, maybe preprocessing the image to isolate the board first, or trying different models. More to come.

## What I learned

The key prompting insight is to make the model reason rank by rank rather than guess the whole FEN at once — that alone improved accuracy a lot. Physical board photos are much harder than screenshots due to shadows and piece angles. And castling rights/en passant can't be recovered from a single image, so those are always set to `-`, which is fine for analysis purposes.
