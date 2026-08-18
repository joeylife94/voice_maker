# Voice Maker

Streamlit utility for:

- PDF → TXT extraction, with optional OCR
- Excel → batch MP3 generation
- Automatic language detection and per-language voice switching
- Korean, English, German, Japanese, Chinese, and Turkish

## Why mixed-language TTS does not use SSML

`edge-tts` no longer supports arbitrary custom SSML voice switching. Voice Maker therefore generates each detected language segment independently and combines the resulting MP3 segments.

Example input:

```text
I have to go to work. 나는 출근해야 해.
```

Generation plan:

```text
English segment -> English voice
Korean segment  -> Korean voice
                 -> combined 001.mp3
```

The same approach is used for German/English/Turkish Latin-script sentences and Korean/Japanese/Chinese script transitions.

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

For exact silence between language segments, install `ffmpeg` and make sure it is available on `PATH`. If ffmpeg is unavailable, the app falls back to raw MP3 stream concatenation; the audio remains usable, but the configured silence gap is omitted.

## Run

```bash
streamlit run app.py
```

## Excel format

The first column is the output filename and the second column is the text to synthesize.

| Filename | Text |
|---|---|
| 001 | I have to go to work. 나는 출근해야 해. |
| 002 | Ich trinke Kaffee. 커피를 마셔요. |

Standard headers such as `Filename | Text` and `파일명 | 내용` are detected automatically and skipped. A headerless two-column workbook is also supported.

The app creates one MP3 per data row and packages all output files into a single ZIP archive.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The unit tests validate language/script segmentation, Excel header detection, and output filename safety without calling the online TTS service.

## Notes

- TTS generation uses Microsoft Edge's online speech service through `edge-tts`; this is not offline/local TTS.
- Custom SSML input is intentionally rejected because the backend does not support arbitrary SSML voice switching.
- OCR can require a model download on first use.
