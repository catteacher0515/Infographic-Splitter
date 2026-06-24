# Infographic Splitter

Local MVP for splitting black-and-white hand-drawn knowledge infographics into reusable visual elements.

## Scope

- Uses OpenCV, Pillow, and Gradio.
- No AI models, SAM, OCR, or large-model dependency.
- Targets white or light backgrounds with black line art and clear spacing.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open the local Gradio URL printed in the terminal.

## Workflow

```text
Upload image
-> Start splitting
-> Preview candidates
-> Select elements
-> Rename files
-> Export ZIP
```

## Output

```text
export.zip
|-- assets/
`-- manifest.json
```

See `SPEC.md` for the full MVP specification.
