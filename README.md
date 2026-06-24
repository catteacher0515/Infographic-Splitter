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

## V2 AI Semantic Grouping

V2 can optionally use Qwen vision models through Alibaba Cloud Model Studio / DashScope.

Set environment variables before launching the app:

```bash
export DASHSCOPE_API_KEY="your-api-key"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export QWEN_MODEL="qwen3-vl-flash"
```

Run a manual smoke test:

```bash
python scripts/smoke_qwen_vision.py
```

Then start the app:

```bash
python app.py
```

Workflow:

```text
Upload image
-> Start splitting
-> AI semantic merge
-> Review grouped elements
-> Export ZIP
```

The AI step sends the annotated candidate image and candidate box JSON to the configured vision model. API keys are read only from environment variables and are not written to exports.
