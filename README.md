# Image Tools

Batch upload images, let an AI model generate clearer Chinese filenames, and remove image backgrounds with `rembg`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export DASHSCOPE_API_KEY="your-api-key"
python app.py
```

The app reads `PORT` and `GRADIO_SERVER_NAME` from the environment, so it can run locally and on hosted platforms like Render.

## Deploy To Render

1. Push this repository to GitHub.
2. In Render, create a new `Blueprint` service from the repo.
3. Render will read [render.yaml](</Users/huapingyu/dev/Infographic Splitter/render.yaml>).
4. Set the secret env var `DASHSCOPE_API_KEY` in Render.
5. Deploy the service.

Notes:
- Render provides the `PORT` value automatically.
- `GRADIO_SERVER_NAME` is set to `0.0.0.0` in `render.yaml`.
- The `output/` directory is ephemeral on Render, so generated files are only suitable for per-session download, not long-term storage.

## Workflow

```text
Tab 1: Upload multiple images -> AI generates Chinese names -> Download renamed copies / zip
Tab 2: Upload multiple images -> Remove backgrounds -> Download transparent copies / zip
```

## Notes

- The original files are not modified.
- New copies are written to an `output/session-*/renamed/` directory.
- The zip file contains only the renamed images.
- If multiple images get the same name, the app appends `_2`, `_3`, and so on.
- Transparent PNG copies are written to an `output/session-*/transparent/` directory.
- The transparent zip file contains only the processed PNG images.
