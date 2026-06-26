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
