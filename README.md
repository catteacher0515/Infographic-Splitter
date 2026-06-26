# Image Renamer

Batch upload images, let an AI model generate clearer Chinese filenames, and download the renamed copies plus a zip package.

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
Upload multiple images
-> AI generates names
-> Download renamed copies
-> Download zip package
```

## Notes

- The original files are not modified.
- New copies are written to an `output/session-*/renamed/` directory.
- The zip file contains only the renamed images.
- If multiple images get the same name, the app appends `_2`, `_3`, and so on.
