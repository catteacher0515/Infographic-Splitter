# Image Tools

批量上传图片，使用 AI 生成更清晰的中文文件名，并通过 `rembg` 去除图片背景。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 中使用的是 `rembg[cpu]`，它会安装透明背景功能所需的 ONNX Runtime，适用于本地环境和 Render。

## 运行

```bash
export DASHSCOPE_API_KEY="your-api-key"
python app.py
```

应用会从环境变量中读取 `PORT` 和 `GRADIO_SERVER_NAME`，因此既可以本地运行，也可以部署到 Render 这类托管平台。

## 部署到 Render

1. 将此仓库推送到 GitHub。
2. 在 Render 中基于该仓库创建一个 `Blueprint` 服务。
3. Render 会自动读取 [render.yaml](</Users/huapingyu/dev/Infographic Splitter/render.yaml>)。
4. 在 Render 中配置密钥环境变量 `DASHSCOPE_API_KEY`。
5. 执行部署。

说明：
- Render 会自动提供 `PORT` 环境变量。
- `render.yaml` 中已将 `GRADIO_SERVER_NAME` 设为 `0.0.0.0`。
- `requirements.txt` 中的 `rembg[cpu]` 会为透明背景功能安装所需的 ONNX Runtime。
- Render 上的 `output/` 目录是临时存储，只适合当前会话下载，不适合长期保存生成文件。

## 使用流程

```text
标签页 1：上传多张图片 -> AI 生成中文文件名 -> 下载重命名后的副本或 zip
标签页 2：上传多张图片 -> 去除背景 -> 下载透明背景副本或 zip
```

## 注意事项

- 原始文件不会被修改。
- 新副本会写入 `output/session-*/renamed/` 目录。
- zip 文件中只包含重命名后的图片。
- 如果多张图片生成了相同文件名，程序会自动追加 `_2`、`_3` 等后缀。
- 透明背景 PNG 会写入 `output/session-*/transparent/` 目录。
- 透明背景 zip 中只包含处理后的 PNG 图片。
