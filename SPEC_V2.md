# Infographic Splitter V2 规格说明

## 1. 目标

V2 在 V1 的 OpenCV 候选框检测基础上，引入多模态模型做语义合并。

核心目标不是让 AI 直接裁剪图片，而是让 AI 判断：

- 哪些 OpenCV 候选框应该合并成一个语义元素。
- 哪些候选框应该保留为独立元素。
- 哪些候选框可以忽略。
- 每个导出元素可以使用什么文件名。

V2 推荐默认模型：

```text
qwen3-vl-flash
```

调用方式：

```text
阿里云百炼 / DashScope OpenAI-compatible API
```

## 2. 背景

V1 已经实现：

```text
上传图片
-> OpenCV 自动检测候选框
-> 自动裁剪
-> 预览候选元素
-> 勾选保留
-> 编辑文件名
-> 导出 ZIP
```

但 V1 的瓶颈是：OpenCV 只能理解像素距离、轮廓、面积和形态学关系，不能理解语义组合。

典型问题：

- 第三部分中，小猫 + 绳子 + 被拉的小猫应该作为一个元素。
- 第四部分中，循环箭头 + 图标 + 小猫 + 标签文字应该作为一个循环图元素。
- 等号、箭头等通用小元素可以忽略或复用，不一定需要每次导出。
- 标题框 + 序号是否合并，取决于语义和用户偏好，不是单纯像素距离。

因此 V2 采用：

```text
OpenCV 候选框
+ 多模态语义合并
+ 本地裁剪导出
```

## 3. 核心原则

### 3.1 OpenCV 仍然是候选框来源

AI 不直接负责检测所有坐标，也不直接负责裁剪。

V2 流程中，候选框仍由 OpenCV 生成：

```text
原图
-> OpenCV 检测候选框
-> 候选框编号
-> 生成带编号预览图
```

AI 只读取：

- 原图或压缩图。
- 带编号候选框预览图。
- 候选框 JSON。

AI 输出：

- 分组建议。
- 忽略建议。
- 命名建议。

本地程序根据 AI 输出合并 bounding box，再裁剪原图。

### 3.2 AI 只做语义决策

AI 不直接输出最终裁剪图片。

AI 输出必须是结构化 JSON，程序负责校验和执行。

### 3.3 保留 V1 流程

V2 不能破坏 V1 的纯 OpenCV 流程。

页面中需要同时存在：

```text
开始拆分
AI 语义合并
导出 ZIP
```

如果 AI 调用失败，用户仍然可以使用 V1 的候选框结果导出。

### 3.4 API Key 不写入代码

所有密钥必须通过环境变量读取。

禁止：

- 把 API Key 写入源码。
- 把 API Key 写入 `.env` 并提交。
- 在日志、manifest、ZIP 中输出 API Key。
- 在异常信息中打印完整请求 header。

## 4. V2 用户流程

```text
上传图片
↓
点击开始拆分
↓
OpenCV 自动检测候选框
↓
预览 V1 候选元素
↓
点击 AI 语义合并
↓
生成带编号候选框预览图
↓
发送给 qwen3-vl-flash
↓
AI 返回分组 JSON
↓
本地合并 bbox 并重新裁剪
↓
预览 V2 分组结果
↓
勾选保留元素
↓
编辑 file 字段
↓
导出 ZIP
```

## 5. V2 支持范围

V2 仍然只支持 V1 定义的图片范围：

- 白色或浅色背景。
- 黑色线稿。
- 知识类总结图。
- 手绘风信息图。
- 元素之间有明显留白。

V2 新增支持：

- 语义场景合并。
- 通用小元素忽略建议。
- AI 文件名建议。
- 分组结果预览。

## 6. V2 非目标

V2 不做：

- AI 直接裁剪。
- SAM 分割。
- 透明背景导出。
- 手动框选。
- 手动拖拽合并。
- PSD 拆层。
- 复杂照片理解。
- 任意复杂插画拆层。
- 训练自定义模型。
- 本地部署 Qwen-VL。
- 自动上传原图到第三方存储。

手动框选、手动合并、重新检测可以作为 V3 以后能力。

## 7. 模型与配置

### 7.1 默认模型

第一版 V2 默认使用：

```text
qwen3-vl-flash
```

原因：

- 支持多模态图片输入。
- 支持 OpenAI-compatible 接口。
- 调用方式容易接入现有 Python 项目。
- 成本和速度更适合作为第一轮语义合并验证。

### 7.2 环境变量

推荐配置：

```text
VISION_PROVIDER=qwen
DASHSCOPE_API_KEY=your_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-vl-flash
```

默认值：

```text
VISION_PROVIDER=qwen
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-vl-flash
```

必填：

```text
DASHSCOPE_API_KEY
```

如果缺少 `DASHSCOPE_API_KEY`，页面应提示用户设置环境变量，不应崩溃。

### 7.3 依赖

V2 可以新增：

```text
openai
```

用途：

- 使用 OpenAI-compatible client 调用阿里云百炼。

V2 不新增：

- SAM。
- OCR。
- 本地大模型推理框架。
- GPU 相关依赖。

## 8. 数据结构

### 8.1 OpenCV candidate

OpenCV 候选框结构：

```json
{
  "id": 1,
  "file": "element_001.png",
  "x": 120,
  "y": 80,
  "width": 240,
  "height": 180,
  "selected": true,
  "preview_path": "output/session/assets/element_001.png"
}
```

要求：

- `id` 从 1 开始。
- 坐标为原图像素坐标。
- 排序沿用 V1：从上到下，同一行从左到右。

### 8.2 AI request payload

发送给模型的信息包含：

```json
{
  "task": "group infographic candidates",
  "rules": {
    "merge_semantic_scenes": true,
    "keep_chinese_labels_as_elements": true,
    "ignore_reusable_symbols": true,
    "do_not_merge_whole_columns": true
  },
  "candidates": [
    {
      "id": 1,
      "x": 120,
      "y": 80,
      "width": 240,
      "height": 180
    }
  ]
}
```

同时发送图片：

- 一张带编号候选框预览图。
- 必要时附带压缩后的原图。

### 8.3 AI response schema

AI 必须返回 JSON：

```json
{
  "groups": [
    {
      "id": 1,
      "file": "prompt_title.png",
      "candidate_ids": [1, 2],
      "type": "title",
      "keep": true,
      "reason": "序号和标题框共同构成标题元素"
    },
    {
      "id": 2,
      "file": "loop_cycle.png",
      "candidate_ids": [18, 19, 20, 21, 22],
      "type": "illustration",
      "keep": true,
      "reason": "这些候选框共同构成循环流程图"
    }
  ],
  "ignored_candidate_ids": [7, 12],
  "notes": "等号和横向箭头属于可复用通用符号，建议忽略"
}
```

字段说明：

- `groups`：最终要展示和导出的语义元素。
- `groups[].file`：建议文件名，用户可编辑。
- `groups[].candidate_ids`：来自 OpenCV 的候选框 ID。
- `groups[].type`：元素类型，允许值见下文。
- `groups[].keep`：默认是否保留。
- `groups[].reason`：用于调试和解释，不进入最终 manifest。
- `ignored_candidate_ids`：建议忽略的候选框。
- `notes`：模型补充说明，可显示在调试区。

### 8.4 type 取值

第一版允许以下类型：

```text
title
label
illustration
symbol
arrow
unknown
```

含义：

- `title`：序号、标题框、标题文字。
- `label`：中文说明文字。
- `illustration`：猫、资料堆、循环图等主要插图。
- `symbol`：等号、齿轮、灯泡等小符号。
- `arrow`：箭头。
- `unknown`：模型不确定。

## 9. Prompt 设计

### 9.1 System prompt

```text
你是一个黑白手绘知识信息图的元素分组助手。

你不会直接裁剪图片。
你只根据图片和候选框编号，判断哪些候选框应该合并为一个可复用元素。

请严格返回 JSON，不要返回 Markdown，不要返回解释性正文。
```

### 9.2 User prompt

```text
请根据图片中的编号候选框，对这些候选框进行语义分组。

规则：
1. 保留中文说明文字为独立元素。
2. 语义上共同表达一个场景的图形应该合并，例如小猫 + 绳子 + 被拉的小猫。
3. 循环图应作为一个整体元素。
4. 序号 + 标题框可以合并为标题元素。
5. 等号、简单箭头等通用符号可以建议忽略。
6. 不要把整列内容合并成一个大元素。
7. 如果不确定，宁可少合并，不要过度合并。

请返回符合 schema 的 JSON。
```

### 9.3 输出约束

必须要求模型：

- 只返回 JSON。
- `candidate_ids` 必须来自输入 ID。
- 不允许创造不存在的候选框 ID。
- 不允许返回空 `candidate_ids`。
- 文件名使用英文小写、数字和下划线。
- 文件名以 `.png` 结尾。

## 10. 带编号预览图

V2 需要生成一张模型专用预览图：

```text
annotated_candidates.png
```

要求：

- 基于原图或压缩原图生成。
- 在每个候选框外画彩色矩形。
- 在候选框左上角显示候选 ID。
- ID 需要清晰可读。
- 图片最长边建议压缩到 `1600px` 以内。
- 压缩时需要保留原图坐标到预览图坐标的比例关系。

注意：

- AI 返回的是 candidate ID，不是预览图坐标。
- 本地程序使用原图坐标合并和裁剪。

## 11. 分组合并规则

本地程序收到 AI 分组后，对每个 group 计算合并 bbox：

```text
x1 = min(candidate.x)
y1 = min(candidate.y)
x2 = max(candidate.x + candidate.width)
y2 = max(candidate.y + candidate.height)
```

合并后：

```text
width = x2 - x1
height = y2 - y1
```

裁剪仍使用原图：

```python
crop = image[y:y+h, x:x+w]
```

padding 策略：

- 第一版 V2 直接复用 OpenCV candidate 已有 padding 后 bbox。
- 不额外增加 group padding。
- 后续如需要，可新增 `group_padding` 参数。

## 12. JSON 校验与容错

模型返回必须经过校验。

校验规则：

- 返回必须是合法 JSON。
- 必须包含 `groups`。
- `groups` 必须是数组。
- `candidate_ids` 必须是非空数组。
- 所有 `candidate_ids` 必须存在于当前 OpenCV candidates。
- 重复 candidate ID 需要处理。
- 文件名必须经过现有导出清洗逻辑。

重复 ID 策略：

```text
同一个 candidate 如果出现在多个 group 中，只保留第一次出现，后续 group 自动忽略该 candidate。
```

无效 group 策略：

```text
如果 group 中没有任何有效 candidate，则跳过该 group。
```

AI 调用失败策略：

```text
保留 V1 OpenCV 结果
页面显示错误信息
允许用户重新点击 AI 语义合并
```

JSON 解析失败策略：

```text
显示模型原始输出的安全摘要
保留 V1 OpenCV 结果
不导出错误结果
```

## 13. UI 要求

V2 页面在 V1 基础上新增：

### 13.1 新按钮

```text
AI 语义合并
```

按钮状态：

- 未完成 OpenCV 拆分前禁用或提示先点击“开始拆分”。
- 调用中显示 loading。
- 调用失败显示错误。

### 13.2 新配置

可在页面上显示但不要求用户每次修改：

- `model`
- `provider`
- `base_url`

API Key 不在页面中展示。

### 13.3 新预览

需要显示：

- OpenCV 候选结果。
- AI 分组结果。

第一版可以复用同一个 Gallery：

```text
点击开始拆分 -> Gallery 显示候选元素
点击 AI 语义合并 -> Gallery 替换为分组元素
```

也可以新增 Tab：

```text
OpenCV 候选
AI 分组
```

第一版优先简单实现，允许替换 Gallery。

### 13.4 表格字段

AI 分组后表格字段：

```text
selected
file
x
y
width
height
type
source_candidate_ids
reason
```

其中：

- `selected` 可编辑。
- `file` 可编辑。
- `type` 可读或可编辑均可，第一版建议只读。
- `source_candidate_ids` 只读。
- `reason` 只读。

导出 manifest 第一版仍只保留 V1 字段：

```text
id
file
x
y
width
height
```

`type`、`source_candidate_ids`、`reason` 暂不进入 manifest，后续可扩展。

## 14. ZIP 导出

V2 复用 V1 导出结构：

```text
export.zip
|-- assets
|   |-- prompt_title.png
|   |-- prompt_cat_speech.png
|   `-- loop_cycle.png
`-- manifest.json
```

如果用户执行了 AI 语义合并，则导出 AI 分组结果。

如果用户没有执行 AI 语义合并，则导出 V1 OpenCV 候选结果。

## 15. 文件命名

AI 可以建议英文文件名。

命名规则：

- 使用英文小写。
- 单词之间使用 `_`。
- 必须以 `.png` 结尾。
- 不允许路径分隔符。
- 不允许空文件名。

示例：

```text
prompt_title.png
prompt_cat_speech.png
context_documents_cat.png
harness_cats_leash.png
loop_cycle.png
```

导出前仍使用 V1 的文件名清洗逻辑。

用户在表格中编辑 `file` 字段后，以用户编辑结果为准。

## 16. 安全与隐私

V2 会把图片发送给外部多模态 API。

页面需要明确提示：

```text
AI 语义合并会将当前图片和候选框信息发送到配置的多模态模型服务。
```

不发送：

- API Key。
- 本地绝对路径。
- ZIP 文件。
- 用户目录结构。

发送：

- 压缩后的图片。
- 候选框 JSON。
- 分组任务 prompt。

日志中禁止输出：

- API Key。
- 完整 base64 图片。
- 完整请求 header。

## 17. 建议模块结构

V2 可以新增以下模块：

```text
vision_client.py
ai_grouper.py
annotator.py
```

职责：

### 17.1 `annotator.py`

负责：

- 生成带编号候选框预览图。
- 压缩图片。
- 输出 base64 data URL。

### 17.2 `vision_client.py`

负责：

- 读取环境变量。
- 初始化 OpenAI-compatible client。
- 调用 qwen3-vl-flash。
- 返回模型文本结果。

### 17.3 `ai_grouper.py`

负责：

- 构造 prompt。
- 构造候选框 JSON。
- 解析模型 JSON。
- 校验分组结果。
- 生成合并后的 elements。

已有模块复用：

- `detector.py`：继续负责 OpenCV 候选框。
- `splitter.py`：继续负责裁剪。
- `exporter.py`：继续负责 ZIP 导出。
- `app.py`：新增 UI 入口和状态管理。

## 18. 测试要求

V2 至少需要新增以下测试：

### 18.1 JSON 解析测试

覆盖：

- 合法 JSON。
- Markdown 包裹 JSON。
- 非法 JSON。
- 缺失字段。
- 空 groups。

### 18.2 分组校验测试

覆盖：

- 有效 candidate IDs。
- 不存在的 candidate ID。
- 重复 candidate ID。
- 空 candidate_ids。
- 文件名清洗。

### 18.3 bbox 合并测试

覆盖：

- 两个候选框合并。
- 多个候选框合并。
- 忽略无效候选框。
- 坐标保持原图像素。

### 18.4 AI client 测试

默认不在单元测试中真实调用 API。

单元测试使用 mock response。

真实 API 测试作为手动 smoke test：

```text
python scripts/smoke_qwen_vision.py
```

smoke test 只验证：

- `DASHSCOPE_API_KEY` 是否可用。
- `qwen3-vl-flash` 是否能读取图片。
- 是否能返回 JSON。

smoke test 不进入默认 `pytest`。

## 19. 验收标准

以 `Prompt / Context / Harness / Loop` 示例图为基准。

V2 通过标准：

- 可以在 V1 OpenCV 候选结果基础上点击 `AI 语义合并`。
- AI 能返回结构化 JSON。
- 页面能展示 AI 分组后的元素。
- 第三部分的小猫 + 绳子 + 被拉的小猫可以合成一个元素。
- 第四部分循环图可以合成一个元素。
- 中文说明文字可以作为独立元素保留。
- 标题框 + 序号允许合成标题元素。
- 等号、简单箭头允许被建议忽略。
- 用户仍可勾选、取消勾选和重命名。
- ZIP 导出结构不变。
- 未配置 API Key 时，V1 流程不受影响。
- AI 调用失败时，V1 结果不丢失。

## 20. 第一版实现优先级

推荐实现顺序：

```text
Task 1: 新增 qwen vision smoke test
Task 2: 新增 annotated candidate preview 生成
Task 3: 新增 AI prompt 和 JSON schema
Task 4: 新增 AI grouping parser 和 validator
Task 5: 新增 group bbox 合并和裁剪
Task 6: 接入 Gradio AI 语义合并按钮
Task 7: 加测试和错误处理
Task 8: 更新 README
```

## 21. 明确暂缓

以下能力先不做：

- 手动画框。
- 手动拖拽合并。
- 透明背景。
- 多模型切换 UI。
- 成本统计。
- 批量处理。
- 自动选择最优模型。
- 模型输出可视化审计面板。

这些能力可以在 V3 以后追加。

