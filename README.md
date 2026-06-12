# lora-img-prep v5.3.1

> LoRA 训练数据集预处理流水线 — 任意背景 → 白底抠图 → 裁剪 → 方形画布 → AI 超分 → 多模式 VLM 打标

[![version](https://img.shields.io/badge/version-v5.3.1-blue)](https://github.com/QYQCAMIAO/LoRA-IMG-Prep-skill)

---

## 功能流程

```
PNG / JPG / WebP
  ↓
[1] 格式筛查 → 自动跳过不支持格式
  ↓
[2] 白底 JPG 转换（四路智能决策树）
    ├─ 真透明 alpha → alpha 合成白底
    ├─ 假透明 → 转 RGB 重新判断
    ├─ 白底图 → 直接转 JPG
    └─ 其他背景 → RMBG-2.0 抠图合成白底
  ↓
[3] 裁剪白边（mask 优先 > alpha > RGB fallback）+ 16px padding
  ↓
[4] 居中 1:1 正方形画布
  ↓
[5] RealESRGAN AI 放大至 1024×1024
  ↓
[6] VLM 多模式自动打标（Qwen3-VL-4B）
```

---

## 快速开始

```bash
# 全流程处理（默认中英双语打标）
python pipeline.py E:/images --full

# 非交互式批处理
python pipeline.py E:/images --full --yes

# 仅打标 - Danbooru 英文标签
python pipeline.py E:/images --tag --tag-mode danbooru

# 仅打标 - 中英 Danbooru 标签
python pipeline.py E:/images --tag --tag-mode danbooru_bilingual

# 仅打标 - 纯中文 / 纯英文
python pipeline.py E:/images --tag --tag-mode chinese_only
python pipeline.py E:/images --tag --tag-mode english_only

# 仅预处理（不打标）
python pipeline.py E:/images --preprocess

# 自定义步骤
python pipeline.py E:/images --steps 2,3,6
```

---

## 打标模式

| `--tag-mode` | 输出格式 | 示例 |
|-------------|---------|------|
| `bilingual` *(默认)* | 一段中文 + 一段英文 | `角色头戴红色帽子…\n\nThe character wears a red hat...` |
| `chinese_only` | 纯中文描述 | `角色头戴红色帽子…` |
| `english_only` | 纯英文描述 | `The character wears a red hat...` |
| `natural` | 自然语言 | 自由格式，不受双语限制 |
| `danbooru` | 英文逗号标签 | `male, red_hair, blue_eyes, armor, sword, standing` |
| `danbooru_bilingual` | 英文标签行 + 中文标签行 | 中英文标签各一行 |

---

## 输出目录结构

```
目标文件夹/
├── xxx.png / xxx.jpg / xxx.webp      ← 原始文件，始终不变
├── 白底/
│   ├── xxx__png.jpg / xxx__jpg.jpg   ← 统一白底 JPG
│   ├── .masks/                       ← RMBG/alpha mask（调试用，不被自动删除）
│   └── 裁剪后/
│       ├── xxx__png.jpg              ← alpha 感知裁剪 + 16px padding
│       └── 方形/
│           ├── xxx__png.jpg          ← 1:1 白色正方形
│           └── HD/
│               ├── xxx__png.jpg      ← 1024×1024
│               └── xxx__png.txt      ← 标注文件（与 jpg 同 stem）
```

---

## 配置说明

编辑 `config.json`：

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `quality` | 95 | JPG 保存质量 |
| `target_size` | 1024 | 最终输出分辨率 |
| `crop_padding` | 16 | 裁剪安全边距 (px) |
| `tag_mode` | `bilingual` | 打标模式 |

### background 背景处理

```json
"background": {
  "mode": "rule_based_rmbg",    // none / dark_only / rule_based_rmbg / rmbg_all
  "output_bg": "#ffffff",       // 合成底色
  "white_threshold": 245,       // 白底判定 RGB 阈值
  "white_ratio_threshold": 0.85,// 边缘白色占比阈值
  "border_ratio": 0.05,         // 检测边缘宽度
  "fallback": "original"        // RMBG 不可用时降级方案
}
```

### RMBG-2.0 抠图

```json
"rmbg": {
  "enabled": true,
  "model_path": "E:/lora-img-prep-skill/required_model/RMBG-2.0"
}
```

- `enabled: false` → 禁用抠图，回退阈值翻白（无 CUDA 环境）
- `model_path` → 本地模型目录

### llama-server

```json
"llama_server": {
  "path": "E:/.../llama-server.exe",
  "host": "127.0.0.1",
  "port": 8080,
  "model": "E:/.../Qwen3-VL-4B-Instruct-Q6_K.gguf",
  "mmproj": "E:/.../mmproj-BF16.gguf",
  "auto_stop": true
}
```

---

## 依赖安装

```bash
pip install Pillow numpy requests torch torchvision realesrgan basicsr
pip install tqdm transformers kornia safetensors
```

**需要：**
- NVIDIA GPU (CUDA) — AI 放大和 RMBG 抠图
- [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server.exe` — VLM 打标
- [Qwen3-VL-4B-Instruct-Q6_K.gguf](https://www.modelscope.cn/models/unsloth/Qwen3-VL-4B-Instruct-GGUF/file/view/master) GGUF 模型
- [RealESRGAN x4plus](https://github.com/xinntao/Real-ESRGAN) 模型
- [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)](https://www.modelscope.cn/models/AI-ModelScope/RMBG-2.0/summary) BiRefNet 模型

---

## 鲁棒性

- **原子写入** — `.tmp` + `os.replace()`，崩溃不留半成品
- **逐文件断点续传** — 缺哪个补哪个，100 张中断 99 张不丢
- **参数变更自动检测** — manifest SHA256 哈希，修改 config 自动重跑受影响步骤
- **算法版本追踪** — 代码逻辑变更自动触发重跑
- **删除确认** — 任何删除操作暂停询问（`--yes` 跳过）
- **非交互式安全** — 管道环境下自动全流程，不卡 `input()`
- **GBK 安全输出** — Windows 控制台编码兼容

---

## 硬件要求

| 组件 | 最低要求 |
|------|---------|
| GPU 显存 | 12 GB (RTX 3060) — RMBG + RealESRGAN 分阶段使用 |
| 系统内存 | 16 GB |
| 磁盘 | 模型文件约 5 GB |

---

## License

MIT
