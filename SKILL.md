---
name: lora-img-prep
description: LoRA training dataset preprocessing pipeline — batch convert images (PNG/JPG/WebP) to white-background JPG (real-alpha composite / white-bg detect / RMBG-2.0 for complex backgrounds), crop with mask priority, center on square canvas, AI upscale to 1024x1024, and multi-mode VLM auto-tagging (bilingual / Chinese-only / English-only / natural / Danbooru tags).
---

# lora-img-prep v5.3.1

LoRA 训练数据集预处理流水线：格式筛查(PNG/JPG/WebP) → 白底JPG（真透明alpha合成 / 白底直转 / 其他RMBG-2.0抠图四路策略） → 裁剪白边(mask优先+alpha+RGB fallback) → 居中1:1正方形 → RealESRGAN AI放大至1024x1024 → VLM多模式打标（Qwen3-VL-4B：中英双语/纯中文/纯英文/自然语言/Danbooru标签/中英Danbooru标签）

## 快速开始

```bash
# 全流程处理（默认中英双语）
python "C:/Users/A/.claude/skills/lora-img-prep/pipeline.py" E:/打标处理/image-test4 --full

# 非交互式批处理
python "C:/Users/A/.claude/skills/lora-img-prep/pipeline.py" E:/打标处理/image-test4 --full --yes

# 仅打标 - Danbooru 英文标签
python "C:/Users/A/.claude/skills/lora-img-prep/pipeline.py" E:/打标/potato --tag --tag-mode danbooru

# 仅打标 - 中英 Danbooru 标签
python "C:/Users/A/.claude/skills/lora-img-prep/pipeline.py" E:/打标/potato --tag --tag-mode danbooru_bilingual

# 仅打标 - 纯中文/纯英文
python "C:/Users/A/.claude/skills/lora-img-prep/pipeline.py" E:/打标/potato --tag --tag-mode chinese_only
python "C:/Users/A/.claude/skills/lora-img-prep/pipeline.py" E:/打标/potato --tag --tag-mode english_only

# 仅预处理（白底+裁剪+方形+放大，不打标）
python "C:/Users/A/.claude/skills/lora-img-prep/pipeline.py" E:/打标处理/image-test4 --preprocess

# 自定义步骤
python "C:/Users/A/.claude/skills/lora-img-prep/pipeline.py" E:/打标处理/image-test4 --steps 2,3,6
```

不传步骤选项时进入**交互式选择**模式，显示菜单让用户选择步骤。传入 `--yes` 可跳过所有删除确认，适合非交互式批处理。

**处理完毕后自动终止 llama-server.exe**，释放 GPU 显存和内存，无需手动清理。

自动断点续传，重复执行跳过已完成步骤。

## 核心准则

**永远不能覆盖原图。** 所有结果写入原文件夹下的子目录，原图不变。

## 输出目录结构

```
目标文件夹/
├── xxx.png / xxx.jpg / xxx.webp      (原始文件，始终不变)
├── 白底/
│   ├── xxx__png.jpg / xxx__jpg.jpg   (统一白底JPG，保留原始扩展名编码防同名覆盖)
│   └── 裁剪后/
│       ├── xxx__png.jpg              (alpha感知裁剪+16px padding)
│       └── 方形/
│           ├── xxx__png.jpg          (1:1白色正方形)
│           └── HD/                   (AI放大至1024x1024)
│               ├── xxx__png.jpg      (1024x1024，始终保证此尺寸)
│               └── xxx__png.txt      (双语标注文件，与jpg同stem)
```

> 文件名规则：`{原文件名}__{原始扩展名}.{输出扩展名}`，例如 `abc.png` → `abc__png.jpg`，确保 `abc.png` 和 `abc.jpg` 不会互相覆盖。

## 输入格式支持

| 格式 | 透明通道 | 白底策略 |
|------|---------|---------|
| **PNG** | 真透明 (alpha范围含0~255) | **alpha合成白底**，简单直接，不做额外检测 |
| **PNG** | 假透明 (alpha全满) | 转RGB → 走黑底检测(RMBG/阈值翻白) |
| **PNG** | 无透明 (RGB/P) | 走黑底检测 → RMBG抠图/阈值翻白/原样 |
| **JPG/JPEG** | 无 | 走黑底检测 → RMBG抠图(优先)/阈值翻白(降级)/原样 |
| **WebP** | 真透明 (RGBA) | alpha合成白底，完事 |
| **WebP** | 假透明 (alpha全满) | 转RGB → 走黑底检测（修复：v3.1前直接跳过会漏掉黑底WebP） |
| **WebP** | 无透明 (RGB) | 走黑底检测 → RMBG抠图/阈值翻白/原样 |
| WEBM/MP4/GIF | - | 自动跳过 |

格式不区分大小写。所有输入统一输出 1024×1024 JPG + 双语 txt。


## 配置说明

编辑 [config.json](config.json) 可调整：

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `quality` | 95 | JPG保存质量 |
| `threshold` | 245 | 裁边白阈值（0-255，越小越严格） |
| `black_threshold` | 40 | 黑底检测阈值（边缘环带暗色像素的亮度上限） |
| `target_size` | 1024 | 放大目标分辨率 |
| `crop_padding` | 16 | 裁剪后安全边距（px） |
| `vlm_max_tokens` | 400 | VLM最大生成长度 |
| `vlm_temperature` | 0.1 | VLM推理温度 |
| `vlm_presize_max` | 512 | 打标前预缩放最长边 |
| `tag_mode` | `bilingual` | 打标模式，见上方模式表 |

### background 背景处理配置

```json
{
  "background": {
    "mode": "rule_based_rmbg",
    "output_bg": "#ffffff",
    "white_threshold": 245,
    "white_ratio_threshold": 0.85,
    "border_ratio": 0.05,
    "fallback": "original"
  }
}
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mode` | `rule_based_rmbg` | 背景模式：`none`/`dark_only`/`rule_based_rmbg`/`rmbg_all` |
| `output_bg` | `#ffffff` | 最终合成背景色（支持 `#ffffff`、`white`、`#f5f5f5` 等） |
| `white_threshold` | 245 | 白底检测 RGB 通道阈值 |
| `white_ratio_threshold` | 0.85 | 边缘白色占比阈值 |
| `border_ratio` | 0.05 | 白底检测边缘环带宽度 |
| `fallback` | `original` | RMBG 不可用时的降级：`original`/`dark_threshold` |

### llama-server 配置

详见 config.json 中 `llama_server` 块。所有性能参数（batch_size、flash_attn、cache_type_k/v 等）已验证适用于 RTX 3060 12GB + Qwen3-VL-4B。

- `host` 默认为 `127.0.0.1`，仅监听本地回环
- `auto_stop` 默认为 `true`，处理完毕后默认自动终止本脚本启动的 llama-server，以释放显存。如果检测到用户已有服务，则不会主动关闭。可通过 llama_server.auto_stop=false 保留本脚本启动的服务。
- `flash_attn` 从 config 读取，修改即生效（不再硬编码）

### RealESRGAN 镜像源

多镜像轮询下载，可按需添加国内加速镜像。支持 `sha256` 校验防止下载损坏或镜像污染。

### RMBG-2.0 抠图配置

```json
{
  "rmbg": {
    "enabled": true,
    "model_path": "E:/lora-img-prep-skill/required_model/RMBG-2.0",
    "batch_size": 4
  }
}
```

- `enabled: false` → 禁用RMBG，回退到阈值翻白，适合无CUDA环境
- `model_path` → 本地模型目录（含 `birefnet.py`、`model.safetensor`、`config.json`）
- `batch_size` → 预留字段，当前版本逐张处理，暂未使用批处理

## 步骤选择（节省时间 + 资源）

默认全流程执行（格式筛查 → 白底 → 裁剪 → 方形 → 放大 → 打标）。支持命令行选项自定义：

| 选项 | 执行步骤 | 适用场景 |
|------|---------|---------|
| `--full` | 1-6 (全部) | 从头处理一批图片 |
| `--tag` | 6 (仅打标) | 图片已是白底 JPG，只需 VLM 标注 |
| `--preprocess` | 1-5 (预处理) | 只需白底+裁剪+放大，不要标注 |
| `--tag-mode M` | 自定义打标模式 | `bilingual`/`chinese_only`/`english_only`/`natural`/`danbooru`/`danbooru_bilingual` |
| `--yes` / `--force` | 跳过确认 | 自动确认所有删除操作，适合非交互式批处理 |
| (无步骤选项) | 交互式菜单 | 显示步骤列表让用户输入选择 |

**链式依赖自动补全：** 选择步骤时自动补全前置依赖。例如 `--steps 5` 自动补上 4(AI放大)、3(裁剪)、2(白底)，`--steps 6` 独立运行不补任何步骤。方便快捷，无需手动罗列依赖。

**非交互式安全：** 在管道/批处理等非交互式环境（`sys.stdin.isatty()=False`）中自动默认全流程，不会卡在 `input()` 等待。

**交互式菜单示例：**

```
请选择要执行的步骤（可多选）:
--------------------------------------------------
  [1] 图片格式筛查
  [2] 白底JPG (透明合成+黑底翻白)
  [3] 裁剪白边 (alpha感知+padding)
  [4] 居中1:1正方形
  [5] AI放大至1024x1024
  [6] VLM双语打标
--------------------------------------------------
  输入格式: 逗号分隔如 1,2,3 或范围如 1-6
  快捷: a=全流程   p=预处理(1-5)   t=仅打标(6)
--------------------------------------------------
  >>>
```

> 提示：`--tag` 模式图片目录只需包含 JPG（或 PNG），系统会自动识别并跳过预处理输出目录。

### 第2步：白底JPG（智能四路策略）

```
输入图片
  ├─ 有真透明 alpha 通道？（alpha 至少 0.1% 像素 < 250）
  │   └─ 是 → alpha 合成白底 JPG，并保存 mask 用于裁剪
  └─ 否 / 假透明转 RGB
      ├─ 已是白底图？（边缘环带白色≥85% + 亮度≥240）
      │   └─ 是 → 直接统一转白底 JPG
      └─ 否 → background.mode 分支：
          ├─ none       → 不处理
          ├─ dark_only  → 只处理黑底 (RMBG优先/阈值翻白降级)
          ├─ rule_based_rmbg (默认) → RMBG-2.0 抠图合成白底
          └─ rmbg_all   → 全部 RMBG（含白底图）
```

### 第3步：裁剪白边（mask优先 + alpha + RGB fallback）

裁剪优先级：
1. `白底/.masks/xxx__png_mask.png`（RMBG/alpha mask，最可靠）
2. 原图 alpha 通道（PNG/WebP）
3. RGB 白色阈值 fallback

### 第6步：多模式打标

| `tag_mode` | 输出形式 | 校验规则 | 适用场景 |
|-----------|---------|---------|---------|
| `bilingual` (默认) | 一段中文 + 一段英文 | 中文≥10字 + 英文≥6词 | 默认 LoRA 描述 |
| `chinese_only` | 纯中文描述 | 中文≥10字 | 中文训练集 |
| `english_only` | 纯英文描述 | 英文≥6词 | 常规英文 caption |
| `natural` | 自然语言（不限格式） | 不为空 | 快速自然描述 |
| `danbooru` | 英文逗号标签 (`male, red_hair, ...`) | ≥5 个英文标签 | 标签式训练 |
| `danbooru_bilingual` | 英文标签行 + 中文标签行 | ≥5 中英标签各 | 双语标签存档 |

**校验失败处理：**
- `bilingual` 模式：缺英文直接用 VLM 翻译中文文本
- `danbooru_bilingual` 模式：补跑1轮 → 仍失败则 VLM 翻译英文标签为中文标签
- 其他模式：告警列出失败文件

优先从原图（PNG/WebP）获取 alpha 通道定位主体，找不到回退 RGB 阈值检测。裁剪后加 padding 防贴边。

### 第6步：VLM 段落级双语校验

```
校验通过条件：
  1. 分段数 ≥ 2（兼容空行分隔和单换行分隔）
  2. 第一段中文字符 ≥ 10
  3. 第二段英文单词(≥2字母) ≥ 6
  4. 无禁用前缀（"中文："、"English:"等）
  5. 无风格词（"动漫"、"cartoon"等）
  6. 无主观词（"可爱"、"beautiful"等）
```

补救机制：一轮打标后检查不合格 txt。若内容包含有效中文描述但缺少英文，
则调用 VLM 文本翻译接口，将中文描述翻译为英文并合并为双语格式。
若翻译后仍不合格，则记录告警，交由用户手动检查。

## 鲁棒性设计

### 原子写入
每步保存先用 `.tmp` 临时文件，写入成功再 `os.replace()` 覆盖正式文件。崩溃不会留下半成品。

### 有效性检查
- JPG: `Image.open(fp).verify()`
- TXT: `os.path.getsize(fp) >= 20`

### GBK 安全输出
```python
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def eprint(msg):
    text = str(msg)
    enc = sys.stdout.encoding or "utf-8"
    safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe, flush=True)
```

### VLM 响应兼容
兼容 `message.content`、`delta.content`、`text` 三种响应格式，同时检测 `error` 字段。

### 缺 txt 补漏
一轮打标后自动检查 jpg→txt 对应关系，丢失文件用更强重试（3次×5秒）补打。

## 依赖

```bash
pip install Pillow numpy requests torch torchvision realesrgan basicsr
pip install tqdm transformers kornia safetensors    # RMBG用
```

需要 NVIDIA GPU（CUDA）执行 AI 放大和 RMBG 抠图。RMBG 可在 `config.json` 中关闭。

## 依赖间关系

```
lora-img-prep/
├── SKILL.md          ← v5.3.1 完整文档
├── SKILL.v5.3.md.bak  ← v5.3 备份
```

## 常见问题

**Q: 模型下载失败？**
A: 在 config.json 的 `realesrgan.download_mirrors` 中添加可用的国内镜像。

**Q: 想重跑某一步？**
A: 删除该步骤及之后的所有输出文件夹。例如重新裁剪：删除 `裁剪后/` 目录再运行。

**Q: 只用打标，跳过前5步？**
A: 使用 `--tag` 选项：`python pipeline.py <目录> --tag`。图片可以是 JPG / PNG / WebP。系统会自动识别支持格式并生成对应 txt。
如果是预处理后的 HD 目录，通常为 JPG。

**Q: RMBG 抠图效果不好怎么办？**
A: 在 config.json 中设 `"enabled": false`，回退到阈值翻白方案。

**Q: 如何在脚本/管道中自动运行而不被确认提示卡住？**
A: 使用 `--yes` 或 `--force` 参数跳过所有删除确认：`python pipeline.py <目录> --full --yes`

## Manifest 参数哈希（自动检测配置变更）

v3.1 新增 `_ensure_step()` 系统，每个步骤输出目录保留 `.manifest.json`（或 `.manifest_tag.json`），记录该步骤关心的配置参数子集哈希。

### 参数分组

| 步骤 | 检测的参数 | manifest 位置 |
|------|-----------|-------------|
| 白底JPG | algorithm_version, quality, threshold, black_threshold, rmbg_enabled, rmbg_model_path, bg_mode, bg_white_threshold, bg_white_ratio, bg_border_ratio, bg_output_bg, bg_fallback | `白底/.manifest.json` |
| 裁剪白边 | algorithm_version, threshold, crop_padding | `裁剪后/.manifest.json` |
| 1:1方形 | quality, target_size | `方形/.manifest.json` |
| AI放大 | quality, target_size, realesrgan.model_path | `HD/.manifest.json` |
| VLM打标 | algorithm_version, vlm_max_tokens, vlm_temperature, vlm_presize_max, vlm_prompt, system_prompt, model, mmproj, tag_mode | `HD/.manifest_tag.json` |

### 行为

- 运行前检查对应 `.manifest.json`，计算当前配置的 SHA256 哈希与存储值对比
- 如果哈希不同 → 打印 `[参数变更]`，**暂停并询问用户确认后**再删除旧结果重跑
- 如果哈希相同 → 不清空目录，由步骤内部逐文件检查。
- 已存在且有效的输出会跳过，缺失或损坏的文件会自动补跑。
- 修改指定步骤的参数后，不需要手动删除文件夹，代码自动检测并重跑

> 注意：首次确认后同一会话不再重复询问。VLM 参数变更是独立的（.manifest_tag.json）。



## 资源管理（显存策略）

由于 RTX 3060 12GB 显存限制，三步 GPU 模型分阶段使用以防 OOM：

| 阶段 | 步骤 | GPU 模型 | 显存策略 |
|------|------|---------|---------|
| 预处理 | 第2步（白底） | RMBG-2.0 (BiRefNet) | 完成后卸载模型 + `clear_cuda()` |
| 放大 | 第5步（AI放大） | RealESRGAN x4plus | 完成后卸载 + `clear_cuda()` |
| 打标 | 第6步（VLM） | Qwen3-VL-4B (llama-server) | 常驻显存，独立于前两阶段 |

```python
def clear_cuda():
    """释放CUDA显存"""
    import gc
    gc.collect()
    if _HAS_CUDA:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
```

RMBG 和 RealESRGAN 不会同时加载，llama-server 在两者完全卸载后才被调用。可通过 `config.json` 中 `"rmbg.enabled": false` 完全禁用 RMBG 避免显存压力。

## 全版本改动汇总

### v1 → v2（框架搭建）
- numpy 矢量裁剪替代逐像素循环（10-50x加速）
- 配置外置到 config.json
- RealESRGAN 多镜像自动下载
- llama-server 自动检测/启动
- 断点续传：每步检查输出文件夹
- tqdm 进度条

### v2 → v2.1（GPT5.5审查修复）
- **原子写入**：临时文件 `.tmp` + `os.replace`，防崩溃留下半成品
- **有效性检查**：JPG 用 `im.verify()`，TXT 检查大小≥20byte
- **GBK 真实修复**：`sys.stdout.reconfigure(errors="replace")` + encode/decode 双重保险
- **alpha感知裁剪**：优先用原 PNG/WebP 的 alpha 通道定位主体，回退 RGB 阈值
- **裁剪 padding**：16px 安全边距，防主体贴边
- **段落级双语校验**：检查段落数≥2、中文字符≥10、英文单词≥6、禁用风格/主观词
- **system prompt 加强**：首次就精确约束中英两段格式，大幅降低补跑率
- **strip thinking 标签**：正则清除 `<think>...</think>` 污染
- **VLM error 检测**：解析 content 前先检查 `error` 字段

### v2.1 → v3（多格式 + 抠图集成）
- **多格式支持**：`classify_files()` 识别 `.png/.jpg/.jpeg/.webp`，其余自动跳过
- **PNG P mode 透明修复**：检测 `P + transparency` 正确转为 RGBA 再合成白底（原为黑底bug）
- **JPG 黑底翻白**：`_has_dark_background()` 检测边缘环带暗色占比和平均亮度，判定深底图
- **RMBG-2.0 抠图**：惰性加载 BiRefNet 模型，仅首次遇到黑底图才加载；一次性常驻显存后续复用
- **真/假透明检测**：读取 alpha 通道范围，区分真实透明与假透明（alpha全满）
- **透明PNG简化**：真透明 PNG 直接 alpha 合成白底，不做额外检测，消除误检
- **`_bg_removal()` 智能路由**：检测到黑底→RMBG(开启+CUDA) → 阈值翻白(降级)；否则原样

### v3.1（第二次GPT5.5审查修复）
- **同名防止覆盖**：`_output_stem()` 编码原始扩展名到输出文件名，`abc.png`→`abc__png.jpg`，`abc.jpg`→`abc__jpg.jpg`，彻底解决多格式同名覆盖问题
- **`_decode_stem()` 原图查找**：裁剪步骤从 `abc__png.jpg` 解码回原始 stem + 扩展名，正确查找原 PNG/WebP 的 alpha 信息
- **边缘环带黑底检测**：替代四角色块检测，取上下左右各 5% 宽度的边缘环带，计算暗色像素占比和平均亮度。不易被角落的黑色装饰/光影误判，也不容易漏判带有浅色边缘的暗底图
- **WebP 假透明走黑底检测**：v3 中假透明 WebP 直接转 RGB 不做任何处理，可能漏掉黑底图。v3.1 改为转 RGB 后走 `_bg_removal`，正确检测并翻白
- **TXT 跳过内容校验**：断点续传时不仅检查 txt 是否存在，还检查 `validate_bilingual()` 是否通过，不合格的自动补跑
- **`atomic_save_jpg` format 显式指定**：`format="JPEG"` 避免 `.tmp` 后缀导致 PIL 无法推断格式
- **`is_valid_jpg` reopen + load**：`verify()` 之后再次 `open().load()` 确认可完整解码
- **删除确认**：任何删除旧结果的操作（参数变更/续传清理）都会暂停并询问用户，输入 `yes` 确认、`skip` 跳过或 `Ctrl+C` 终止。同一次会话首次确认后不再重复询问。


### v3.1 改动汇总

- **参数哈希 Manifest**：每步输出目录 `.manifest.json` 记录配置子集哈希，修改参数后自动检测并重跑受影响步骤
- **tag manifest 隔离**：使用 `.manifest_tag.json` 避免与 HD 步骤的 manifest 冲突
- **删除确认**：任何删除旧结果的操作（参数变更/续传清理）都会暂停并询问用户确认，防误删

### v3.2 改动汇总

- **步骤选择系统**：支持 `--full`、`--preprocess`、`--tag`、`--steps N,M` 四种模式，不传参数时进入交互式选择菜单
- **仅打标模式**：`--tag` 跳过所有预处理步骤，直接查找 JPG 目录进行 VLM 打标，适合图片已完工只需标注的场景
- **步骤依赖检查**：自动检查前置步骤是否包含，不满足时报错退出
- **智能输出目录查找**：汇总时自动定位到最深的输出目录

### v4 改动汇总

- **链式依赖自动补全**：`STEP_DIRECT_DEPS` 仅定义直接前置关系，`_resolve_step_deps()` 自动补全全链。选 `--steps 5` 自动补 4,3,2；选 `--steps 6` 独立不补
- **非交互式环境检测**：`sys.stdin.isatty()` 判断，管道/批处理环境自动默认全流程，不会卡死在 `input()`
- **自动清理日志增强**：会话内首次确认后不再弹提示，但打印 `[自动清理]` 日志让用户知晓被删除的内容
- **版本升到 v4**

### v4.1 改动汇总

- **修复 final_dir 覆盖 bug**：汇总报告中 `final_dir = d` 用循环残余变量覆盖正确值，导致最终输出路径指向错误目录（P0）
- **修复 alpha 裁剪 fallback 覆盖 bug**：`continue` 只跳出内层 `for try_ext` 循环，未跳过文件级 RGB fallback，导致 alpha 感知裁剪结果被覆盖（P0）
- **提取 `_alpha_crop_from_original()` 复用函数**：消除 step3_crop 中 ~25 行重复代码（P2）
- **修复空目录误弹删除确认**：step6_tag 中 `os.makedirs` 先于 manifest 检查执行，导致刚创建的空白目录也弹出"需要删除"提示（P1）
- **备份 v4 源文件**：`pipeline.v4.py.bak`、`config.v4.json.bak`、`SKILL.v4.md.bak`

### v5.0 改动汇总（GPT5.5 审查修复）

**P0（必须修）：**
- **逐文件断点续传**：`_ensure_step()` 不再因单个有效文件跳过整批。每步内部逐文件检查，缺哪个补哪个，100张中断99张不丢
- **HD 输出强制 1024x1024**：超大图不再直接复制，统一 resize 到 `TARGET_SIZE × TARGET_SIZE`，保证输出一致性
- **TXT 文件名修复**：`abc__png.jpg` → `abc__png.txt`（不再二次编码为 `abc__png__jpg.txt`），与 jpg 同 stem
- **`stop_llama_server()` 只调用一次**：module-level `try/finally` 统一清理，不再双重执行
- **`--tag` 目录检测支持全格式**：`main()` 中检查所有 `SUPPORTED_IMG_EXTS`，PNG/WebP 目录也能正确识别

**P1（强烈建议修）：**
- **只关闭自己启动的 llama-server**：`_LLAMA_STARTED_BY_US` 标记，不再 `taskkill` 全局进程杀用户已有服务
- **mmproj 文件存在性检查**：`start_llama_server()` 中增加 `os.path.isfile(mmproj_path)` 检查，提前报错而非等超时
- **`host` 改为 `127.0.0.1`**：安全加固，仅监听本地回环
- **`flash_attn` 从 config 读取**：不再硬编码 `"on"`，config 修改即生效
- **`auto_stop` 配置**：新增 `llama_server.auto_stop`，设为 `false` 可保留 llama-server 不自动清理

**P2（发布前优化）：**
- **相对路径解析**：`resolve_path()` 支持相对路径（相对于 skill 目录），config 中路径可写 `./required_model/...`
- **VLM 中文禁用词匹配修复**：中文用子串匹配（`\b` 对中文不可靠），英文用 `\b` 词边界
- **RealESRGAN SHA256 校验**：下载后验证文件哈希，防止镜像污染或下载损坏
- **`--yes` / `--force` 参数**：跳过所有删除确认，适合无人值守批处理
- **Manifest 参数分组精简**：`white_bg` 不再包含 `target_size`/`crop_padding`，减少无关参数变更触发重跑
- **文档同步**：修正"不传参数"表述、依赖关系图、temperature 递减等与代码不一致处

### v5.1 改动汇总

- **llama-server 健康检查**：`check_llama_server()` 改用 `/health` 端点检查 `{"status":"ok"}`，避免模型未加载完就发请求导致 503
- **llama-server 启动优化**：去掉 bat 包装，直接 `subprocess.Popen(cmd)`，超时从 30s 延长到 60s，第 5 秒输出"模型仍在加载中"提示
- **`stop_llama_server()` 简化**：有了真正的 Popen 对象后直接用 `terminate()`/`kill()`，不再需要 taskkill 降级分支
- **`_verify_and_retry()` 翻译补救**：bilingual 模式缺英文时不再多轮补跑浪费 GPU，直接用 VLM 翻译中文文本为英文
- **变量名修复**：`jpg_count` → `img_count`、`missing_txt` 未定义
- **输出目录检测**：`main()` 中查找 JPG 改为检查所有 `SUPPORTED_IMG_EXTS`
- **备份**：`pipeline.v5.1.py.bak`、`config.v5.1.json.bak`、`SKILL.v5.1.md.bak`

### v5.2 改动汇总

- **背景处理重构**：从"黑底特化"升级为通用四路决策树
  - `_is_real_alpha()` — 真实透明判断（比例而非单点阈值）
  - `_has_white_background()` — 边缘环带 RGB+亮度双重校验
  - `process_background()` — 真透明→alpha合成 | 假透明→转RGB | 白底→直转 | 其他→RMBG
- **RMBG mask 保存**：`_remove_background_rmbg()` 返回 `(result, mask_numpy)`，mask 保存至 `白底/.masks/`
- **裁剪升级**：新增 `_crop_from_mask()` + `get_bbox_from_mask()`，裁剪优先级：mask > alpha > RGB fallback
- **新增 `background` 配置节**：`mode`/`output_bg`/`white_threshold`/`white_ratio_threshold`/`border_ratio`/`fallback`
- **支持复杂背景**：红底/蓝底/噪点底/渐变底等非白非黑背景自动 RMBG 抠图
- **alpha 合成后二次检验**：合成白底后检测边缘是否真的白了，不白则自动走 RMBG（已废弃于 v5.3.1）
- **备份**：`pipeline.v5.2.py.bak`、`config.v5.2.json.bak`、`SKILL.v5.2.md.bak`

### v5.3 改动汇总

- **多模式打标系统**：`TAG_MODE_SPECS` 定义 6 种模式，每种独立 system prompt + user prompt + 校验规则
  - `bilingual` — 中英双语段落
  - `chinese_only` — 纯中文
  - `english_only` — 纯英文
  - `natural` — 自然语言
  - `danbooru` — 英文 Danbooru 标签
  - `danbooru_bilingual` — 中英双语标签
- **CLI `--tag-mode`**：`python pipeline.py <目录> --tag --tag-mode danbooru`
- **模式感知校验**：`validate_tag(text, mode)` 替代固定 `validate_bilingual()`，每种模式独立规则
- **`danbooru_bilingual` 补跑+翻译降级**：补跑 1 轮 → 仍失败则 VLM 翻译英文标签为中文标签
- **`_find_image_for_txt()`**：适配 `abc__png.txt` → `abc__png.jpg` 的映射
- **Manifest 含 `tag_mode`**：切换模式时自动清空旧 txt 重打
- **config.json 新增 `tag_mode`**：默认 `"bilingual"`
- **备份**：`pipeline.v5.3.py.bak`、`config.v5.3.json.bak`、`SKILL.v5.3.md.bak`
