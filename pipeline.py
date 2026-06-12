#!/usr/bin/env python3
"""
批量图片处理流水线 v5.3.1
PNG → 白底JPG → 裁剪白边(alpha感知+padding) → 1:1方形 → AI放大1024 → VLM多模式打标

v5.3.1 改动（GPT5.5审查修复）：
  - mask_dir 路径修正（step3 从白底/.masks 读 mask，不再错查目标/.masks）
  - 新增 get_bbox_from_mask() 替代 get_bbox_from_array() 处理 mask (亮=主体，暗=背景)
  - RMBG mask 保存修复（不再乘255两次导致 mask 破坏）
  - 真透明 alpha 直接合成白底，不再用 _has_white_background(bg) 二次判断
  - step6_tag 不再传固定 VLM_MAX_TOKENS，让 TAG_SPEC.max_tokens 生效
  - danbooru_bilingual 补跑找图逻辑修复（适配 abc__png.txt 命名）
  - background.output_bg 真正生效（parse_bg_color + 全局 BG_COLOR）
  - background.mode 分支实现 (none/rule_based_rmbg/rmbg_all/dark_only)
  - .masks 目录不再自动删除，方便调试和续传
  - manifest 增加 border_ratio/output_bg/fallback/rmbg_model_path
"""

import sys, os, io, base64, time, json, subprocess, shutil, re, hashlib
from pathlib import Path

# ── 安全输出（防止GBK崩溃） ──
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def eprint(msg):
    text = str(msg)
    enc = sys.stdout.encoding or "utf-8"
    safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe, flush=True)

def log(step, msg):
    eprint(f"  [{step}] {msg}")

# ── 配置加载 ──
SKILL_DIR = Path(__file__).parent
DEFAULT_CONFIG = SKILL_DIR / "config.json"

def load_config(config_path=None):
    path = Path(config_path or DEFAULT_CONFIG)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print("  ! 未找到 config.json，使用内置默认值")
    return {}

CFG = load_config()

QUALITY = CFG.get("quality", 95)
THRESHOLD = CFG.get("threshold", 245)
BLACK_THRESHOLD = CFG.get("black_threshold", 40)
TARGET_SIZE = CFG.get("target_size", 1024)
CROP_PADDING = CFG.get("crop_padding", 16)
VLM_MAX_TOKENS = CFG.get("vlm_max_tokens", 400)
VLM_TEMP = CFG.get("vlm_temperature", 0.1)
VLM_PRESIZE = CFG.get("vlm_presize_max", 512)
VLM_PROMPT = CFG.get("vlm_prompt",
    '核心要求：用一段中文一段英文（只用输出中文和英文，'
    '不要出现"中文：" "English:"）的自然语言简要描述'
    '这张图片里的角色的外观，不要掺杂任何主观性的描述，'
    '禁止任何风格类的描述词出现。禁止出现："这个角色是谁"')
TAG_MODE = CFG.get("tag_mode", "bilingual")

LLAMA_CFG = CFG.get("llama_server", {})
ESRGAN_CFG = CFG.get("realesrgan", {})
RMBG_CFG = CFG.get("rmbg", {"enabled": False, "model_path": "E:/RMBG-2/RMBG-2.0"})
BG_CFG = CFG.get("background", {
    "mode": "rule_based_rmbg",
    "output_bg": "#ffffff",
    "white_threshold": 245,
    "white_ratio_threshold": 0.85,
    "border_ratio": 0.05,
    "fallback": "original"
})

def _parse_bg_color(value):
    """解析背景色：支持 'white'、'#ffffff'、'#f5f5f5' 等"""
    if not value:
        return (255, 255, 255)
    v = str(value).strip().lower()
    if v in ("white", "#fff", "#ffffff"):
        return (255, 255, 255)
    if v.startswith("#") and len(v) == 7:
        return tuple(int(v[i:i+2], 16) for i in (1, 3, 5))
    return (255, 255, 255)

BG_COLOR = _parse_bg_color(BG_CFG.get("output_bg", "#ffffff"))

# ── 依赖检测 ──
_MISSING_DEPS = []
try:
    import numpy as np
except ImportError:
    _MISSING_DEPS.append("numpy (pip install numpy)")
try:
    from PIL import Image
except ImportError:
    _MISSING_DEPS.append("Pillow (pip install Pillow)")
try:
    import requests
except ImportError:
    _MISSING_DEPS.append("requests (pip install requests)")

_HAS_TQDM = False
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    pass

_HAS_CUDA = False
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

# RMBG-2.0 全局缓存（惰性加载）
_RMBG_MODEL = None
_RMBG_TRANSFORM = None
_RMBG_DEVICE = None

# llama-server 进程追踪（只关闭自己启动的，不杀用户已有的）
_LLAMA_PROCESS = None
_LLAMA_STARTED_BY_US = False

# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════

def _output_stem(filename):
    """生成不重复的输出文件名stem：'abc.png' → 'abc__png'"""
    p = Path(filename)
    return f"{p.stem}__{p.suffix.lower().lstrip('.')}"

def _decode_stem(stem):
    """'abc__png' → ('abc', 'png')"""
    idx = stem.rfind('__')
    if idx == -1:
        return stem, ''
    return stem[:idx], stem[idx+2:]

def tag_txt_name(filename):
    """生成打标 txt 文件名：已处理的 JPG (含 __) 直接用 stem，否则用 _output_stem 防重名。
    abc__png.jpg → abc__png.txt (不再二次编码为 abc__png__jpg.txt)"""
    p = Path(filename)
    if p.suffix.lower() in {'.jpg', '.jpeg'} and '__' in p.stem:
        return p.stem + '.txt'
    return _output_stem(filename) + '.txt'

def resolve_path(p):
    """解析路径：绝对路径直接返回，相对路径相对于 SKILL_DIR 解析"""
    if not p:
        return p
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((SKILL_DIR / path).resolve())

def is_ascii_path(path):
    """检查路径是否只包含 ASCII 字符（防止 llama-server 中文路径问题）"""
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

def clear_cuda():
    """释放CUDA显存"""
    import gc
    gc.collect()
    if _HAS_CUDA:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def is_valid_jpg(path):
    """验证JPG文件是否完整可读"""
    try:
        with Image.open(path) as im:
            im.verify()
        # reopen + load 确认可解码
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False

def is_valid_txt(path, min_size=20):
    """验证txt文件是否存在且非空"""
    return os.path.isfile(path) and os.path.getsize(path) >= min_size

def atomic_save_jpg(img, path, quality=95):
    """原子写入JPG：先写.tmp再os.replace（显式format防PIL推断失败）"""
    tmp = path + ".tmp"
    img.save(tmp, format="JPEG", quality=quality)
    os.replace(tmp, path)

def atomic_write_text(content, path, encoding="utf-8"):
    """原子写入文本：先写.tmp再os.replace"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(content)
    os.replace(tmp, path)

def has_any_valid_output(path):
    """检查某步是否至少有一个输出文件（仅用于快速判断目录是否为空）。
    不再用于决定是否跳过整步——每步内部逐文件检查完整性。"""
    if not os.path.isdir(path):
        return False
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if f.lower().endswith('.jpg') and is_valid_jpg(fp):
            return True
        if f.endswith('.txt') and is_valid_txt(fp):
            return True
    return False

def iter_files(src_dir, exts=None):
    """返回该目录下指定扩展名集合的文件（排序后）。
    默认找 .jpg，传入多个扩展名（如 {'.jpg','.png','.webp'}）可同时匹配。
    """
    if exts is None:
        exts = {'.jpg'}
    return sorted(f for f in os.listdir(src_dir) if os.path.splitext(f)[1].lower() in exts)

def retry_call(fn, retries=3, delay=2, label=""):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt < retries - 1:
                log("重试", f"{label}: {e}，{delay}s后重试 ({attempt+1}/{retries})")
                time.sleep(delay)
            else:
                raise

# ── 算法版本（manifest 中检测，确保算法变更后旧结果自动重跑）──
_BG_ALGO_VERSION = "rule_based_rmbg_v2"
_CROP_ALGO_VERSION = "mask_bbox_v2"
_TAG_ALGO_VERSION = "tag_mode_v2"

# ════════════════════════════════════════════════════════
# 参数哈希 + Manifest（断点续传一致性检测）
# ════════════════════════════════════════════════════════

def _get_step_config(step_name):
    """返回某一步所关心的配置参数子集。
    不同步骤只检测与自己相关的参数，互不影响。
    """
    if step_name == "white_bg":
        return {
            "algorithm_version": _BG_ALGO_VERSION,
            "quality": QUALITY,
            "threshold": THRESHOLD,
            "black_threshold": BLACK_THRESHOLD,
            "rmbg_enabled": RMBG_CFG.get("enabled", False),
            "rmbg_model_path": RMBG_CFG.get("model_path", ""),
            "bg_mode": BG_CFG.get("mode", ""),
            "bg_white_threshold": BG_CFG.get("white_threshold", 245),
            "bg_white_ratio": BG_CFG.get("white_ratio_threshold", 0.85),
            "bg_border_ratio": BG_CFG.get("border_ratio", 0.05),
            "bg_output_bg": BG_CFG.get("output_bg", "#ffffff"),
            "bg_fallback": BG_CFG.get("fallback", "original"),
        }
    elif step_name == "crop":
        return {
            "algorithm_version": _CROP_ALGO_VERSION,
            "threshold": THRESHOLD,
            "crop_padding": CROP_PADDING,
        }
    elif step_name == "square":
        return {
            "quality": QUALITY,
            "target_size": TARGET_SIZE,
        }
    elif step_name == "upscale":
        return {
            "quality": QUALITY,
            "target_size": TARGET_SIZE,
            "model_path": ESRGAN_CFG.get("model_path", ""),
        }
    elif step_name == "tag":
        return {
            "algorithm_version": _TAG_ALGO_VERSION,
            "vlm_max_tokens": VLM_MAX_TOKENS,
            "vlm_temperature": VLM_TEMP,
            "vlm_presize_max": VLM_PRESIZE,
            "vlm_prompt": VLM_PROMPT,
            "system_prompt": SYSTEM_PROMPT_VLM,
            "model": LLAMA_CFG.get("model", ""),
            "mmproj": LLAMA_CFG.get("mmproj", ""),
            "tag_mode": TAG_MODE,
        }
    return {}

def _compute_config_hash(config_dict):
    """对配置子集计算 16 位 SHA256 哈希"""
    raw = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

def _check_manifest(step_dir, step_name):
    """检查 manifest 中的 config_hash 是否与当前配置一致。
    返回 True → 需要重跑；False → 可以跳过。
    tag 步骤使用单独的 .manifest_tag.json，避免与同目录的 HD manifest 冲突。
    """
    manifest_name = ".manifest_tag.json" if step_name == "tag" else ".manifest.json"
    manifest_path = os.path.join(step_dir, manifest_name)
    if not os.path.isfile(manifest_path):
        return True
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return True
    expected = _compute_config_hash(_get_step_config(step_name))
    return manifest.get("config_hash") != expected

def _write_manifest(step_dir, step_name):
    """写入 manifest 文件（tag 使用 .manifest_tag.json，避免与同目录的 HD manifest 冲突）"""
    manifest_name = ".manifest_tag.json" if step_name == "tag" else ".manifest.json"
    manifest = {
        "step": step_name,
        "version": "v5.3.1",
        "config_hash": _compute_config_hash(_get_step_config(step_name)),
    }
    with open(os.path.join(step_dir, manifest_name), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

# ── 删除确认（安全防护） ──
_DELETE_CONFIRMED = False
_FORCE_YES = False  # --yes/--force 参数跳过所有确认


def _confirm_deletion(prompt_msg):
    """请求用户确认删除操作。同一会话首次确认后不再重复询问。
    --yes/--force 模式下自动确认。"""
    global _DELETE_CONFIRMED
    if _FORCE_YES or _DELETE_CONFIRMED:
        if not _DELETE_CONFIRMED:
            eprint(f"  [--yes 自动确认] {prompt_msg.split('需要')[1] if '需要' in prompt_msg else prompt_msg.split('中有')[1] if '中有' in prompt_msg else ''}")
        else:
            eprint(f"  [自动清理] {prompt_msg.split('需要')[1] if '需要' in prompt_msg else prompt_msg.split('中有')[1] if '中有' in prompt_msg else ''}")
        return True
    eprint(f"\n  !!! {prompt_msg}")
    eprint("  !!! 输入 yes 确认删除，输入 skip 跳过该步骤，或按 Ctrl+C 终止")
    try:
        answer = input("  >>> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        eprint("\n  用户终止")
        sys.exit(1)
    if answer == "yes":
        _DELETE_CONFIRMED = True
        return True
    elif answer == "skip":
        eprint("  已跳过该步骤")
        return False
    else:
        eprint("  输入错误，默认跳过删除")
        return False

def _get_step_label(step_name):
    return {
        "white_bg": "白底JPG",
        "crop": "裁剪白边",
        "square": "1:1方形",
        "upscale": "AI放大",
        "tag": "VLM打标",
    }.get(step_name, step_name)

def _ensure_step(step_dir, step_name):
    """检查 manifest 参数变更 + 创建目录。
    始终返回 True（让调用方逐文件检查完整性），
    仅在用户拒绝删除参数变更的旧结果时返回 False。
    """
    if not os.path.isdir(step_dir):
        os.makedirs(step_dir, exist_ok=True)
        _write_manifest(step_dir, step_name)
        return True

    if _check_manifest(step_dir, step_name):
        label = _get_step_label(step_name)
        log("参数变更", f"{label}: 检测到参数修改，需删除旧结果重新处理")
        if not _confirm_deletion(f"检测到 [{label}] 的配置参数已变更，需要删除 {step_dir} 下的所有文件重新处理"):
            log("跳过", f"{label}: 用户选择跳过")
            return False
        shutil.rmtree(step_dir)
        os.makedirs(step_dir, exist_ok=True)
        _write_manifest(step_dir, step_name)
        return True

    # 目录存在、manifest 匹配 → 仍需逐文件检查，不整步跳过
    # 每个步骤函数内部会逐文件 skip 已完成的文件
    return True

# ════════════════════════════════════════════════════════
# 第1步：图片格式筛查
# ════════════════════════════════════════════════════════

SUPPORTED_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
SKIP_EXTS = {'.webm', '.mp4', '.avi', '.mov', '.gif', '.bmp', '.tiff'}

def classify_files(src_dir):
    """扫描目录，返回 (supported, skipped)，均为 (filename, ext)"""
    supported = []
    skipped = []
    for f in sorted(os.listdir(src_dir)):
        fp = os.path.join(src_dir, f)
        if not os.path.isfile(fp):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext in SUPPORTED_IMG_EXTS:
            supported.append((f, ext))
        else:
            skipped.append((f, ext))
    return supported, skipped

def step1_scan(src_dir):
    log("扫描", "筛选支持格式图片 (PNG/JPG/WebP)...")
    supported, skipped = classify_files(src_dir)

    skipped_names = [f for f, ext in skipped]
    if skipped_names:
        log("跳过", f"不支持格式 ({len(skipped_names)}个): {skipped_names}")

    if not supported:
        eprint("  ! 未找到任何支持的图片文件，退出")
        sys.exit(1)

    log("扫描", f"共发现 {len(supported)} 个图片文件:")
    ext_counts = {}
    for f, ext in supported:
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    for ext in sorted(ext_counts):
        log("扫描", f"  .{ext.lstrip('.')}: {ext_counts[ext]} 个")

    return supported

# ════════════════════════════════════════════════════════
# 第2步：统一转为白底JPG（支持PNG/JPG/WebP）
# ════════════════════════════════════════════════════════

def _has_dark_background(img, black_threshold=40, border_ratio=0.05, dark_ratio_threshold=0.6):
    """检测图片是否为暗色背景。
    取上下左右各 border_ratio 宽度的边缘环带，计算暗色像素占比和平均亮度。
    比单纯四角更鲁棒，不易被角落的黑色装饰/光影误判。
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    bw = max(1, int(w * border_ratio))
    bh = max(1, int(h * border_ratio))

    top = arr[:bh, :, :]
    bottom = arr[-bh:, :, :]
    left = arr[:, :bw, :]
    right = arr[:, -bw:, :]

    border = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)

    # 亮度公式 Y = 0.2126*R + 0.7152*G + 0.0722*B
    luma = 0.2126 * border[:, 0] + 0.7152 * border[:, 1] + 0.0722 * border[:, 2]
    dark_ratio = np.mean(luma < black_threshold)
    mean_luma = np.mean(luma)
    return dark_ratio >= dark_ratio_threshold and mean_luma < 80

def _remove_dark_background(img, black_threshold=40):
    """将黑底图片的主体分离并合成到白底上。"""
    arr = np.array(img.convert("RGB"))
    mask = np.any(arr > black_threshold, axis=2).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode="L")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img.convert("RGB"), mask=mask_img)
    return bg

def _init_rmbg():
    """惰性加载 RMBG-2.0 抠图模型"""
    global _RMBG_MODEL, _RMBG_TRANSFORM, _RMBG_DEVICE
    if _RMBG_MODEL is not None:
        return True

    if not RMBG_CFG.get("enabled", False):
        return False
    if not _HAS_CUDA:
        eprint("  ! RMBG 需要 CUDA，但不可用")
        return False

    model_path = resolve_path(RMBG_CFG.get("model_path", "E:/RMBG-2/RMBG-2.0"))
    model_path = os.path.abspath(model_path)
    if not os.path.isdir(model_path):
        eprint(f"  ! RMBG 模型目录不存在: {model_path}")
        return False

    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    log("RMBG", "加载 RMBG-2.0 抠图模型...")
    model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    device = 'cuda' if _HAS_CUDA else 'cpu'
    model.to(device)
    model.eval()

    _RMBG_MODEL = model
    _RMBG_DEVICE = device
    _RMBG_TRANSFORM = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    log("RMBG", "模型就绪")
    return True

def _is_real_alpha(alpha, threshold=250, min_ratio=0.001):
    """判断 alpha 通道是否为真实透明（至少 0.1% 像素接近透明）。
    比 alpha.min < N 更稳：不会因单个像素误判。"""
    if alpha.max() == 0:
        return False
    return np.mean(alpha < threshold) >= min_ratio


def _save_mask(mask_arr, mask_dir, out_stem):
    """统一保存 mask 到 .masks 目录，自动处理 uint8/float 转换"""
    if mask_dir is None or out_stem is None:
        return
    os.makedirs(mask_dir, exist_ok=True)
    arr = np.array(mask_arr)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    Image.fromarray(arr, mode="L").save(
        os.path.join(mask_dir, out_stem + "_mask.png"))


def _remove_background_rmbg(img):
    """检测图片是否已经是白底。
    边缘环带三重校验：RGB阈值 + 亮度 + 非白像素比例。
    配置可调：background.white_threshold / white_ratio_threshold / border_ratio。"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    wt = BG_CFG.get("white_threshold", 245)
    br = BG_CFG.get("border_ratio", 0.05)
    rt = BG_CFG.get("white_ratio_threshold", 0.85)
    bw = max(1, int(w * br))
    bh = max(1, int(h * br))

    border = np.concatenate([
        arr[:bh, :, :].reshape(-1, 3),
        arr[-bh:, :, :].reshape(-1, 3),
        arr[:, :bw, :].reshape(-1, 3),
        arr[:, -bw:, :].reshape(-1, 3)
    ], axis=0)

    # 1) 白色像素占比（RGB 三通道都 >= wt）
    is_white = np.all(border >= wt, axis=1)
    white_ratio = is_white.mean()

    # 2) 平均亮度
    luma = 0.2126 * border[:, 0] + 0.7152 * border[:, 1] + 0.0722 * border[:, 2]
    mean_luma = luma.mean()

    return white_ratio >= rt and mean_luma >= 240

def _remove_background_rmbg(img):
    """使用 RMBG-2.0 生成alpha遮罩，去除背景并合成白底。返回 (result_img, mask_numpy_float)"""
    from torchvision.transforms.functional import to_pil_image
    import torch
    input_tensor = _RMBG_TRANSFORM(img).unsqueeze(0).to(_RMBG_DEVICE)
    with torch.no_grad():
        preds = _RMBG_MODEL(input_tensor)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    mask_pil = to_pil_image(pred).resize(img.size, Image.LANCZOS)
    bg = Image.new("RGB", img.size, BG_COLOR)
    bg.paste(img.convert("RGB"), mask=mask_pil)
    return bg, np.array(mask_pil)


def _has_white_background(img):
    """检测图片是否已经是白底。
    边缘环带三重校验：RGB阈值 + 亮度。
    配置可调：background.white_threshold / white_ratio_threshold / border_ratio。"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    wt = BG_CFG.get("white_threshold", 245)
    br = BG_CFG.get("border_ratio", 0.05)
    rt = BG_CFG.get("white_ratio_threshold", 0.85)
    bw = max(1, int(w * br))
    bh = max(1, int(h * br))

    border = np.concatenate([
        arr[:bh, :, :].reshape(-1, 3),
        arr[-bh:, :, :].reshape(-1, 3),
        arr[:, :bw, :].reshape(-1, 3),
        arr[:, -bw:, :].reshape(-1, 3)
    ], axis=0)

    is_white = np.all(border >= wt, axis=1)
    white_ratio = is_white.mean()
    luma = 0.2126 * border[:, 0] + 0.7152 * border[:, 1] + 0.0722 * border[:, 2]
    mean_luma = luma.mean()

    return white_ratio >= rt and mean_luma >= 240


def process_background(img, out_stem=None, mask_dir=None):
    """
    统一背景处理决策树（支持 background.mode 分支）：
      none: 不处理
      dark_only: 旧黑底逻辑
      rule_based_rmbg: 真透明→alpha合成 | 假透明→转RGB | 白底→直转 | 其他→RMBG
      rmbg_all: 除真透明外全部RMBG
    返回 (result_rgb_img, mask_array_or_None)
    """
    bg_mode = BG_CFG.get("mode", "rule_based_rmbg")

    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")

    # ── 1. 真透明 alpha ──
    if img.mode == "RGBA":
        alpha = np.array(img.split()[3])
        if alpha.max() == 0:
            raise ValueError("图片 alpha 全透明，疑似空图")
        if _is_real_alpha(alpha):
            bg = Image.new("RGB", img.size, BG_COLOR)
            bg.paste(img.convert("RGBA"), mask=Image.fromarray(alpha, mode="L"))
            _save_mask(alpha, mask_dir, out_stem)
            return bg, alpha
        img = img.convert("RGB")

    elif img.mode != "RGB":
        img = img.convert("RGB")

    # ── 2. 模式分支 ──
    if bg_mode == "none":
        return img, None

    if bg_mode == "dark_only":
        if _has_dark_background(img, black_threshold=BLACK_THRESHOLD):
            return _rmbg_or_fallback(img, mask_dir, out_stem)
        return img, None

    if bg_mode == "rmbg_all":
        return _rmbg_or_fallback(img, mask_dir, out_stem)

    # rule_based_rmbg (default)
    if _has_white_background(img):
        return img, None
    return _rmbg_or_fallback(img, mask_dir, out_stem)


def _rmbg_or_fallback(img, mask_dir=None, out_stem=None):
    """RMBG 抠图，失败则按 fallback 处理"""
    if RMBG_CFG.get("enabled", False) and _init_rmbg():
        result, mask = _remove_background_rmbg(img)
        _save_mask(mask, mask_dir, out_stem)
        return result, mask

    fallback = BG_CFG.get("fallback", "original")
    if fallback == "dark_threshold" and _has_dark_background(img, black_threshold=BLACK_THRESHOLD):
        return _remove_dark_background(img, black_threshold=BLACK_THRESHOLD), None
    log("白底", "非白底且RMBG不可用(fallback=original)，保持原样")
    return img, None

def step2_to_jpg(src_dir, supported, out_dir):
    """统一背景处理：所有格式统一走 process_background 决策树，逐文件断点续传"""
    if not _ensure_step(out_dir, "white_bg"):
        log("跳过", "白底JPG: 用户选择跳过")
        return

    mask_dir = os.path.join(out_dir, ".masks")
    os.makedirs(mask_dir, exist_ok=True)

    need_process = []
    for f, ext in supported:
        jpg_name = _output_stem(f) + ".jpg"
        if os.path.isfile(os.path.join(out_dir, jpg_name)) and is_valid_jpg(os.path.join(out_dir, jpg_name)):
            continue
        need_process.append((f, ext))

    if not need_process:
        log("跳过", f"白底JPG: 全部 {len(supported)} 张已完成")
        return

    log("转换", f"统一转为白底JPG ({len(need_process)}/{len(supported)}张)...")

    items = need_process if not _HAS_TQDM else tqdm(need_process, desc="白底JPG")
    for f, ext in items:
        try:
            src_path = os.path.join(src_dir, f)
            jpg_name = _output_stem(f) + ".jpg"
            dst_path = os.path.join(out_dir, jpg_name)
            stem = _output_stem(f)

            img = Image.open(src_path)
            result, _mask = process_background(img, out_stem=stem, mask_dir=mask_dir)
            atomic_save_jpg(result, dst_path, quality=QUALITY)

        except ValueError as e:
            log("跳过", f"{f}: {e}")
        except Exception as e:
            log("ERROR", f"{f}: {e}")

    log("完成", f"白底JPG -> {len(need_process)}张")

# ════════════════════════════════════════════════════════
# 第3步：裁剪白边（alpha感知 + padding）
# ════════════════════════════════════════════════════════

def get_bbox_from_array(arr, threshold=245, padding=16):
    """智能检测主体bbox。
    如果RGBA有有效alpha通道，用alpha检测；
    否则用RGB白色阈值检测。
    返回 (x_min, y_min, x_max_exclusive, y_max_exclusive) 或 None。
    """
    h, w = arr.shape[:2]

    if arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        if alpha.min() < 255 and alpha.max() > 10:
            mask = alpha > 10
        else:
            mask = np.any(arr[:, :, :3] < threshold, axis=2)
    else:
        mask = np.any(arr < threshold, axis=2)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y_min, y_max = int(y_indices[0]), int(y_indices[-1])
    x_min, x_max = int(x_indices[0]), int(x_indices[-1])

    # padding（不越界）
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w - 1, x_max + padding)
    y_max = min(h - 1, y_max + padding)

    return x_min, y_min, x_max + 1, y_max + 1

def get_bbox_from_mask(mask_arr, threshold=10, padding=16):
    """从灰度 mask 获取主体 bbox（亮=主体，暗=背景）"""
    h, w = mask_arr.shape[:2]
    mask = mask_arr > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y_min, y_max = int(y_indices[0]), int(y_indices[-1])
    x_min, x_max = int(x_indices[0]), int(x_indices[-1])
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w - 1, x_max + padding)
    y_max = min(h - 1, y_max + padding)
    return x_min, y_min, x_max + 1, y_max + 1


def _crop_from_mask(mask_dir, stem, src_dir, filename, out_dir):
    """从保存的 mask 文件中获取裁剪 bbox（优先级最高：RMBG mask 或 alpha mask）"""
    mask_path = os.path.join(mask_dir, stem + "_mask.png")
    if not os.path.isfile(mask_path):
        return False
    try:
        mask = Image.open(mask_path).convert("L")
        bbox = get_bbox_from_mask(np.array(mask), threshold=10, padding=CROP_PADDING)
        if bbox:
            cropped = Image.open(os.path.join(src_dir, filename)).crop(bbox)
            atomic_save_jpg(cropped, os.path.join(out_dir, filename), quality=QUALITY)
            return True
    except Exception:
        pass
    return False


def _alpha_crop_from_original(base, orig_src_dir, src_dir, filename, out_dir):
    """尝试从原图（PNG/WebP）的alpha通道获取裁剪bbox。成功返回True。"""
    for ext in ['.png', '.webp']:
        orig_path = os.path.join(orig_src_dir, base + ext)
        if not os.path.isfile(orig_path):
            continue
        orig_img = Image.open(orig_path)
        has_alpha = "A" in orig_img.mode or (orig_img.mode == "P" and "transparency" in orig_img.info)
        if not has_alpha:
            continue
        src_arr = np.array(orig_img.convert("RGBA"))
        bbox = get_bbox_from_array(src_arr, threshold=THRESHOLD, padding=CROP_PADDING)
        if bbox:
            cropped = Image.open(os.path.join(src_dir, filename)).crop(bbox)
            atomic_save_jpg(cropped, os.path.join(out_dir, filename), quality=QUALITY)
            return True
    return False


def step3_crop(src_dir, out_dir, orig_src_dir=None):
    """裁剪白边。优先级：白底/.masks mask > 原图alpha > RGB阈值fallback。逐文件断点续传。"""
    if not _ensure_step(out_dir, "crop"):
        log("跳过", "裁剪白边: 用户选择跳过")
        return

    files = iter_files(src_dir)
    need_crop = [f for f in files if not (os.path.isfile(os.path.join(out_dir, f)) and is_valid_jpg(os.path.join(out_dir, f)))]

    if not need_crop:
        log("跳过", f"裁剪白边: 全部 {len(files)} 张已完成")
        return

    # mask 目录在白底目录内（白底/.masks/）
    mask_dir = os.path.join(src_dir, ".masks")
    mask_dir = mask_dir if os.path.isdir(mask_dir) else None

    log("裁剪", f"mask优先裁剪+padding={CROP_PADDING} ({len(need_crop)}/{len(files)}张)...")

    items = need_crop if not _HAS_TQDM else tqdm(need_crop, desc="裁剪白边")
    for f in items:
        try:
            alpha_done = False
            base, orig_ext_enc = _decode_stem(f.rsplit('.', 1)[0])
            stem = Path(f).stem

            # 优先级1：白底/.masks/ 中保存的 RMBG/alpha mask
            if mask_dir:
                alpha_done = _crop_from_mask(mask_dir, stem, src_dir, f, out_dir)

            # 优先级2：原图 alpha 通道
            if not alpha_done and orig_src_dir:
                alpha_done = _alpha_crop_from_original(base, orig_src_dir, src_dir, f, out_dir)

            if alpha_done:
                continue
            # 优先级3：RGB 阈值
            img = Image.open(os.path.join(src_dir, f)).convert("RGB")
            arr = np.array(img)
            bbox = get_bbox_from_array(arr, threshold=THRESHOLD, padding=CROP_PADDING)
            if bbox:
                cropped = img.crop(bbox)
                atomic_save_jpg(cropped, os.path.join(out_dir, f), quality=QUALITY)
            else:
                atomic_save_jpg(img, os.path.join(out_dir, f), quality=QUALITY)
        except Exception as e:
            log("ERROR", f"{f}: {e}")

    log("完成", f"裁剪白边 → {len(need_crop)}张")

# ════════════════════════════════════════════════════════
# 第4步：居中1:1正方形
# ════════════════════════════════════════════════════════

def step4_square(src_dir, out_dir):
    if not _ensure_step(out_dir, "square"):
        log("跳过", "1:1方形: 用户选择跳过")
        return

    files = iter_files(src_dir)
    need_square = [f for f in files if not (os.path.isfile(os.path.join(out_dir, f)) and is_valid_jpg(os.path.join(out_dir, f)))]

    if not need_square:
        log("跳过", f"1:1方形: 全部 {len(files)} 张已完成")
        return

    log("方形", f"居中1:1正方形 ({len(need_square)}/{len(files)}张)...")

    items = need_square if not _HAS_TQDM else tqdm(need_square, desc="1:1方形")
    for f in items:
        try:
            img = Image.open(os.path.join(src_dir, f))
            w, h = img.size
            side = max(w, h)
            sq = Image.new("RGB", (side, side), (255, 255, 255))
            sq.paste(img, ((side - w) // 2, (side - h) // 2))
            atomic_save_jpg(sq, os.path.join(out_dir, f), quality=QUALITY)
        except Exception as e:
            log("ERROR", f"{f}: {e}")

    log("完成", f"1:1方形 → {len(need_square)}张")

# ════════════════════════════════════════════════════════
# 第5步：AI放大1024x1024 + 模型自动下载
# ════════════════════════════════════════════════════════

def _download_model(url, save_path, expected_sha256=None):
    import urllib.request
    log("下载", f"尝试 {url}")
    urllib.request.urlretrieve(url, save_path)
    size = os.path.getsize(save_path)
    if size < 1024 * 1024:
        os.remove(save_path)
        raise ValueError(f"文件太小 ({size} bytes)，不是有效的模型文件")
    # SHA256 校验
    if expected_sha256:
        actual = _sha256_file(save_path)
        if actual != expected_sha256:
            os.remove(save_path)
            raise ValueError(f"SHA256 不匹配！期望 {expected_sha256[:16]}...，实际 {actual[:16]}...")
        log("校验", f"SHA256 OK ({actual[:16]}...)")
    return save_path

def _sha256_file(path):
    """计算文件的 SHA256 哈希"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_realesrgan_model():
    model_path = resolve_path(ESRGAN_CFG.get("model_path", "E:/llamacpp/RealESRGAN_x4plus.pth"))
    expected_sha256 = ESRGAN_CFG.get("sha256", "")

    if os.path.isfile(model_path) and os.path.getsize(model_path) > 1024 * 1024:
        if expected_sha256:
            actual = _sha256_file(model_path)
            if actual != expected_sha256:
                eprint(f"  ! RealESRGAN 模型 SHA256 不匹配，将重新下载")
                eprint(f"    期望: {expected_sha256[:16]}...")
                eprint(f"    实际: {actual[:16]}...")
                os.remove(model_path)
            else:
                log("模型", f"RealESRGAN_x4plus.pth 已存在 ({os.path.getsize(model_path)//1024//1024}MB, SHA256 OK)")
                return model_path
        else:
            log("模型", f"RealESRGAN_x4plus.pth 已存在 ({os.path.getsize(model_path)//1024//1024}MB)")
            return model_path

    mirrors = ESRGAN_CFG.get("download_mirrors", [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ])

    log("下载", "RealESRGAN_x4plus.pth 不存在，尝试自动下载...")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    for url in mirrors:
        try:
            _download_model(url, model_path, expected_sha256 or None)
            log("下载", f"成功！({os.path.getsize(model_path)//1024//1024}MB)")
            return model_path
        except Exception as e:
            log("下载", f"失败: {e}")
            continue

    eprint("  ! 所有镜像下载失败，请手动下载:")
    eprint("    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
    eprint(f"    放到: {model_path}")
    return None

def step5_upscale(src_dir, out_dir):
    if not _ensure_step(out_dir, "upscale"):
        log("跳过", "AI放大: 用户选择跳过")
        return True

    files = iter_files(src_dir)
    if not files:
        return True

    # 逐文件跳过已完成的
    need_upscale = []
    for f in files:
        dst = os.path.join(out_dir, f)
        if os.path.isfile(dst) and is_valid_jpg(dst):
            # 额外验证尺寸是否为 TARGET_SIZE
            try:
                with Image.open(dst) as im:
                    if im.size == (TARGET_SIZE, TARGET_SIZE):
                        continue
            except Exception:
                pass
        need_upscale.append(f)

    if not need_upscale:
        log("跳过", f"AI放大: 全部 {len(files)} 张已完成")
        return True

    if not _HAS_CUDA:
        eprint("  ! CUDA不可用，跳过放大步骤")
        return False

    model_path = ensure_realesrgan_model()
    if not model_path:
        return False

    log("放大", f"AI放大至{TARGET_SIZE}x{TARGET_SIZE} ({len(need_upscale)}张)...")

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        eprint("  ! 缺少 realesrgan 包，请安装: pip install realesrgan basicsr")
        return False

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4, model_path=model_path, model=model,
        tile=400, tile_pad=10, pre_pad=0, half=False, device="cuda",
    )

    os.makedirs(out_dir, exist_ok=True)
    items = need_upscale if not _HAS_TQDM else tqdm(need_upscale, desc="AI放大")
    for f in items:
        try:
            img = Image.open(os.path.join(src_dir, f)).convert("RGB")
            w, h = img.size
            # 无论是否需要超分，最终都 resize 到 TARGET_SIZE x TARGET_SIZE
            if min(w, h) >= TARGET_SIZE:
                # 无需超分，直接缩放到目标尺寸
                result = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            else:
                output, _ = upsampler.enhance(np.array(img), outscale=4)
                result = Image.fromarray(output).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            atomic_save_jpg(result, os.path.join(out_dir, f), quality=QUALITY)
        except Exception as e:
            log("ERROR", f"{f}: {e}")

    log("完成", f"AI放大 → {len(need_upscale)}张")
    return True

# ════════════════════════════════════════════════════════
# llama-server 管理
# ════════════════════════════════════════════════════════

def check_llama_server():
    """检查 llama-server 是否运行且模型已加载完毕。
    使用 /health 端点而非仅 ping 端口，避免模型未就绪时返回 503。"""
    port = LLAMA_CFG.get("port", 8080)
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            # 模型已就绪时 status 为 "ok"，加载中为 "loading model" 等
            return data.get("status") == "ok"
        return False
    except (requests.RequestException, ValueError):
        return False

def start_llama_server():
    global _LLAMA_PROCESS, _LLAMA_STARTED_BY_US

    server_path = resolve_path(LLAMA_CFG.get("path", ""))
    if not server_path or not os.path.isfile(server_path):
        eprint("  ! llama-server.exe 未找到，请检查 config.json 中的 path")
        return False

    model_path = resolve_path(LLAMA_CFG.get("model", ""))
    mmproj_path = resolve_path(LLAMA_CFG.get("mmproj", ""))
    if not model_path or not os.path.isfile(model_path):
        eprint(f"  ! 模型文件未找到: {model_path}")
        return False
    if not mmproj_path or not os.path.isfile(mmproj_path):
        eprint(f"  ! mmproj 文件未找到: {mmproj_path}")
        return False

    if not is_ascii_path(model_path) or not is_ascii_path(mmproj_path):
        raise RuntimeError(
            "llama-server 对中文路径支持不稳定。请将 model/mmproj 放到纯英文路径，"
            "例如 E:/models/qwen3_vl_4b/，或使用 mklink /J 创建英文目录别名。"
        )

    port = LLAMA_CFG.get("port", 8080)
    host = LLAMA_CFG.get("host", "127.0.0.1")
    ngl = LLAMA_CFG.get("ngl", 42)
    ctx = LLAMA_CFG.get("ctx_size", 32000)
    batch_size = LLAMA_CFG.get("batch_size", 512)
    ubatch_size = LLAMA_CFG.get("ubatch_size", 128)
    threads = LLAMA_CFG.get("threads", 8)
    threads_batch = LLAMA_CFG.get("threads_batch", 8)
    cache_k = LLAMA_CFG.get("cache_type_k", "q8_0")
    cache_v = LLAMA_CFG.get("cache_type_v", "q8_0")
    no_mmap = LLAMA_CFG.get("no_mmap", True)
    kv_unified = LLAMA_CFG.get("kv_unified", True)
    flash_attn = LLAMA_CFG.get("flash_attn", "on")

    cmd = [
        server_path,
        "-m", model_path,
        "--mmproj", mmproj_path,
        "--host", host,
        "--port", str(port),
        "-ngl", str(ngl),
        "--ctx-size", str(ctx),
        "--batch-size", str(batch_size),
        "--ubatch-size", str(ubatch_size),
        "--threads", str(threads),
        "--threads-batch", str(threads_batch),
        "--cache-type-k", cache_k,
        "--cache-type-v", cache_v,
        "--flash-attn", str(flash_attn),
    ]
    if no_mmap:
        cmd.append("--no-mmap")
    if kv_unified:
        cmd.append("--kv-unified")

    log("服务", "启动 llama-server（后台，需等待约10秒加载模型）...")

    try:
        # 直接 Popen（路径全为 ASCII，无需 bat 包装）
        _LLAMA_PROCESS = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        _LLAMA_STARTED_BY_US = True

        for i in range(60):
            time.sleep(1)
            if check_llama_server():
                log("服务", "llama-server 就绪 OK")
                return True
            if i == 5:
                log("服务", "模型仍在加载中，请等待...")
        eprint("  ! llama-server 启动超时（60s）")
        return False
    except Exception as e:
        eprint(f"  ! 启动llama-server失败: {e}")
        return False

def ensure_llama_server():
    if check_llama_server():
        # 用户已有服务，不标记为我们的
        return True
    eprint("  ! llama-server 未运行")
    if LLAMA_CFG.get("path"):
        return start_llama_server()
    return False

# ════════════════════════════════════════════════════════
# 第6步：VLM 多模式打标
# ════════════════════════════════════════════════════════

# ── 打标模式定义 ──
TAG_MODE_SPECS = {
    "bilingual": {
        "label": "中英双语",
        "system": (
            "You are an image captioning assistant.\n"
            "Output exactly two paragraphs separated by one blank line.\n"
            "Paragraph 1 must be Simplified Chinese.\n"
            "Paragraph 2 must be English.\n"
            "Do not use labels such as Chinese, English, CN, EN.\n"
            "Do not identify the character by name.\n"
            "Do not mention art style, medium, image quality, or subjective opinions.\n"
            "Describe only visible physical appearance, clothing, accessories, pose, and colors."
        ),
        "user": (
            "用一段中文一段英文简要描述这张图片里的角色外观，"
            "不要出现\"中文：\" \"English:\"标签，不要掺杂主观描述，"
            "禁止风格类描述词，禁止出现\"这个角色是谁\""
        ),
        "max_tokens": 400,
        "validator": "bilingual",
        "cleanup_prefixes": ["CN:", "CN：", "EN:", "EN：", "中文：", "English：", "English:"],
    },
    "chinese_only": {
        "label": "纯中文",
        "system": (
            "You are an image captioning assistant.\n"
            "Describe the character in Simplified Chinese ONLY.\n"
            "Write 2-4 sentences describing visible appearance, clothing, accessories, pose, and colors.\n"
            "Do not identify the character by name.\n"
            "Do not mention art style, medium, image quality, or subjective opinions."
        ),
        "user": (
            "用中文简要描述这张图片里角色的外观、服装、配饰、姿势和颜色，"
            "不要掺杂主观描述，禁止风格类词汇"
        ),
        "max_tokens": 250,
        "validator": "chinese",
        "cleanup_prefixes": ["中文：", "中文:", "CN：", "CN:"],
    },
    "english_only": {
        "label": "纯英文",
        "system": (
            "You are an image captioning assistant.\n"
            "Describe the character in English ONLY.\n"
            "Write 2-4 sentences describing visible appearance, clothing, accessories, pose, and colors.\n"
            "Do not identify the character by name.\n"
            "Do not mention art style, medium, image quality, or subjective opinions."
        ),
        "user": (
            "Briefly describe this character's appearance, clothing, accessories, pose and colors in English. "
            "No subjective opinions, no art style words."
        ),
        "max_tokens": 250,
        "validator": "english",
        "cleanup_prefixes": ["English:", "EN:", "EN：", "英文：", "english:"],
    },
    "natural": {
        "label": "自然语言",
        "system": (
            "You are an image captioning assistant.\n"
            "Describe what you see in natural language.\n"
            "Mention visible appearance, clothing, accessories, pose, colors, and any notable details.\n"
            "You may write in Chinese, English, or both — whatever feels natural.\n"
            "Do not use form labels like 'Chinese:' or 'English:'.\n"
            "Do not mention art style or subjective opinions."
        ),
        "user": (
            "用自然的语言描述这张图片里你看到的内容，不需要拘泥于格式，不要掺杂主观性描述"
        ),
        "max_tokens": 350,
        "validator": "any",
        "cleanup_prefixes": [],
    },
    "danbooru": {
        "label": "Danbooru标签",
        "system": (
            "You are a Danbooru-style image tagger.\n"
            "List comma-separated tags describing the character ONLY.\n"
            "Tags must be in English (use underscores for spaces, e.g. 'red_hair').\n"
            "Include tags for: gender, hair color, hair style, eye color, clothing items, "
            "accessories, pose, expression, and any visible features.\n"
            "Do NOT include: artist names, series names, character names, rating tags, "
            "art style tags (anime, realistic, etc.), or subjective tags (cute, beautiful, etc.).\n"
            "Output only the comma-separated tags, nothing else."
        ),
        "user": (
            "List Danbooru-style tags (comma-separated, English with underscores) for this character. "
            "Include: gender, hair, eyes, clothing, accessories, pose, expression. "
            "NO artist/series/character names, NO style tags, NO subjective tags."
        ),
        "max_tokens": 500,
        "validator": "danbooru",
        "cleanup_prefixes": ["Tags:", "tags:", "Tag:", "标签：", "标签:"],
    },
    "danbooru_bilingual": {
        "label": "中英Danbooru",
        "system": (
            "You are a Danbooru-style tagger. Output EXACTLY TWO lines, nothing else.\n"
            "Line 1: English tags (comma-separated, underscores for spaces, e.g. male, red_hair, blue_eyes, armor)\n"
            "Line 2: Chinese tags (comma-separated, e.g. 男性, 红发, 蓝眼, 盔甲)\n"
            "Do NOT label lines with 'English:' or 'Chinese:'. Do NOT write sentences.\n"
            "Include: gender, hair, eyes, clothing, accessories, pose, expression.\n"
            "5-10 tags per line. NO artist/series/character names, NO style/subjective tags."
        ),
        "user": (
            "FIRST LINE: English tags. SECOND LINE: Chinese tags. "
            "Comma-separated tags only. Gender, hair, eyes, clothes, accessories, pose, expression. "
            "NO sentences, NO labels, NO style tags."
        ),
        "max_tokens": 500,
        "validator": "danbooru_bilingual",
        "cleanup_prefixes": ["中文：", "中文:", "English:", "EN:", "标签：", "Tags:", "英文：", "英文:", "english:"],
    },
}

# 默认模式
TAG_MODE = CFG.get("tag_mode", "bilingual")
TAG_SPEC = TAG_MODE_SPECS.get(TAG_MODE, TAG_MODE_SPECS["bilingual"])

# 向后兼容：保留旧的全局引用
SYSTEM_PROMPT_VLM = TAG_SPEC["system"]

BAD_PREFIXES = ["中文：", "中文:", "English:", "英文：", "CN:", "EN:"]
STYLE_WORDS = ["动漫", "卡通", "写实", "插画", "风格", "anime", "cartoon", "realistic", "illustration", "style"]
SUBJECTIVE_WORDS = ["可爱", "漂亮", "好看", "帅气", "美少女", "beautiful", "cute", "handsome", "lovely"]

# ── 模式感知校验 ──

def _has_banned_words(text):
    """检查是否含有禁用词（中英文分开处理）"""
    lowered = text.lower()
    for w in STYLE_WORDS + SUBJECTIVE_WORDS:
        wl = w.lower()
        if re.search(r'[一-鿿]', wl):
            if wl in lowered:
                return True
        else:
            if re.search(r'\b' + re.escape(wl) + r'\b', lowered):
                return True
    return False


def validate_tag(text, mode=None):
    """根据打标模式校验输出质量。mode 为 None 时使用当前 TAG_MODE。"""
    mode = mode or TAG_MODE
    text = text.strip()
    if not text or len(text) < 10:
        return False

    # 检查禁用前缀
    for p in BAD_PREFIXES:
        if p in text:
            return False

    # 禁用词检查
    if _has_banned_words(text):
        return False

    validator = TAG_MODE_SPECS.get(mode, {}).get("validator", "any")

    if validator == "bilingual":
        return _validate_bilingual(text)
    elif validator == "chinese":
        cn = len(re.findall(r'[一-鿿]', text))
        return cn >= 10
    elif validator == "english":
        en = len(re.findall(r'\b[a-zA-Z]{2,}\b', text))
        return en >= 6
    elif validator == "danbooru":
        tags = [t.strip() for t in text.split(',') if t.strip()]
        return len(tags) >= 5 and all(' ' not in t or '_' in t for t in tags)
    elif validator == "danbooru_bilingual":
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) < 2:
            return False
        # 兼容中文逗号
        a = lines[0].replace('，', ',')
        b = lines[1].replace('，', ',')
        a_tags = [t.strip() for t in a.split(',') if t.strip()]
        b_tags = [t.strip() for t in b.split(',') if t.strip()]
        # 判断哪行是中文
        a_cn = len(re.findall(r'[一-鿿]', a))
        b_cn = len(re.findall(r'[一-鿿]', b))
        if a_cn >= 5 and b_cn < 5:
            cn_tags, en_tags = a_tags, b_tags
            cn_chars = a_cn
        elif b_cn >= 5 and a_cn < 5:
            cn_tags, en_tags = b_tags, a_tags
            cn_chars = b_cn
        elif a_cn >= 5 and b_cn >= 5:
            # 两行都有中文，取更长的那行作为中文
            if a_cn >= b_cn:
                cn_tags, en_tags = a_tags, b_tags
                cn_chars = a_cn
            else:
                cn_tags, en_tags = b_tags, a_tags
                cn_chars = b_cn
        else:
            return False
        return len(cn_tags) >= 5 and len(en_tags) >= 5 and cn_chars >= 5
    elif validator == "any":
        return True
    return True


def strip_thinking(text):
    """清理 <think>...</think> 标签"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _validate_bilingual(text):
    """段落级双语校验（内部使用，validate_tag 在 bilingual 模式下调此函数）"""
    text = text.strip()
    if not text:
        return False

    for p in BAD_PREFIXES:
        if p in text:
            return False

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) < 2:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) >= 2:
            cn_idx = 0
            for i, line in enumerate(lines):
                cn_count = len(re.findall(r'[一-鿿]', line))
                en_count = len(re.findall(r'\b[a-zA-Z]{2,}\b', line))
                if cn_count < 5 and en_count > cn_count:
                    cn_idx = i
                    break
            if cn_idx > 0:
                cn_part = "".join(lines[:cn_idx])
                en_part = " ".join(lines[cn_idx:])
            else:
                cn_part, en_part = lines[0], " ".join(lines[1:])
        else:
            return False
    else:
        cn_part, en_part = paragraphs[0], " ".join(paragraphs[1:])

    cn_chars = len(re.findall(r'[一-鿿]', cn_part))
    if cn_chars < 10:
        return False

    en_words = len(re.findall(r'\b[a-zA-Z]{2,}\b', en_part))
    if en_words < 6:
        return False

    if _has_banned_words(text):
        return False

    return True


# 向后兼容别名
validate_bilingual = lambda t: validate_tag(t, "bilingual")


def tag_one_image(img_path, api_url, max_tokens=None, temperature=None, system_prompt=None, user_prompt=None):
    """对单张图片进行VLM打标（预缩放+base64+api）。使用当前 TAG_MODE 的 prompt。"""
    img = Image.open(img_path)
    w, h = img.size
    if max(w, h) > VLM_PRESIZE:
        scale = VLM_PRESIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, "JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    sp = system_prompt or TAG_SPEC["system"]
    up = user_prompt or TAG_SPEC.get("user", VLM_PROMPT)
    mt = max_tokens or TAG_SPEC.get("max_tokens", VLM_MAX_TOKENS)
    tp = temperature if temperature is not None else VLM_TEMP

    payload = {
        "messages": [
            {"role": "system", "content": sp},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": up}
            ]}
        ],
        "max_tokens": mt,
        "temperature": tp,
    }
    resp = requests.post(api_url, json=payload, timeout=120)
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"VLM API error: {data['error']}")

    if "choices" not in data or not data["choices"]:
        raise KeyError(f"无choices: {list(data.keys())[:5]} | {str(data)[:200]}")

    choice = data["choices"][0]
    if "message" in choice and "content" in choice["message"]:
        raw = choice["message"]["content"]
    elif "delta" in choice and "content" in choice["delta"]:
        raw = choice["delta"]["content"]
    elif "text" in choice:
        raw = choice["text"]
    else:
        raise KeyError(f"choices无内容字段: {list(choice.keys())}")

    raw = strip_thinking(raw)
    # 根据模式清理前缀
    for prefix in TAG_SPEC.get("cleanup_prefixes", []):
        raw = raw.replace(prefix, "")
    return raw.strip()


def step6_tag(input_dir, output_dir):
    """VLM 打标：支持所有格式（PNG/JPG/WebP），txt 使用 _output_stem 命名防重"""
    files = iter_files(input_dir, exts=SUPPORTED_IMG_EXTS)
    if not files:
        return

    port = LLAMA_CFG.get("port", 8080)
    api_url = f"http://127.0.0.1:{port}/chat/completions"

    # manifest 检查：如果 VLM 参数变了，清空所有 txt 重新打标
    if _check_manifest(output_dir, "tag"):
        if os.path.isdir(output_dir) and any(f.endswith('.txt') for f in os.listdir(output_dir)):
            log("参数变更", "VLM配置已修改，需清空旧标签重新打标")
            if _confirm_deletion("VLM 标注参数已变更，需要删除现有 txt 文件重新打标"):
                for f in os.listdir(output_dir):
                    if f.endswith('.txt'):
                        os.remove(os.path.join(output_dir, f))
                _write_manifest(output_dir, "tag")
            else:
                log("跳过", "VLM打标: 用户选择跳过，保留旧标签")
                _write_manifest(output_dir, "tag")
                return
        else:
            _write_manifest(output_dir, "tag")

    os.makedirs(output_dir, exist_ok=True)

    # 跳过已标注且内容合格的文件
    existing_txts = {}
    for f in os.listdir(output_dir):
        if f.endswith('.txt'):
            fp = os.path.join(output_dir, f)
            if os.path.getsize(fp) >= 10:
                content = open(fp, encoding='utf-8').read()
                if validate_tag(content):
                    existing_txts[f.rsplit('.', 1)[0]] = True

    need_tag = [f for f in files if tag_txt_name(f).rsplit('.', 1)[0] not in existing_txts]

    # 打标第一轮
    if need_tag:
        log("打标", f"VLM标注 ({len(need_tag)}张)...")
        items = need_tag if not _HAS_TQDM else tqdm(need_tag, desc="VLM打标")
        for f in items:
            try:
                raw = retry_call(
                    lambda: tag_one_image(os.path.join(input_dir, f), api_url),
                    retries=2, delay=3, label=f
                )
                txt_name = tag_txt_name(f)
                atomic_write_text(raw, os.path.join(output_dir, txt_name))
            except Exception as e:
                log("ERROR", f"{f}: {e}")

        # 缺txt补漏
        missing = []
        for f in iter_files(input_dir, exts=SUPPORTED_IMG_EXTS):
            txt_name = tag_txt_name(f)
            if txt_name not in os.listdir(output_dir):
                missing.append(f)
        if missing:
            log("补漏", f"发现 {len(missing)} 张缺txt，重试...")
            for f in missing:
                try:
                    raw = retry_call(
                        lambda: tag_one_image(os.path.join(input_dir, f), api_url),
                        retries=3, delay=5, label=f
                    )
                    txt_name = tag_txt_name(f)
                    atomic_write_text(raw, os.path.join(output_dir, txt_name))
                except Exception as e:
                    log("ERROR", f"(补漏) {f}: {e}")

    # 校验 + 补跑
    _verify_and_retry(input_dir, output_dir, api_url)


def _verify_and_retry(input_dir, output_dir, api_url):
    """模式感知校验 + 翻译补救（仅 bilingual 模式）。"""
    all_txt = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    if not all_txt:
        return

    failed = [f for f in all_txt
              if not validate_tag(open(os.path.join(output_dir, f), encoding='utf-8').read())]

    if not failed:
        log("校验", f"全部有效 OK ({len(all_txt)}/{len(all_txt)}) [{TAG_SPEC['label']}]")
        return

    # 只有 bilingual 模式才做翻译补救
    if TAG_MODE == "bilingual":
        log("校验", f"{len(failed)} 个文件缺少英文，直接用VLM翻译...")
        _translate_fallback(input_dir, output_dir, api_url, failed)

        remaining = sum(1 for f in all_txt
                        if not validate_tag(open(os.path.join(output_dir, f), encoding='utf-8').read()))
        if remaining:
            log("校验", f"警告: {remaining} 个文件翻译仍未通过")
        else:
            log("校验", f"全部双语有效 OK ({len(all_txt)}/{len(all_txt)})")
    elif TAG_MODE == "danbooru_bilingual":
        # 先尝试用更强的 prompt 补跑英→中缺失的
        log("校验", f"{len(failed)} 个文件格式不完整，用更强prompt补跑...")
        _danbooru_retry(input_dir, output_dir, api_url, failed)

        # 补跑后仍失败的，尝试翻译英文标签→中文标签
        still_failed = [f for f in all_txt
                        if not validate_tag(open(os.path.join(output_dir, f), encoding='utf-8').read(), "danbooru_bilingual")]
        if still_failed:
            log("校验", f"{len(still_failed)} 个仍缺中文，翻译英文标签补救...")
            _danbooru_translate(output_dir, api_url, still_failed)

        remaining = sum(1 for f in all_txt
                        if not validate_tag(open(os.path.join(output_dir, f), encoding='utf-8').read(), "danbooru_bilingual"))
        if remaining:
            log("校验", f"警告: {remaining} 个文件翻译仍未通过")
        else:
            log("校验", f"全部有效 OK ({len(all_txt)}/{len(all_txt)}) [{TAG_SPEC['label']}]")
    else:
        log("校验", f"警告: {len(failed)} 个文件未通过 [{TAG_SPEC['label']}] 模式校验")
        for f in failed:
            eprint(f"    - {f}")


def _find_image_for_txt(input_dir, txt_name):
    """根据 txt 文件名查找对应图片（适配 abc__png.txt 命名）"""
    stem = Path(txt_name).stem
    # 1. 同 stem 的图片（适配已处理的 JPG）
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = os.path.join(input_dir, stem + ext)
        if os.path.isfile(p):
            return p
    # 2. _decode_stem 后的原始扩展名
    base, orig_ext = _decode_stem(stem)
    if orig_ext:
        for ext in [f".{orig_ext}", ".jpg", ".jpeg", ".png", ".webp"]:
            p = os.path.join(input_dir, base + ext)
            if os.path.isfile(p):
                return p
    return None


def _danbooru_retry(input_dir, output_dir, api_url, failed_files):
    """对 danbooru_bilingual 格式不完整的，用超强 prompt 补跑1轮"""
    retry_prompt = (
        "EXAMPLE OUTPUT (YOU MUST COPY THIS FORMAT EXACTLY):\n"
        "male, red_hair, blue_eyes, armor, sword, standing, smiling\n"
        "男性, 红发, 蓝眼, 盔甲, 剑, 站立, 微笑\n\n"
        "Follow this format: Line 1 = English tags (comma, underscores). "
        "Line 2 = Chinese tags (comma). Both lines REQUIRED."
    )
    for f in failed_files:
        try:
            src_img_path = _find_image_for_txt(input_dir, f)
            if src_img_path is None:
                continue

            raw = retry_call(
                lambda: tag_one_image(src_img_path, api_url,
                                      system_prompt=retry_prompt,
                                      temperature=0.05),
                retries=1, delay=2, label=f
            )
            atomic_write_text(raw, os.path.join(output_dir, f))
        except Exception as e:
            log("ERROR", f"补跑 danbooru {f}: {e}")


def _danbooru_translate(output_dir, api_url, failed_files):
    """对只有英文标签的 danbooru 输出，用 VLM 将英文标签翻译为中文标签"""
    for f in failed_files:
        try:
            fp = os.path.join(output_dir, f)
            content = open(fp, encoding='utf-8').read().strip()
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            if not lines:
                continue
            # 找英文标签行
            en_line = lines[0]
            cn = len(re.findall(r'[一-鿿]', en_line))
            if cn >= 5:
                # 第一行已经是中文，跳过
                continue
            log("翻译", f"{f}: 英文标签→中文标签")
            translate_prompt = (
                "Translate these Danbooru-style English tags into Chinese tags.\n"
                "Keep them as comma-separated single words/phrases.\n"
                "Output ONLY the Chinese tags, nothing else.\n\n"
                f"English: {en_line}"
            )
            cn_raw = retry_call(
                lambda: _tag_text_only(api_url, translate_prompt, max_tokens=300),
                retries=2, delay=2, label=f
            )
            cn_tags = cn_raw.strip().replace('，', ',')
            # 组合：英文第一行，中文第二行
            merged = en_line + "\n" + cn_tags
            atomic_write_text(merged, fp)
            log("翻译", f"{f}: 翻译补救成功")
        except Exception as e:
            log("ERROR", f"翻译 danbooru {f}: {e}")


def _translate_fallback(input_dir, output_dir, api_url, failed_files):
    """对只有中文的 txt 文件，用 VLM 将中文翻译为英文并补全双语格式"""
    for f in failed_files:
        try:
            fp = os.path.join(output_dir, f)
            content = open(fp, encoding='utf-8').read().strip()
            if not content:
                continue

            # 提取中文部分（去掉可能的英文残留）
            lines = content.split('\n')
            cn_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                cn_chars = len(re.findall(r'[一-鿿]', line))
                en_words = len(re.findall(r'\b[a-zA-Z]{2,}\b', line))
                # 中文主导：中文字符远多于英文单词
                if cn_chars >= 5 and cn_chars > en_words:
                    cn_lines.append(line)

            cn_text = ' '.join(cn_lines)
            cn_chars = len(re.findall(r'[一-鿿]', cn_text))
            if cn_chars < 10:
                continue  # 中文太少，无法翻译

            log("翻译", f"{f}: 中文→英文翻译 ({cn_chars}字)")

            translate_prompt = (
                "Translate the following Chinese description into English.\n"
                "Keep it factual, describe only visible appearance.\n"
                "No art style words, no subjective opinions.\n"
                "Output ONLY the English translation, nothing else.\n\n"
                f"Chinese: {cn_text}"
            )

            en_raw = retry_call(
                lambda: _tag_text_only(api_url, translate_prompt),
                retries=2, delay=2, label=f
            )
            en_text = en_raw.strip()

            # 去掉可能的前缀
            for prefix in ["English:", "英文：", "EN:", "Translation:", "翻译："]:
                if en_text.lower().startswith(prefix.lower()):
                    en_text = en_text[len(prefix):].strip()

            en_words = len(re.findall(r'\b[a-zA-Z]{2,}\b', en_text))
            if en_words < 4:
                log("翻译", f"{f}: 翻译结果英文不足 ({en_words}词)，跳过")
                continue

            # 组合中英双语
            merged = cn_text + "\n\n" + en_text
            atomic_write_text(merged, fp)
            log("翻译", f"{f}: 翻译补救成功")

        except Exception as e:
            log("ERROR", f"翻译补救 {f}: {e}")


def _tag_text_only(api_url, user_text, max_tokens=400, temperature=0.1):
    """纯文本 VLM 请求（无图片），用于翻译等文本任务"""
    payload = {
        "messages": [
            {"role": "system", "content": "You are a translator. Output only the translation."},
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(api_url, json=payload, timeout=120)
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"VLM API error: {data['error']}")

    choice = data["choices"][0]
    if "message" in choice and "content" in choice["message"]:
        return choice["message"]["content"]
    elif "delta" in choice and "content" in choice["delta"]:
        return choice["delta"]["content"]
    elif "text" in choice:
        return choice["text"]
    else:
        raise KeyError(f"choices无内容字段: {list(choice.keys())}")



# ════════════════════════════════════════════════════════
# 步骤选择 + 命令行解析
# ════════════════════════════════════════════════════════

STEP_LABELS = {
    1: "图片格式筛查",
    2: "白底JPG (透明合成+黑底翻白)",
    3: "裁剪白边 (alpha感知+padding)",
    4: "居中1:1正方形",
    5: "AI放大至1024x1024",
    6: "VLM双语打标",
}

STEP_KEYS = {1: "scan", 2: "white_bg", 3: "crop", 4: "square", 5: "upscale", 6: "tag"}

# 步骤链式依赖：只需定义直接前置，自动补全全链
# 例如选步骤5，自动补全 4,3,2
STEP_DIRECT_DEPS = {
    2: set(),      # 白底无前置
    3: {2},        # 裁剪需要白底
    4: {3},        # 方形需要裁剪
    5: {4},        # 放大需要方形
    6: set(),      # 打标独立，不自动补全
}

def _resolve_step_deps(steps):
    """自动补全链式依赖：选 5 则自动补 4,3,2"""
    result = set(steps)
    changed = True
    while changed:
        changed = False
        for s in list(result):
            for dep in STEP_DIRECT_DEPS.get(s, set()):
                if dep not in result:
                    result.add(dep)
                    changed = True
    return result

# 打标模式下，如果 HD 有图就用 HD，否则用 方形
TAG_INPUT_DIRS = {5: "HD", 4: "方形"}

def parse_steps():
    global _FORCE_YES
    # 解析参数：收集目录和选项
    src_dir = None
    steps_flag = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--steps':
            if i + 1 < len(sys.argv):
                steps_flag = sys.argv[i + 1]
                i += 2
                continue
            else:
                eprint("! --steps 需要参数，如 --steps 2,3,6")
                sys.exit(1)
        elif arg in ('--yes', '--force', '-y'):
            _FORCE_YES = True
            i += 1
            continue
        elif arg == '--tag-mode':
            if i + 1 < len(sys.argv):
                global TAG_MODE, TAG_SPEC, SYSTEM_PROMPT_VLM
                mode = sys.argv[i + 1]
                if mode not in TAG_MODE_SPECS:
                    eprint(f"! 未知打标模式: {mode}，支持: {', '.join(TAG_MODE_SPECS.keys())}")
                    sys.exit(1)
                TAG_MODE = mode
                TAG_SPEC = TAG_MODE_SPECS[mode]
                SYSTEM_PROMPT_VLM = TAG_SPEC["system"]
                i += 2
                continue
            else:
                eprint("! --tag-mode 需要参数，如 --tag-mode danbooru")
                sys.exit(1)
        elif arg.startswith('--'):
            # 其他 -- 标记，跳过
            i += 1
            continue
        else:
            if src_dir is None:
                src_dir = arg
            i += 1

    if src_dir is None:
        eprint("用法: python pipeline.py <目标文件夹> [选项]")
        eprint("选项:")
        eprint("  --full        全流程 (1-6，默认)")
        eprint("  --preprocess  预处理 (1-5: 白底+裁剪+方形+放大)")
        eprint("  --tag         仅打标 (6)")
        eprint("  --steps N,M   自定义步骤 (如 --steps 2,3,6)")
        eprint("  --tag-mode M  打标模式: bilingual/chinese_only/english_only/natural/danbooru/danbooru_bilingual")
        eprint("  --yes, --force  跳过所有删除确认（非交互式批处理）")
        eprint("示例:")
        eprint("  python pipeline.py E:/打标/potato --tag")
        eprint("  python pipeline.py E:/打标/potato --steps 2,3,4")
        sys.exit(1)

    src_dir = os.path.abspath(src_dir)
    if not os.path.isdir(src_dir):
        eprint(f"! 目录不存在: {src_dir}")
        sys.exit(1)

    args_set = set(a for a in sys.argv[1:] if a.startswith('--'))

    # 默认全流程
    steps = set(range(1, 7))

    if '--full' in args_set:
        pass  # 默认全流程
    elif '--tag' in args_set:
        steps = {6}
    elif '--preprocess' in args_set:
        steps = {1, 2, 3, 4, 5}
    elif steps_flag is not None:
        try:
            steps = set(int(s.strip()) for s in steps_flag.split(','))
            if not steps.issubset(set(range(1, 7))):
                raise ValueError
        except (ValueError, IndexError):
            eprint("! 步骤格式错误，请使用逗号分隔，如 --steps 2,3,6")
            sys.exit(1)
    else:
        # 无参数 → 交互式选择
        if not sys.stdin.isatty():
            eprint("  [检测到非交互式环境，默认执行全流程 --full]")
        else:
            steps = _interactive_step_select()

    # 自动补全链式依赖：选 5 自动补 4,3,2；选 6 独立运行不补
    resolved = _resolve_step_deps(steps)
    added = resolved - steps
    if added:
        eprint(f"  提示: 自动补全依赖步骤 {', '.join(f'{s}({STEP_LABELS[s]})' for s in sorted(added))}")
        steps = resolved

    tag_mode_label = f" [{TAG_SPEC['label']}]" if 6 in steps else ""
    eprint(f"\n{'='*60}")
    eprint(f"  图片处理流水线 v5.3.1")
    eprint(f"  源目录: {src_dir}")
    eprint(f"  打标模式: {TAG_SPEC['label']}" if 6 in steps else f"  打标模式: (未启用)")
    eprint(f"  执行步骤: {', '.join(f'{s}({STEP_LABELS[s]})' for s in sorted(steps))}")
    eprint(f"{'='*60}\n")

    return src_dir, steps


def _interactive_step_select():
    """交互式选择要执行的步骤"""
    eprint("\n请选择要执行的步骤（可多选）:")
    eprint("-" * 50)
    for s in range(1, 7):
        eprint(f"  [{s}] {STEP_LABELS[s]}")
    eprint("-" * 50)
    eprint("  输入格式: 逗号分隔如 1,2,3 或范围如 1-6")
    eprint("  快捷: a=全流程   p=预处理(1-5)   t=仅打标(6)")
    eprint("-" * 50)

    while True:
        try:
            answer = input("  >>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            eprint("\n  用户终止")
            sys.exit(1)

        if answer in ('a', 'all', ''):
            return set(range(1, 7))
        elif answer in ('p', 'preprocess'):
            return {1, 2, 3, 4, 5}
        elif answer in ('t', 'tag'):
            return {6}

        # 范围格式: 1-6
        if '-' in answer and ',' not in answer:
            parts = answer.split('-')
            if len(parts) == 2:
                try:
                    lo, hi = int(parts[0]), int(parts[1])
                    return set(range(lo, hi + 1))
                except ValueError:
                    pass

        # 逗号分隔
        try:
            nums = [int(x.strip()) for x in answer.split(',') if x.strip()]
            if all(1 <= n <= 6 for n in nums):
                return set(nums)
        except ValueError:
            pass

        eprint("  输入无效，请重新输入")


def main():
    global _RMBG_MODEL, _RMBG_TRANSFORM, _RMBG_DEVICE

    if _MISSING_DEPS:
        eprint("! 缺少依赖:")
        for d in _MISSING_DEPS:
            eprint(f"    {d}")
        sys.exit(1)

    src_dir, steps = parse_steps()

    dir_white = os.path.join(src_dir, "白底")
    dir_cropped = os.path.join(dir_white, "裁剪后")
    dir_square = os.path.join(dir_cropped, "方形")
    dir_hd = os.path.join(dir_square, "HD")

    T0 = time.time()

    # ── 第1步：图片格式筛查 ──
    supported = None
    if 1 in steps or 2 in steps:
        supported = step1_scan(src_dir)

    # ── 第2步：白底JPG ──
    if 2 in steps:
        step2_to_jpg(src_dir, supported, dir_white)
        _write_manifest(dir_white, "white_bg")
        log("清理", "释放RMBG显存...")
        _RMBG_MODEL = _RMBG_TRANSFORM = _RMBG_DEVICE = None
        clear_cuda()
    elif not os.path.isdir(dir_white):
        dir_white = None

    # ── 第3步：裁剪白边 ──
    if 3 in steps and dir_white:
        step3_crop(dir_white, dir_cropped, orig_src_dir=src_dir)
        _write_manifest(dir_cropped, "crop")
    elif not os.path.isdir(dir_cropped):
        dir_cropped = None

    # ── 第4步：1:1正方形 ──
    if 4 in steps and dir_cropped:
        step4_square(dir_cropped, dir_square)
        _write_manifest(dir_square, "square")
    elif not os.path.isdir(dir_square):
        dir_square = None

    # ── 第5步：AI放大 ──
    if 5 in steps and dir_square:
        step5_upscale(dir_square, dir_hd)
        _write_manifest(dir_hd, "upscale")
        log("清理", "释放RealESRGAN显存...")
        clear_cuda()
    elif not os.path.isdir(dir_hd):
        dir_hd = None

    # ── 第6步：VLM打标 ──
    if 6 in steps:
        # 查找可打标的目录：优先 HD > 方形 > 裁剪后 > 白底 > 源目录
        tag_src = None
        for d in [dir_hd, dir_square, dir_cropped, dir_white, src_dir]:
            if d and os.path.isdir(d) and any(
                os.path.splitext(f)[1].lower() in SUPPORTED_IMG_EXTS
                for f in os.listdir(d)
            ):
                tag_src = d
                break

        if tag_src is None:
            eprint("  ! 未找到可打标的图片，请先执行预处理步骤")
        else:
            log("信息", f"输入目录: {tag_src}")
            if 5 not in steps and 4 not in steps and 3 not in steps and 2 not in steps:
                log("信息", "仅打标模式：图片需已经是白底JPG且为1024x1024")

            ensure_llama_server()
            if check_llama_server():
                step6_tag(tag_src, tag_src)
                _write_manifest(tag_src, "tag")
            else:
                log("跳过", "llama-server 未运行，跳过VLM打标")

    # ── 汇总 ──
    elapsed = time.time() - T0
    # 找到最深的输出目录（优先HD，其次方形→裁剪后→白底→源目录仅当有txt时）
    final_dir = None
    for d in [dir_hd, dir_square, dir_cropped, dir_white]:
        if d and os.path.isdir(d) and any(f.lower().endswith('.jpg') for f in os.listdir(d)):
            final_dir = d
            break
    # 仅打标模式：源目录也有JPG
    if final_dir is None:
        if os.path.isdir(src_dir) and any(f.lower().endswith('.jpg') for f in os.listdir(src_dir)):
            final_dir = src_dir

    if final_dir:
        img_count = len([
            f for f in os.listdir(final_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_IMG_EXTS
        ])
        txt_count = len([f for f in os.listdir(final_dir) if f.endswith('.txt')])
        missing_txt = max(0, img_count - txt_count)
        if txt_count > 0 and os.path.isdir(final_dir):
            bilingual_ok = sum(
                1 for f in os.listdir(final_dir) if f.endswith('.txt')
                and validate_tag(open(os.path.join(final_dir, f), encoding='utf-8').read())
            )

        eprint(f"\n{'='*60}")
        eprint(f"  处理完成！耗时 {elapsed:.0f}s")
        eprint(f"  输出位置: {final_dir}")
        eprint(f"  JPG: {img_count} 张  |  TXT: {txt_count} 个")
        if missing_txt:
            eprint(f"  !!! {missing_txt} 个JPG缺少txt文件，需手动补标")
        if txt_count > 0:
            eprint(f"  双语有效: {bilingual_ok}/{txt_count}")
        eprint(f"{'='*60}")
    else:
        eprint(f"\n{'='*60}")
        eprint(f"  处理完成！耗时 {elapsed:.0f}s")
        eprint(f"{'='*60}")

    # 自动清理：终止 llama-server，释放 GPU 显存和内存
    # 注：stop_llama_server() 在 module-level 的 try/finally 中统一调用


def stop_llama_server():
    """只关闭自己启动的 llama-server，不杀用户已有的进程"""
    global _LLAMA_PROCESS, _LLAMA_STARTED_BY_US

    if not _LLAMA_STARTED_BY_US:
        return

    auto_stop = LLAMA_CFG.get("auto_stop", True)
    if not auto_stop:
        log("清理", "auto_stop=false，保留 llama-server 运行")
        return

    try:
        if _LLAMA_PROCESS is not None:
            _LLAMA_PROCESS.terminate()
            try:
                _LLAMA_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _LLAMA_PROCESS.kill()
            log("清理", "llama-server 已终止，显存和内存已释放")
    except Exception as e:
        log("清理", f"终止 llama-server 时出错: {e}")
    finally:
        _LLAMA_PROCESS = None
        _LLAMA_STARTED_BY_US = False


if __name__ == "__main__":
    try:
        main()
    finally:
        stop_llama_server()
