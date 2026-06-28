# IBL2SVS

**IBL2SVS** 将包含 `.ibl`、`.kfb`、`.svs`、`.tif/.tiff` 病理切片文件的文件夹批量转换为标准 Whole-Slide Image (WSI) 格式：**Aperio SVS** 或 **Generic Pyramidal TIFF (BigTIFF)**。

当前桌面端已迁移为 **React + Tauri v2**：Tauri 负责桌面窗口、目录选择、打开输出目录/报告和前端状态；Python sidecar 负责 IBL/SVS/TIFF 转换。

## 快速开始

### Python API

```bash
python -m pip install -r requirements.txt
```

```python
from ibl2svs import convert_file, ConvertOptions

result = convert_file("sample.ibl", "output.tif", ConvertOptions())
print(result.success, result.duration_sec)
```

```python
from ibl2svs import convert_file, ConvertOptions

result = convert_file("sample.ibl", "output.svs", ConvertOptions(output_format="svs"))
print(result.success, result.backend)
```

### Tauri Desktop

```bash
python -m pip install -r requirements.txt
cd desktop
npm install
npm run tauri dev
```

> Tauri 开发和打包需要本机安装 Rust/Cargo。Python 转换逻辑本身不需要 Rust。

## 桌面端架构

```text
React / TypeScript UI
  │
  ├─ Tauri commands: start_conversion / cancel_conversion / worker_status
  │
  ├─ Tauri plugins: dialog / opener / shell
  │
  ▼
Python sidecar: ibl2svs-worker
  │
  ├─ stdin: JSON Lines start / cancel / ping
  ├─ stdout: JSON Lines ready / log / progress / done / error
  ▼
ibl2svs.converter.convert_folder()
```

- 前端文件位于 `desktop/src/`。
- Tauri/Rust 桥接位于 `desktop/src-tauri/`。
- Python sidecar 入口为 `ibl2svs/backend_worker.py` 和 `worker_main.py`。
- Sidecar PyInstaller 配置为 `IBL2SVSWorker.spec`。

## 输出格式

### Generic TIFF (`.tif`)

- 密集金字塔，逐级 2x 降采样直至短边 < 512 px。
- JPEG tile，默认 `256x256`，质量 `90`。
- BigTIFF 容器，支持超大文件。
- 兼容 OpenSlide / QuPath / Bio-Formats。

### SVS (`.svs`)

- Aperio-compatible SVS。
- Classic TIFF 默认行为。
- 主图、缩略图、动态金字塔、label、macro 页面。
- RGB Adobe JPEG tile + JPEGTables 共享表。
- OpenSlide / QuPath 兼容。

### KFB/TIFF/SVS 互转

- `.svs -> .tif` 输出 Generic Pyramidal TIFF。
- `.tif/.tiff -> .svs` 输出 Aperio-compatible SVS。
- `.kfb -> .tif/.svs` 输出 Generic Pyramidal TIFF 或 Aperio-compatible SVS。
- 第一版采用保守重封装：decode -> rebuild pyramid -> re-encode，不迁移标注或完整私有 metadata。

## 项目结构

```text
ibl2svs/
├── backend_worker.py    # Python sidecar JSONL worker
├── converter.py         # 文件扫描、单/批量转换、CSV 报告
├── kfb_source.py        # KFB JPEG 瓦片读取源
├── models.py            # ConvertOptions / ConvertResult / BatchResult
├── reader.py            # IBL SQLite 读取
├── tiff_source.py       # WSI TIFF/SVS tile/strip 读取源
├── writer.py            # TIFF/SVS 流式写出核心
└── ...

desktop/
├── src/                 # React / TypeScript UI
└── src-tauri/           # Tauri v2 Rust shell

tests/
├── test_backend_worker.py
├── test_converter.py
├── test_writer.py
├── test_tiff_source.py
└── ...
```

## 本地验证

```bash
python -m compileall -q ibl2svs tests
python -m pytest tests/ -q
cd desktop && npm run build
```

Rust/Tauri 验证：

```bash
cd desktop/src-tauri
cargo check
cargo test
```

## Windows 打包

```bat
build_windows.bat C:\Path\To\python.exe
```

构建脚本会：

1. 安装 Python 依赖。
2. 使用 PyInstaller 构建 `ibl2svs-worker.exe`。
3. 复制 sidecar 到 `desktop\src-tauri\binaries\ibl2svs-worker-x86_64-pc-windows-msvc.exe`。
4. 执行 `npm ci`。
5. 执行 `npm run tauri build`。

Tauri NSIS 安装包输出在：

```text
desktop\src-tauri\target\release\bundle\nsis
```

## GitHub Release

仓库已配置 `.github/workflows/release.yml`。推送 `v*` 标签时会自动构建并发布：

- Windows x64：NSIS 安装包。
- macOS：未签名 DMG。
- `SHA256SUMS.txt`：发布文件校验和。

发布示例：

```bash
git tag v0.3.0
git push origin main --tags
```

也可以在 GitHub Actions 页面手动运行 `Release` workflow，并填写要创建或更新的 tag。

## 依赖

Python:

| 包 | 用途 |
|----|------|
| `numpy` | 数组操作 |
| `Pillow` | 图像解码、降采样、label 合成 |
| `tifffile` | TIFF/SVS 读写 |
| `imagecodecs` | JPEG 编码/解码 |
| `psutil` | 内存与 CPU 监控 |
| `pyvips[binary]` | Generic TIFF 可选后端 |
| `pyinstaller` | Python sidecar 打包 |

Desktop:

| 包 | 用途 |
|----|------|
| React / TypeScript / Vite | 前端界面 |
| Tauri v2 | 桌面 shell |
| Tauri dialog/opener/shell plugins | 目录选择、打开文件、sidecar |
| lucide-react | 图标 |

## 已知限制

- 第一版 Tauri 桌面端只支持一个转换任务同时运行。
- GitHub Actions 会构建未签名 macOS DMG；正式分发前仍建议补充 Apple Developer 签名和 notarization。
- Tauri app 不依赖用户本机 Python，但构建机需要 Python、Node.js、Rust 和 PyInstaller。

## 免责声明

本项目仅用于学术研究与技术交流目的。IBL 文件格式为厂商专有格式，本项目通过逆向工程方式实现格式解析与转换，不包含厂商私有代码、SDK 或商业机密。使用者应自行确保转换行为符合相关法律法规及与设备厂商的协议约定。
