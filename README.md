# 镜渡 SlideBridge

[![Release](https://github.com/Ruri0319/SlideBridge/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Ruri0319/SlideBridge/actions/workflows/release.yml)

**镜渡 SlideBridge** 是一个通用病理 Whole-Slide Image (WSI) 转换工具，用于把 `.ibl`、`.kfb/.kfbf`、`.image`、`.svs`、`.tif/.tiff` 批量转换为标准 **Aperio SVS** 或 **Generic Pyramidal TIFF (BigTIFF)**。

项目包含两层入口：

- **桌面应用**：React + Tauri v2，面向日常批量转换。
- **Python API / CLI**：面向脚本、调试和自动化流程。

Python 转换核心负责 WSI 读取、金字塔重建、JPEG tile 编码和 CSV 报告；Tauri 桌面端通过 PyInstaller sidecar 调用转换核心，因此正式桌面发行包不要求用户额外安装 Python。

## 下载

Windows 和 macOS 构建产物会发布在 [GitHub Releases](https://github.com/Ruri0319/SlideBridge/releases)。

- Windows x64：NSIS 安装包。
- macOS：ad-hoc 签名 DMG。
- 每个 release 附带 `SHA256SUMS.txt` 校验和。

macOS 当前未做 Apple Developer ID 签名和 notarization。应用包会在构建时进行完整的 ad-hoc 签名，首次运行时仍可能需要在系统安全设置中手动允许打开。如果 Gatekeeper 将 GitHub 下载的应用标记为“已损坏”，可在把应用拖入“应用程序”后执行：

```bash
xattr -dr com.apple.quarantine /Applications/SlideBridge.app
open /Applications/SlideBridge.app
```

## 支持格式

| 输入 | 输出 Generic TIFF | 输出 SVS | 说明 |
|----|----|----|----|
| `.ibl` | 支持 | 支持 | 读取厂商 IBL SQLite/tile 数据 |
| `.kfb/.kfbf` | 支持 | 支持 | 支持普通 KFB 和 KFBF JPEG tile 结构 |
| `.image` | 支持 | 支持 | 读取已验证的私有 JPEG 瓦片金字塔结构 |
| `.svs` | 支持 | 不适用 | SVS 转 Generic Pyramidal TIFF |
| `.tif/.tiff` | 不适用 | 支持 | Generic TIFF 转 Aperio-compatible SVS |

KFB 支持采用保守重封装路径：`decode -> rebuild pyramid -> re-encode`。当前不会迁移标注或完整厂商私有 metadata；遇到新的 KFB 变体时可能需要补充解析逻辑，欢迎提交修改意见。

`.image` 支持会优先保留厂家原生资源：解析 64 位资源偏移和 8 级列优先索引，Generic TIFF 直接写出原生金字塔层，SVS 从最接近的厂家层生成兼容页面；thumbnail、macro、label 以原始尺寸写出。主层 JPEG 需要轴转置，若运行环境提供 `jpegtran`（或将其路径放入 `IBL2SVS_JPEGTRAN`），会使用 DCT 域无损转置，否则回退到高质量重编码并在转换结果中标记。机构、病例号、设备号和扫描时间会作为扫描 metadata 读取，但不会原样写回厂商私有结构。

## 快速开始

### 桌面开发版

```bash
python -m pip install -r requirements.txt
cd desktop
npm install
npm run tauri dev
```

`tauri dev` 会先用当前已安装依赖的 Python 自动构建独立 sidecar；最终 `.app` 不依赖系统 `python` 命令。Tauri 开发和打包需要本机安装 Rust/Cargo。Python 转换逻辑本身不需要 Rust。

### Python API

```bash
python -m pip install -r requirements.txt
```

```python
from ibl2svs import ConvertOptions, convert_file

result = convert_file("sample.ibl", "output.tif", ConvertOptions())
print(result.success, result.duration_sec)
```

```python
from ibl2svs import ConvertOptions, convert_file

result = convert_file(
    "sample.kfb",
    "output.svs",
    ConvertOptions(output_format="svs"),
)
print(result.success, result.backend)
```

## 桌面端架构

```text
React / TypeScript UI
  |
  |- Tauri commands: start_conversion / cancel_conversion / worker_status
  |- Tauri plugins: dialog / opener / shell
  v
Python sidecar: slidebridge-worker
  |
  |- stdin: JSON Lines start / cancel / ping
  |- stdout: JSON Lines ready / log / progress / done / error
  v
ibl2svs.converter.convert_folder()
```

- 前端文件位于 `desktop/src/`。
- Tauri/Rust 桥接位于 `desktop/src-tauri/`。
- Python sidecar 入口为 `ibl2svs/backend_worker.py` 和 `worker_main.py`。
- Sidecar PyInstaller 配置为 `SlideBridgeWorker.spec`。

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

## 项目结构

```text
ibl2svs/
├── backend_worker.py    # Python sidecar JSONL worker
├── converter.py         # 文件扫描、单/批量转换、CSV 报告
├── kfb_source.py        # KFB JPEG 瓦片读取源
├── punuoxi_source.py    # 私有 .image JPEG 金字塔读取源
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
├── test_kfb_source.py
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
2. 使用 PyInstaller 构建 `slidebridge-worker.exe`。
3. 复制 sidecar 到 `desktop\src-tauri\binaries\slidebridge-worker-x86_64-pc-windows-msvc.exe`。
4. 执行 `npm ci`。
5. 执行 `npm run tauri build`。

Tauri NSIS 安装包输出在：

```text
desktop\src-tauri\target\release\bundle\nsis
```

## GitHub Release

仓库已配置 `.github/workflows/release.yml`。推送 `v*` 标签时会自动构建并发布：

- Windows x64：NSIS 安装包。
- macOS：ad-hoc 签名 DMG。
- `SHA256SUMS.txt`：发布文件校验和。


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

- GitHub Actions 会对 macOS 应用进行完整的 ad-hoc 签名和校验；要消除首次启动安全提示，仍需补充 Apple Developer ID 签名和 notarization。
- KFB 解析目前覆盖当前已验证的 KFB 结构；不同厂商或不同版本 KFB 可能需要进一步适配。
- Tauri app 不依赖用户本机 Python，但构建机需要 Python、Node.js、Rust 和 PyInstaller。

## 许可证

本项目使用 MIT License，详见 `LICENSE`。

## 免责声明

本项目仅用于学术研究与技术交流目的。IBL/KFB 等文件格式可能包含厂商专有结构，本项目通过逆向工程方式实现格式解析与转换，不包含厂商私有代码、SDK 或商业机密。使用者应自行确保转换行为符合相关法律法规及与设备厂商的协议约定。
