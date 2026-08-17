# 镜渡 SlideBridge

[![Release](https://github.com/Ruri0319/SlideBridge/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Ruri0319/SlideBridge/actions/workflows/release.yml)

**镜渡 SlideBridge** 是一个通用病理 Whole-Slide Image (WSI) 转换工具，用于把 `.ibl`、`.kfb/.kfbl/.kfbf/.kfba/.kfbx`、`.image`、`.svs`、`.tif/.tiff`、`.afi` 批量转换为 **Pyramidal OME-TIFF**、明场 **Aperio SVS**、单通道荧光 SVS 或 AFI 文件集。

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

| 输入数据 | OME-TIFF | 明场 SVS | 荧光 SVS | AFI |
|----|----:|----:|----:|----:|
| 明场 / HE | 支持 | 支持 | 禁止 | 禁止 |
| 8-bit、C=1、Field/Z/T=1 荧光 | 支持 | 禁止 | 支持 | 支持 |
| 8-bit、C>1、Field/Z/T=1 荧光 | 支持 | 禁止 | 禁止 | 支持 |
| 高位深或多 Field/Z/T 荧光 | 支持 | 禁止 | 禁止 | 禁止 |

程序会在转换前异步预检文件头，显示模态、C/Z/T/Field、位深、codec、通道定义来源及每种输出格式的兼容数量。未知通道默认按 C1/C2 编号，不根据通道数量、像素颜色、扩展名或文件名猜测 DAPI、AF 等染料。用户可在当前任务中覆盖通道名称、显示色和可选波长；覆盖只影响输出元数据与 AFI 文件名，不改变像素。

OME-TIFF 会读取 OME-XML 中的 Channel、Fluor、Color、波长和曝光；单通道荧光 SVS 会读取 Aperio `Dye`、`DisplayColor` 和曝光字段；AFI 会按 XML 中的相对 Path 组合通道，并在目录预检时自动排除其已引用的子 SVS，避免重复转换。明场 HE/RGB 文件不显示荧光通道定义。

KFB 家族按内容签名识别真实容器和版本。当前真实 KFBF 2.1 灰度 JPEG 样本可从厂家 Header 识别 DAPI、蓝色显示色、曝光信息和 16 层原生金字塔；其他 KFB/KFBA/KFBX 路径在没有真实样本前标记为 `static_unverified`。未知版本或布局会明确失败，不会套用固定偏移猜测转换。荧光输出会保留到首个不大于 2×2 tile 的有效概览层，忽略 1×1、1×3 等会破坏 QuPath 自动显示范围的厂家占位层；报告仍记录完整源层尺寸。

`.image` 会优先保留厂家原生资源：解析 64 位资源偏移和 8 级列优先索引，OME-TIFF 直接写出原生金字塔层，明场 SVS 从最接近的厂家层生成兼容页面；thumbnail、macro、label 以原始尺寸写出。主层 JPEG 需要轴转置，若运行环境提供 `jpegtran`（或将其路径放入 `IBL2SVS_JPEGTRAN`），会使用 DCT 域无损转置，否则使用高质量重编码并在转换结果中标记。

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

result = convert_file("sample.ibl", "output.ome.tif", ConvertOptions(output_format="ome_tiff"))
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

预检和荧光 AFI：

```python
from ibl2svs import ConvertOptions, convert_file, inspect_file

inspection = inspect_file("sample.kfbf")
print(inspection.source_modality, inspection.allowed_output_formats)

result = convert_file(
    "sample.kfbf",
    "output.afi",
    ConvertOptions(output_format="afi"),
)
print(result.output_files)
```

## 桌面端架构

```text
React / TypeScript UI
  |
  |- Tauri commands: start_inspection / start_conversion / cancel_conversion / worker_status
  |- Tauri plugins: dialog / opener / shell
  v
Python sidecar: slidebridge-worker
  |
  |- stdin: JSON Lines inspect / start / cancel / ping
  |- stdout: JSON Lines inspection_* / log / progress / done / error
  v
ibl2svs.converter.convert_folder()
```

- 前端文件位于 `desktop/src/`。
- Tauri/Rust 桥接位于 `desktop/src-tauri/`。
- Python sidecar 入口为 `ibl2svs/backend_worker.py` 和 `worker_main.py`。
- Sidecar PyInstaller 配置为 `SlideBridgeWorker.spec`。

## 输出格式

### Pyramidal OME-TIFF (`.ome.tif`)

- 始终使用 BigTIFF 和 SubIFD；缩小层不再作为并列顶层页面。
- 明场主 Image 为 uint8 RGB `YXS`；可直通的厂家 JPEG 保留原始压缩数据。
- 荧光只保存原始 `TZCYX` 通道平面，每个 Field 是独立 OME Image，并保留原始 dtype、SignificantBits、物理尺寸和通道元数据。
- 可直通的完整原生 JPEG 保持字节不变；尺寸不足一个标准 tile 的厂家 JPEG 会补背景并重新编码为完整 tile，避免 Bio-Formats/QuPath 在低倍率产生条纹。
- thumbnail、macro、label 是独立命名 OME Image，使用 LZW 无损压缩。

### 明场 SVS (`.svs`)

- 保留原有 Aperio-compatible 写出后端、240×240 JPEG 瓦片、质量设置和页面布局。
- 仅接受明场输入；不会把荧光通道合成为明场 SVS。

### 单通道荧光 SVS (`.svs`)

- 仅接受 8-bit、C=1、Field/Z/T=1 且具有独立灰度原生平面的荧光输入。
- 256×256 灰度 JPEG 瓦片；满足条件时直接重封装原始 JPEG。
- 金字塔停止在可用于稳定直方图统计的概览层，避免 QuPath 用 1×1、1×3 占位层把显示窗错误收窄。
- ImageDescription 写入 Dye、DisplayColor、MPP、AppMag 及源文件存在的曝光/波长。

### AFI (`.afi` + `.svs`)

- AFI 是一个 UTF-8 XML 和一组完整单通道荧光 SVS 的文件集。
- 单通道也可生成 AFI，例如 `sample.afi` + `sample_C01_DAPI.svs`。
- 整套文件先写临时文件并全部验证，成功后统一发布；失败或取消不会留下残缺文件集。

## 项目结构

```text
ibl2svs/
├── backend_worker.py    # Python sidecar JSONL 预检/转换 worker
├── converter.py         # 文件扫描、单/批量转换、CSV 报告
├── inspection.py        # 模态、通道和输出资格预检
├── ome_tiff_writer.py   # Pyramidal OME-TIFF 写出
├── fluorescence_svs_writer.py # 单通道荧光 SVS
├── afi_writer.py        # AFI 文件集事务写出
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
- AFI 与荧光 SVS 只承担 8-bit 二维阅片交换；高位深、多 Field/Z/T 数据必须使用 OME-TIFF。
- Bio-Formats 仅用于开发验收，不作为运行时依赖或随软件分发。
- Tauri app 不依赖用户本机 Python，但构建机需要 Python、Node.js、Rust 和 PyInstaller。

## 许可证

本项目使用 MIT License，详见 `LICENSE`。

## 免责声明

本项目仅用于学术研究与技术交流目的。IBL/KFB 等文件格式可能包含厂商专有结构，本项目通过逆向工程方式实现格式解析与转换，不包含厂商私有代码、SDK 或商业机密。使用者应自行确保转换行为符合相关法律法规及与设备厂商的协议约定。
