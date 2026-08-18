镜渡 SlideBridge React/Tauri Desktop
=====================================

用途
----
选择一个包含 .ibl / .kfb / .kfbl / .kfbf / .kfba / .kfbx / .image / .svs / .tif / .tiff / .afi 文件的输入文件夹，再选择输出文件夹，程序会批量转换为 Pyramidal OME-TIFF、明场 SVS、单通道荧光 SVS 或 AFI 文件集。

架构
----
- React + Tauri v2 负责桌面界面、目录选择、打开输出目录/报告和前端状态。
- Python sidecar 负责 IBL/KFB/.image/SVS/TIFF/AFI 转换，通过 JSON Lines 向 Tauri 回传日志、进度和结果。
- 转换核心仍使用 ibl2svs.converter / writer / reader，不依赖 Tkinter。

运行环境
--------
Windows 10/11 x64，建议 16 GB 及以上内存。
正式打包产物内置 Python sidecar，无需用户安装 Python 运行时。

本地开发
--------
1. 安装 Python 依赖
   python -m pip install -r requirements.txt

2. 安装前端依赖
   cd desktop
   npm install

3. 启动 Tauri 开发版
   npm run tauri dev

打包成 Windows EXE
------------------
1. 在 Windows 上准备 Python 3.11+、Node.js、Rust 工具链。
2. 运行：
   build_windows.bat C:\Path\To\python.exe
3. 产物位于：
   desktop\src-tauri\target\release\bundle\nsis

GitHub Release
--------------
推送 v* 标签会触发 .github/workflows/release.yml，自动构建 Windows x64 NSIS 安装包和未签名 macOS DMG，并发布到 GitHub Release。

示例：
   git tag v0.5.0
   git push origin main --tags

界面使用说明
------------
1. 选择输入文件夹。
2. 选择输出文件夹。
3. 等待文件头预检完成，并查看每种输出格式的兼容文件数量。
4. 选择输出格式：Pyramidal OME-TIFF、明场 SVS、荧光 SVS 或 AFI。
5. 如有未知荧光通道，按序号 C1/C2 继续，或修改本任务的通道定义。
6. 混合批次存在不兼容文件时，明确选择“只转换兼容文件”或返回修改格式。
7. 根据需要勾选“扫描子文件夹”或“输出至新子文件夹”，然后点击“开始转换”。
8. 在任务流、进度摘要和 transcript 日志中查看转换状态。
9. 转换完成后可打开输出目录或 conversion_report.csv。

输出内容
--------
- 转换后的 .ome.tif、.svs，或 .afi + 通道 SVS 文件集。
- conversion_report.csv 批处理报告。
- slidebridge_*.log 运行日志。

WSI TIFF/SVS 互转
-----------------
- 明场输入可输出 OME-TIFF 或原有 240×240 Aperio SVS。
- 单通道 8-bit 二维荧光可输出 OME-TIFF、256×256 荧光 SVS 或 AFI。
- 多通道 8-bit 二维荧光可输出 OME-TIFF 或 AFI。
- 高位深或多 Field/Z/T 荧光只允许输出 OME-TIFF。
- 荧光金字塔不会输出 1×1、1×3 等厂家占位层；不足标准 tile 的原生 JPEG 会补齐后写出，以兼容 QuPath/Bio-Formats 的低倍率显示与自动亮度范围。
- TIFF/SVS 输入采用保守重封装：重新生成金字塔，不迁移标注或完整私有 metadata。

注意事项
--------
- 桌面端只允许一个预检或转换 worker 同时运行；输入目录变化会废弃旧预检并重新开始。
- Python sidecar 通过取消事件安全停止，避免破坏 .part 清理逻辑。
- 未知通道不会被猜测为 AF、DAPI 等染料；通道修改只影响输出元数据和显示色，不改变像素。
- GitHub Actions 会构建未签名 macOS DMG；正式分发前仍建议补充 Apple Developer 签名和 notarization。
