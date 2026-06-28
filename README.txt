IBL2SVS React/Tauri Desktop
===========================

用途
----
选择一个包含 .ibl / .kfb / .svs / .tif / .tiff 文件的输入文件夹，再选择输出文件夹，程序会批量转换为带金字塔层级的 Aperio SVS 或 Generic TIFF 文件。

架构
----
- React + Tauri v2 负责桌面界面、目录选择、打开输出目录/报告和前端状态。
- Python sidecar 负责 IBL/SVS/TIFF 转换，通过 JSON Lines 向 Tauri 回传日志、进度和结果。
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
   git tag v0.3.0
   git push origin main --tags

界面使用说明
------------
1. 选择输入文件夹。
2. 选择输出文件夹。
3. 选择输出格式：Generic TIFF 或 SVS。
4. 根据需要勾选“包含子文件夹”。
5. 点击“开始转换”。
6. 在任务流、进度摘要和 transcript 日志中查看转换状态。
7. 转换完成后可打开输出目录或 conversion_report.csv。

输出内容
--------
- 转换后的 .tif 或 .svs 文件。
- conversion_report.csv 批处理报告。
- ibl2svs_*.log 运行日志。

WSI TIFF/SVS 互转
-----------------
- 输出 SVS 时读取 .ibl / .kfb / .tif / .tiff。
- 输出 Generic TIFF 时读取 .ibl / .kfb / .svs。
- TIFF/SVS 输入采用保守重封装：重新生成金字塔，不迁移标注或完整私有 metadata。

注意事项
--------
- 第一版 Tauri 桌面端只允许一个转换任务同时运行。
- Python sidecar 通过取消事件安全停止，避免破坏 .part 清理逻辑。
- GitHub Actions 会构建未签名 macOS DMG；正式分发前仍建议补充 Apple Developer 签名和 notarization。
