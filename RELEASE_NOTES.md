# Release Notes

## Unreleased

- 项目正式更名为 **镜渡 SlideBridge**，定位为通用 WSI 格式转换工具。
- 更新桌面应用、安装包、运行横幅、日志及 GitHub Release 产物品牌。
- 保留 Python 包导入路径 `ibl2svs` 和旧版构建环境变量作为兼容接口。

## 0.3.0

- **桌面端架构迁移**：
  - 用 React + Tauri v2 替换 Tkinter GUI
  - Python 转换核心改为 PyInstaller sidecar，通过 JSON Lines 向 Tauri 回传日志、进度和结果
  - Windows 打包链路改为先构建 Python worker，再执行 Tauri NSIS 打包
- **重写 SVS 写出路径**：零临时文件、单次 IBL 读取、流式写出 Aperio SVS
  - 移除旧 pyvips-hybrid / experimental-fallback 双路径
  - 新增 `StripDownsampleDrive` 同时执行主图 tile 迭代与 4× 内存降采样
  - 金字塔生成改为纯内存操作（PIL resize），不依赖 pyvips
- **JPEG 剥离编码 + JPEGTables 共享表**：
  - 每 tile 编码为 RGB Adobe JPEG（82,71,66='R','G','B'），无 YCbCr 色彩空间转换
  - tile 数据仅含 SOI+SOF0+SOS+data，DQT/DHT 表写入 JPEGTables TIFF 标签 (347)
  - OpenSlide 解码时所有 tile 复用同一 libjpeg 解压器，大幅降低内存占用
- **动态金字塔层级**：从 4× 起逐级翻倍直至短边 < 512 px，典型 158k×61k 图像生成 5 级金字塔
  - 最低层仅 ~55 tiles（vs 旧版 2,739 tiles），QuPath 概览图加载量减少 50×
- **新增原生 Label / Macro 图像支持**：优先使用 `tbl_airimg_info` 中的 JPEG 概览图像
  - `IBLSlide.get_label_image()` 和 `get_scan_metadata()` 新增
  - label 页面叠加扫描设备、时间、MPP 等元数据
- **旧 GUI 增强记录**：
  - 三级进度跟踪（子任务/当前文件/总体批次），带实时 ETA
  - 阶段指示灯（解析 IBL → 构建主图 → 生成金字塔 → 生成附属图像 → 写出文件）
  - 后端/内存/速度/时间实时刷新
  - 修复 `batch_finished_at` 缺失导致的 UI 刷新循环崩溃
  - 添加 `_refresh_ui` / `_drain_queue` 防御性异常保护
- **清理 15 个废弃函数/类**（`_write_svs_hybrid`、`VipsImageSource` 等），净减少 ~350 行

## 0.2.0

- 新增 Windows 批量转换入口
- 支持 `.ibl/.IBL` 扫描、批处理和 `conversion_report.csv`
- 新增 PyInstaller spec 与 GitHub Actions Windows 打包流程
- 新增应用版本/构建元数据
- 新增取消转换时的半成品输出清理逻辑
- 新增 writer 集成测试与桌面端冒烟测试

## 已知限制

- 真实 `.ibl` 的最终验收需在本地 Windows 环境完成
- 不处理标注迁移
- 不做断点续转和并行转换
- pyvips/libvips 为 Generic TIFF 可选依赖（未安装时回退到 tifffile 路径）
