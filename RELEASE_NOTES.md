# Release Notes

## Unreleased

- Generic TIFF 输出升级为 Pyramidal OME-TIFF：始终使用 BigTIFF + SubIFD，并以 `.ome.tif` 为固定后缀。
- 新增荧光原始 `TZCYX` OME 写出、多 Field 独立 Image、通道名称/颜色/波长/曝光与 SignificantBits 元数据。
- 新增 256×256 单通道荧光 SVS，以及事务式 AFI（XML + 多个单通道 SVS）文件集输出。
- 新增转换前批量预检、输出兼容数量、未知通道确认、任务级通道覆盖、混合批次兼容文件筛选与 skipped 报告。
- KFBF 2.1 Header 通道数组使用完整 64 位位置解引用；当前真实样本识别为 DAPI、蓝色显示色、曝光 85。
- 预检后文件变化或通道覆盖结构不匹配时明确停止，不继续使用陈旧定义。
- `.image` 原生保真转换：支持 64 位瓦片偏移、原生 8 级金字塔以及 thumbnail、macro、label RGB 资源。
- OME-TIFF 保留原生 256×256 JPEG 瓦片和无损附属 Image；明场 SVS 从厂家金字塔层独立生成兼容页面。
- 修复 KFBF 荧光 OME-TIFF、SVS 和 AFI 在 QuPath 中严重过曝：不再暴露 1×1、1×3 等占位层作为自动直方图来源。
- 修复厂家短 JPEG 被直接写入固定 256×256 tile 后在低倍率右侧产生条纹；完整瓦片继续字节直通，短瓦片补零后规范化写出。
- JPEG DCT 域无损轴转置改用进程内持久化 libjpeg-turbo 转换器，避免为每个瓦片启动子进程；动态库不可用时自动使用高质量重编码并记录转换模式。
- 修复大型 `.image` 写出 OME-TIFF 时首层长期无进度、ETA 持续增加的问题；无损 JPEG 转置改为有序并行，并按瓦片持续汇报进度。
- 支持 KFBF（`.kfbf`）切片变体：识别 `KFBF` 文件头和间接 JPEG 瓦片指针。
- Python sidecar 改为应用生命周期内常驻；预检最多并行读取 4 个文件，转换复用预检快照，不再重复承担冷启动和逐文件预检。
- 新增渐进式 WSI 类型/数量摘要、持续任务状态横幅和明确的引擎初始化状态，并在任务启动后锁定输出格式。
- 数字参数改为统一的加减 Stepper，荧光预设菜单改用 MIT 许可的 Radix UI Select；修复“自定义”选择、焦点和任务重置后的重新预检。
- 修复 macOS WKWebView 暗色模式仍使用浅色原生滚动轨道的问题；原生 color scheme、轨道和滑块现在与应用主题同步。
- 修复空闲状态隐藏任务横幅后 Transcript 被放入自动高度网格行、无法占满剩余空间的问题。
- 新增 `worker_startup_ms`、`inspection_ms`、`conversion_prepare_ms` 性能日志与前端/Rust/并行预检自动测试。

## 0.5.0

- 新增主图、预览图和金字塔独立 JPG/J2K 质量设置，默认值分别为 90、70、60。
- 新增“输出至新子文件夹”和“扫描子文件夹”任务选项，并统一任务选项样式。
- 改进 SVS 原生关联图像、旧版 IBL label、J2K 自动识别和大文件 BigTIFF 判定。

## 0.4.5

- 隐藏 Windows 正式版启动时附带的空白命令窗口，关闭应用不再依赖控制台窗口。
- 修复转换完成、报错或 worker 异常退出后的任务清理时序，并新增“重置任务”操作，使界面可可靠开始下一次转换。

## 0.4.4

- 支持 `.image` 金字塔索引中的零长度空白占位瓦片，按背景色写出，避免真实扫描文件因“瓦片长度无效”而转换失败。

## 0.4.3

- 修复 Windows 读取 `.image` 时整文件内存映射可能触发 `STATUS_IN_PAGE_ERROR (0xC0000006)` 并直接终止 worker 的问题，改为按偏移读取文件数据。

## 0.4.2

- 修复 Windows 本地代码页使 UTF-8 worker 协议中的中文路径被错误解码，导致现有输入目录被报告为无效的问题。

## 0.4.1

- 修复 Windows 路径传入 Python sidecar 时的 JSON Lines 转义问题，避免转换任务在开始前因 `Invalid \escape` 终止。

## 0.4.0

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
