# 迅音 - 本地语音转文字工具

基于 OpenAI Whisper 的桌面语音转文字应用，完全本地运行，保护隐私。

## 功能特点

- 🎤 **按住说话** - 简单易用的录音方式
- 🧠 **本地识别** - 使用 Whisper 模型，无需联网
- 📋 **自动复制** - 识别结果自动复制到剪贴板
- 💾 **保存文件** - 支持保存识别结果为文本文件
- 🎯 **多模型选择** - tiny/base/small/medium 可选

## 安装依赖

```bash
pip install -r requirements.txt
```

**注意：**
- 首次运行会自动下载 Whisper 模型（base 模型约 74MB）
- 需要安装 PyAudio，Windows 用户可能需要安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## 运行程序

```bash
python xunyin.py
```

或使用启动脚本：

```bash
# Windows
start.bat

# 或 PowerShell
./start.ps1
```

## 模型说明

| 模型 | 大小 | 速度 | 准确率 | 适用场景 |
|------|------|------|--------|----------|
| tiny | 39MB | 最快 | 一般 | 快速测试 |
| base | 74MB | 快 | 良好 | **推荐日常使用** |
| small | 244MB | 中等 | 较好 | 追求准确率 |
| medium | 769MB | 慢 | 很好 | 高质量需求 |

## 使用说明

1. 选择适合的模型（首次使用建议 base）
2. **按住** "按住说话" 按钮开始录音
3. **松开** 按钮结束录音，自动开始识别
4. 识别结果会自动复制到剪贴板
5. 可使用 "复制到剪贴板"、"保存到文件" 或 "清空" 按钮

## 注意事项

- 确保麦克风正常工作
- 首次使用需要下载模型，请保持网络连接
- 识别完成后模型会缓存在本地，下次启动更快

## 技术栈

- Python 3.8+
- PyQt6 - GUI 框架
- OpenAI Whisper - 语音识别
- PyAudio - 音频录制

## 许可证

MIT License
