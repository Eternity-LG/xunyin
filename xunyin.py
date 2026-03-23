#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迅音 - 本地语音转文字工具
基于 OpenAI Whisper 的桌面应用
"""

import sys
import os
import threading
import wave
import tempfile
import pyperclip
from datetime import datetime

import numpy as np
import json
import pyaudio
import whisper
from scipy import signal
from pynput import mouse, keyboard as pynput_keyboard

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QComboBox, QStatusBar,
    QMessageBox, QFileDialog, QDialog, QLineEdit, QFormLayout,
    QCheckBox, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont


class ConfigManager:
    """配置管理器"""
    CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".xunyin")
    CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
    
    @classmethod
    def load(cls):
        """加载配置"""
        if os.path.exists(cls.CONFIG_FILE):
            try:
                with open(cls.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置失败: {e}")
        return {"model": "base"}
    
    @classmethod
    def save(cls, config):
        """保存配置"""
        try:
            os.makedirs(cls.CONFIG_DIR, exist_ok=True)
            with open(cls.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置失败: {e}")


class TranscriptionWorker(QThread):
    """后台转录线程"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, audio_path, model_name="base"):
        super().__init__()
        self.audio_path = audio_path
        self.model_name = model_name
        
    def run(self):
        try:
            import warnings
            warnings.filterwarnings("ignore")
            
            # 加载音频文件（不使用ffmpeg）
            import numpy as np
            import wave
            
            # 读取wav文件
            with wave.open(self.audio_path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                
                # 读取音频数据
                raw_data = wf.readframes(n_frames)
                
                # 转换为numpy数组
                if sample_width == 2:
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                elif sample_width == 1:
                    audio_data = np.frombuffer(raw_data, dtype=np.uint8)
                else:
                    raise ValueError(f"不支持的采样宽度: {sample_width}")
                
                # 转换为float32并归一化到[-1, 1]
                audio_data = audio_data.astype(np.float32) / 32768.0
                
                # 如果是立体声，转换为单声道
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                # 重采样到16kHz（如果必要）
                if sample_rate != 16000:
                    # 简单的线性插值重采样
                    from scipy import signal
                    audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))
            
            # 加载模型并转写
            model = whisper.load_model(self.model_name)
            result = model.transcribe(audio_data, language="zh", initial_prompt="请使用简体中文回答。")
            text = result["text"].strip()
            self.finished.emit(text)
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self.error.emit(f"{str(e)}\n\n详细错误:\n{error_detail}")


class AudioRecorder:
    """音频录制器"""
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        
    def start_recording(self):
        """开始录音"""
        self.frames = []
        self.is_recording = True
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
    def read_chunk(self):
        """读取音频块"""
        if self.stream and self.is_recording:
            data = self.stream.read(1024, exception_on_overflow=False)
            self.frames.append(data)
            
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        # 保存为临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.frames))
            
        return temp_path
        
    def cleanup(self):
        """清理资源"""
        self.audio.terminate()


# 配置文件路径
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".xunyin")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")


def load_config():
    """加载配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"model": "base"}  # 默认配置


def save_config(config):
    """保存配置"""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存配置失败: {e}")


class XunYinWindow(QMainWindow):
    """主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("迅音 - 语音转文字")
        self.setGeometry(100, 100, 400, 350)
        self.setMinimumSize(400, 350)
        self.setMaximumSize(400, 350)
        
        # 加载配置
        self.config = load_config()
        
        # 初始化组件
        self.recorder = AudioRecorder()
        self.worker = None
        self.temp_file = None
        
        self.setup_ui()
        
        # 检查模型
        self.check_model()
        
    def setup_ui(self):
        """设置UI - 400x500 布局"""
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # === 顶部区域：标题和模型选择 ===
        top_layout = QHBoxLayout()
        
        # 标题
        title_label = QLabel("迅音")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #333;")
        top_layout.addWidget(title_label)
        top_layout.addStretch()
        
        # 模型选择
        model_label = QLabel("模型:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium"])
        self.model_combo.setFixedWidth(80)
        # 加载保存的模型配置
        saved_model = self.config.get("model", "base")
        if saved_model in ["tiny", "base", "small", "medium"]:
            self.model_combo.setCurrentText(saved_model)
        else:
            self.model_combo.setCurrentText("base")
        # 模型改变时保存配置
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        top_layout.addWidget(model_label)
        top_layout.addWidget(self.model_combo)
        
        # 设置按钮
        self.settings_btn = QPushButton("⚙️")
        self.settings_btn.setFixedSize(32, 32)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.settings_btn.clicked.connect(self.open_settings)
        top_layout.addWidget(self.settings_btn)
        
        layout.addLayout(top_layout)
        
        # === 中间区域：录音按钮（圆形大按钮） ===
        btn_container = QWidget()
        btn_container.setFixedHeight(70)
        btn_layout_center = QVBoxLayout(btn_container)
        btn_layout_center.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.record_btn = QPushButton("按住说话")
        self.record_btn.setFixedSize(50, 50)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 11px;
                font-weight: bold;
                border: none;
                border-radius: 25px;
            }
            QPushButton:pressed {
                background-color: #f44336;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.record_btn.pressed.connect(self.start_recording)
        self.record_btn.released.connect(self.stop_recording)
        btn_layout_center.addWidget(self.record_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(btn_container)
        
        # === 文本输出区域 ===
        self.text_output = QTextEdit()
        self.text_output.setPlaceholderText("识别结果将显示在这里...")
        self.text_output.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                background-color: #f9f9f9;
            }
        """)
        self.text_output.setMinimumHeight(80)
        layout.addWidget(self.text_output)
        
        # === 底部按钮区域 ===
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.copy_btn = QPushButton("📋 复制")
        self.copy_btn.setFixedHeight(36)
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        self.copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        btn_layout.addWidget(self.copy_btn)
        
        self.save_btn = QPushButton("💾 保存")
        self.save_btn.setFixedHeight(36)
        self.save_btn.clicked.connect(self.save_to_file)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        btn_layout.addWidget(self.save_btn)
        
        self.clear_btn = QPushButton("🗑️ 清空")
        self.clear_btn.setFixedHeight(36)
        self.clear_btn.clicked.connect(self.clear_text)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        btn_layout.addWidget(self.clear_btn)
        
        layout.addLayout(btn_layout)
        
        # === 状态栏 ===
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("准备就绪，按住圆形按钮开始录音")
        
    def apply_config(self):
        """应用配置"""
        # 应用自动复制设置
        auto_copy = self.config.get("auto_copy", True)
        self.status_bar.showMessage(f"设置已更新 (自动复制: {'开启' if auto_copy else '关闭'})")
        
    def open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 保存新配置
            self.config = dialog.get_config()
            ConfigManager.save(self.config)
            # 应用新配置
            self.apply_config()
        
    def start_recording(self):
        """开始录音"""
        self.record_btn.setText("录音中... 松开结束")
        self.status_bar.showMessage("正在录音，请说话...")
        self.recorder.start_recording()
        
        # 启动录音循环线程
        self.recording_thread = threading.Thread(target=self.recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
    def recording_loop(self):
        """录音循环，持续读取音频数据"""
        while self.recorder.is_recording:
            self.recorder.read_chunk()
        
    def stop_recording(self):
        """停止录音并开始转录"""
        # 停止录音循环
        self.recorder.is_recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=1)
        
        self.record_btn.setText("按住说话")
        self.record_btn.setEnabled(False)
        self.status_bar.showMessage("正在识别，请稍候...")
        
        # 停止录音并获取文件路径
        self.temp_file = self.recorder.stop_recording()
        
        # 启动后台转录线程
        model_name = self.model_combo.currentText()
        self.worker = TranscriptionWorker(self.temp_file, model_name)
        self.worker.finished.connect(self.on_transcription_finished)
        self.worker.error.connect(self.on_transcription_error)
        self.worker.start()
        
    def on_transcription_finished(self, text):
        """转写完成回调"""
        print(f"识别结果: {text}")  # 调试输出
        self.text_output.clear()
        self.text_output.setPlainText(text)
        self.text_output.repaint()  # 强制刷新
        self.status_bar.showMessage("识别完成！")
        self.record_btn.setEnabled(True)
        
        # 自动复制到剪贴板
        if text:
            self.copy_to_clipboard_silent(text)
    
    def copy_to_clipboard_silent(self, text):
        """静默复制到剪贴板"""
        try:
            # 方法1: 使用 pyperclip
            pyperclip.copy(text)
            self.status_bar.showMessage("识别完成！已自动复制到剪贴板")
            return
        except:
            pass
        
        try:
            # 方法2: 使用 Windows 原生 API
            import ctypes
            
            if ctypes.windll.user32.OpenClipboard(None):
                try:
                    ctypes.windll.user32.EmptyClipboard()
                    text_bytes = text.encode('utf-16-le') + b'\x00\x00'
                    size = len(text_bytes)
                    h_mem = ctypes.windll.kernel32.GlobalAlloc(0x2000, size)
                    if h_mem:
                        p_mem = ctypes.windll.kernel32.GlobalLock(h_mem)
                        ctypes.memmove(p_mem, text_bytes, size)
                        ctypes.windll.kernel32.GlobalUnlock(h_mem)
                        ctypes.windll.user32.SetClipboardData(13, h_mem)
                    self.status_bar.showMessage("识别完成！已自动复制到剪贴板")
                    return
                finally:
                    ctypes.windll.user32.CloseClipboard()
        except:
            pass
        
        self.status_bar.showMessage("识别完成！（复制到剪贴板失败）")
            
        # 清理临时文件
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
                self.temp_file = None
            except:
                pass
        
    def on_transcription_error(self, error_msg):
        """转录错误回调"""
        self.status_bar.showMessage("识别失败，请查看详细错误信息")
        self.record_btn.setEnabled(True)
        self.show_error_dialog(error_msg)
        
        # 清理临时文件
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
                self.temp_file = None
            except:
                pass
        
    def copy_to_clipboard(self):
        """复制到剪贴板"""
        text = self.text_output.toPlainText()
        if text:
            try:
                pyperclip.copy(text)
                self.status_bar.showMessage("已复制到剪贴板")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"复制失败: {str(e)}")
        else:
            QMessageBox.information(self, "提示", "没有内容可复制")
            
    def save_to_file(self):
        """保存到文件"""
        text = self.text_output.toPlainText()
        if not text:
            QMessageBox.information(self, "提示", "没有内容可保存")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存文件", "", "文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.status_bar.showMessage(f"已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
                
    def clear_text(self):
        """清空文本"""
        self.text_output.clear()
        self.status_bar.showMessage("准备就绪，按住按钮开始录音")
        
    def show_error_dialog(self, error_msg):
        """显示可复制的错误对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("错误详情")
        dialog.setGeometry(100, 100, 600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # 错误提示标签
        label = QLabel("识别失败，错误详情如下（可直接复制）：")
        layout.addWidget(label)
        
        # 可复制的文本框
        text_edit = QTextEdit()
        text_edit.setPlainText(error_msg)
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(text_edit)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        
        # 复制按钮
        copy_btn = QPushButton("复制错误信息")
        copy_btn.clicked.connect(lambda: self.copy_error_text(text_edit.toPlainText()))
        btn_layout.addWidget(copy_btn)
        
        btn_layout.addStretch()
        
        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        dialog.exec()
    
    def copy_error_text(self, text):
        """复制错误文本到剪贴板"""
        try:
            pyperclip.copy(text)
            self.status_bar.showMessage("错误信息已复制到剪贴板")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"复制失败: {str(e)}")
    
    def on_model_changed(self, model_name):
        """模型改变时保存配置"""
        self.config["model"] = model_name
        ConfigManager.save(self.config)
        self.status_bar.showMessage(f"已切换到 {model_name} 模型")
    
    def check_model(self):
        """检查模型是否已下载"""
        import os
        model_name = self.model_combo.currentText()
        model_path = os.path.expanduser(f'~/.cache/whisper/{model_name}.pt')
        
        if not os.path.exists(model_path):
            self.status_bar.showMessage(f"正在首次下载 {model_name} 模型，请稍候...")
            try:
                # 预下载模型
                whisper.load_model(model_name)
                self.status_bar.showMessage("模型下载完成，准备就绪")
            except Exception as e:
                self.show_error_dialog(f"模型下载失败: {str(e)}\n\n"
                    f"请确保网络连接正常，或手动下载模型文件到:\n{model_path}")
                self.status_bar.showMessage("模型下载失败，请检查网络")
        else:
            self.status_bar.showMessage("准备就绪，按住按钮开始录音")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.recorder.cleanup()
        
        # 清理临时文件
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except:
                pass
                
        event.accept()


class HotkeyLineEdit(QLineEdit):
    """快捷键输入框 - 捕获键盘按键和鼠标侧键"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("点击此处，按下快捷键或鼠标侧键...")
        self.current_hotkey = ""
        self.mouse_listener = None
        self.is_listening = False
        
    def focusInEvent(self, event):
        """获得焦点时开始监听鼠标"""
        super().focusInEvent(event)
        self.start_mouse_listen()
        
    def focusOutEvent(self, event):
        """失去焦点时停止监听鼠标"""
        super().focusOutEvent(event)
        self.stop_mouse_listen()
        
    def start_mouse_listen(self):
        """开始监听鼠标侧键"""
        if self.is_listening:
            return
        self.is_listening = True
        self.setStyleSheet("background-color: #e3f2fd;")  # 蓝色背景提示正在监听
        
        def on_click(x, y, button, pressed):
            if not pressed:
                return True
            # 鼠标侧键: button.x1 (后退), button.x2 (前进)
            if button == mouse.Button.x1:
                self.current_hotkey = "MouseBack"
                self.setText("鼠标侧键-后退")
                return True  # 继续监听，允许再次修改
            elif button == mouse.Button.x2:
                self.current_hotkey = "MouseForward"
                self.setText("鼠标侧键-前进")
                return True  # 继续监听，允许再次修改
            return True
            
        def on_scroll(x, y, dx, dy):
            if dy > 0:
                self.current_hotkey = "MouseScrollUp"
                self.setText("鼠标滚轮-上")
            else:
                self.current_hotkey = "MouseScrollDown"
                self.setText("鼠标滚轮-下")
            return True  # 继续监听，允许再次修改
            
        self.mouse_listener = mouse.Listener(
            on_click=on_click,
            on_scroll=on_scroll
        )
        self.mouse_listener.start()
        
    def stop_mouse_listen(self):
        """停止监听鼠标"""
        self.is_listening = False
        self.setStyleSheet("")  # 恢复默认样式
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        
    def keyPressEvent(self, event):
        """捕获按键"""
        key = event.key()
        
        # 忽略单独的功能键
        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, 
                   Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            return
            
        # 构建快捷键字符串
        modifiers = []
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            modifiers.append("Ctrl")
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            modifiers.append("Shift")
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            modifiers.append("Alt")
        if event.modifiers() & Qt.KeyboardModifier.MetaModifier:
            modifiers.append("Win")
            
        # 获取按键名称
        key_name = Qt.Key(key).name.replace("Key_", "")
        
        # 特殊按键处理
        special_keys = {
            "Space": "Space",
            "Return": "Enter",
            "Escape": "Esc",
            "Tab": "Tab",
            "Backspace": "Backspace",
            "Delete": "Delete",
            "Insert": "Insert",
            "Home": "Home",
            "End": "End",
            "PageUp": "PageUp",
            "PageDown": "PageDown",
            "Up": "Up",
            "Down": "Down",
            "Left": "Left",
            "Right": "Right",
        }
        
        if key_name in special_keys:
            key_name = special_keys[key_name]
        elif key_name.startswith("F") and key_name[1:].isdigit():
            pass  # F1-F12 保持原样
        elif len(key_name) == 1:
            pass  # 字母数字保持原样
        else:
            key_name = key_name  # 其他按键
            
        # 组合快捷键
        if modifiers:
            self.current_hotkey = "+".join(modifiers + [key_name])
        else:
            self.current_hotkey = key_name
            
        self.setText(self.current_hotkey)
        event.accept()
        
        # 键盘事件后继续监听鼠标（允许切换回鼠标快捷键）
        if not self.is_listening:
            self.start_mouse_listen()
        
    def get_hotkey(self):
        """获取当前快捷键"""
        return self.current_hotkey or self.text()
        
    def set_hotkey(self, hotkey):
        """设置快捷键"""
        self.current_hotkey = hotkey
        # 转换显示名称
        display_names = {
            "MouseBack": "鼠标侧键-后退",
            "MouseForward": "鼠标侧键-前进",
            "MouseScrollUp": "鼠标滚轮-上",
            "MouseScrollDown": "鼠标滚轮-下",
        }
        self.setText(display_names.get(hotkey, hotkey))


class SettingsDialog(QDialog):
    """设置对话框"""
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config.copy()
        self.setWindowTitle("设置")
        self.setGeometry(100, 100, 400, 300)
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # === 快捷键设置 ===
        hotkey_group = QGroupBox("快捷键设置 (点击输入框，按下快捷键)")
        hotkey_layout = QFormLayout(hotkey_group)
        
        # 录音快捷键
        self.record_hotkey = HotkeyLineEdit()
        self.record_hotkey.set_hotkey(self.config.get("record_hotkey", "F6"))
        hotkey_layout.addRow("录音快捷键:", self.record_hotkey)
        
        # 复制快捷键
        self.copy_hotkey = HotkeyLineEdit()
        self.copy_hotkey.set_hotkey(self.config.get("copy_hotkey", "Ctrl+C"))
        hotkey_layout.addRow("复制结果快捷键:", self.copy_hotkey)
        
        layout.addWidget(hotkey_group)
        
        # === 其他设置 ===
        other_group = QGroupBox("其他设置")
        other_layout = QVBoxLayout(other_group)
        
        # 自动复制开关
        self.auto_copy = QCheckBox("识别完成后自动复制到剪贴板")
        self.auto_copy.setChecked(self.config.get("auto_copy", True))
        other_layout.addWidget(self.auto_copy)
        
        # 自动保存开关
        self.auto_save = QCheckBox("识别完成后自动保存到文件")
        self.auto_save.setChecked(self.config.get("auto_save", False))
        other_layout.addWidget(self.auto_save)
        
        layout.addWidget(other_group)
        
        # === 按钮 ===
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("保存")
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        save_btn.clicked.connect(self.save_settings)
        btn_layout.addWidget(save_btn)
        
        layout.addLayout(btn_layout)
        
    def save_settings(self):
        """保存设置"""
        self.config["record_hotkey"] = self.record_hotkey.get_hotkey() or "F6"
        self.config["copy_hotkey"] = self.copy_hotkey.get_hotkey() or "Ctrl+C"
        self.config["auto_copy"] = self.auto_copy.isChecked()
        self.config["auto_save"] = self.auto_save.isChecked()
        self.accept()
        
    def get_config(self):
        """获取配置"""
        return self.config


def main():
    app = QApplication(sys.argv)
    window = XunYinWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
