import sys
import os
import cv2
import json
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QRadioButton,
    QButtonGroup, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QSplitter, QMessageBox, QProgressBar, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap

# Импорт детектора из нашего модуля
from defect_detector import (
    DetectorParams, DefectDetectionPipeline, Defect, ProcessingStats
)


class ProcessingThread(QThread):
    """Поток для выполнения детекции без блокировки интерфейса"""
    frame_processed = Signal(np.ndarray, list, ProcessingStats)  # кадр, дефекты, статистика
    processing_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, params: DetectorParams):
        super().__init__()
        self.params = params
        self._is_running = False

    def run(self):
        self._is_running = True
        try:
            # Создаём пайплайн с колбэком для передачи кадров в GUI
            pipeline = DefectDetectionPipeline(
                self.params,
                frame_callback=self._on_frame
            )
            # Отключаем собственное окно пайплайна, т.к. будем показывать в Qt
            pipeline.params.show_preview = False
            pipeline.run()
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self._is_running = False
            self.processing_finished.emit()

    def _on_frame(self, annotated_frame: np.ndarray, defects: list, stats: ProcessingStats):
        """Колбэк вызывается пайплайном для каждого обработанного кадра"""
        if not self._is_running:
            return
        self.frame_processed.emit(annotated_frame, defects, stats)

    def stop(self):
        self._is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Детектор дефектов лазерной линии")
        self.setMinimumSize(1200, 900)

        # Переменные
        self.current_params = DetectorParams()
        self.processing_thread = None
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._capture_camera_frame)

        # Главный виджет и layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая панель управления (виджет)
        left_panel = self._create_control_panel()
        left_panel.setMaximumWidth(400)

        # Правая панель с видео и логом
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.video_label = QLabel("Видео не запущено")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setMinimumSize(640, 480)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)

        right_layout.addWidget(self.video_label, 1)
        right_layout.addWidget(self.log_text, 0)

        # Сплиттер для разделения левой и правой панелей
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        main_layout.addWidget(splitter)

        # Загрузка параметров по умолчанию
        self._load_default_params()
        self._update_params_from_ui()

        self.statusBar().showMessage("Готов")
            
    def _create_control_panel(self):
        """Создаёт левую панель с элементами управления (с прокруткой)"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)

        # === Источник видео ===
        source_group = QGroupBox("Источник видео")
        source_layout = QVBoxLayout()

        self.radio_file = QRadioButton("Видеофайл")
        self.radio_camera = QRadioButton("Камера")
        self.radio_file.setChecked(True)
        self.source_group_btn = QButtonGroup()
        self.source_group_btn.addButton(self.radio_file, 0)
        self.source_group_btn.addButton(self.radio_camera, 1)
        source_layout.addWidget(self.radio_file)
        source_layout.addWidget(self.radio_camera)

        self.file_path_edit = QLineEdit("test_laser_defects.mp4")
        self.browse_btn = QPushButton("Обзор...")
        self.browse_btn.clicked.connect(self._browse_file)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_btn)
        source_layout.addLayout(file_layout)

        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 10)
        self.camera_id_spin.setValue(0)
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("ID камеры:"))
        camera_layout.addWidget(self.camera_id_spin)
        source_layout.addLayout(camera_layout)

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # === Калибровка камеры ===
        calib_group = QGroupBox("Калибровка камеры")
        calib_layout = QFormLayout()
        self.calib_path_edit = QLineEdit()
        self.calib_path_edit.setPlaceholderText("Не задана")
        self.browse_calib_btn = QPushButton("Обзор...")
        self.browse_calib_btn.clicked.connect(self._browse_calib)
        calib_row = QHBoxLayout()
        calib_row.addWidget(self.calib_path_edit)
        calib_row.addWidget(self.browse_calib_btn)
        calib_layout.addRow("JSON файл:", calib_row)
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)

        # === Параметры лазера ===
        laser_group = QGroupBox("Лазерная линия")
        laser_layout = QFormLayout()

        self.laser_channel_spin = QSpinBox()
        self.laser_channel_spin.setRange(0, 2)
        self.laser_channel_spin.setValue(2)
        self.laser_channel_spin.setToolTip("0=синий, 1=зелёный, 2=красный")
        laser_layout.addRow("Цветовой канал:", self.laser_channel_spin)

        self.min_brightness_spin = QSpinBox()
        self.min_brightness_spin.setRange(0, 255)
        self.min_brightness_spin.setValue(50)
        laser_layout.addRow("Мин. яркость:", self.min_brightness_spin)

        self.search_window_spin = QSpinBox()
        self.search_window_spin.setRange(1, 50)
        self.search_window_spin.setValue(8)
        laser_layout.addRow("Окно поиска:", self.search_window_spin)

        self.laser_outlier_spin = QDoubleSpinBox()
        self.laser_outlier_spin.setRange(1.0, 100.0)
        self.laser_outlier_spin.setValue(20.0)
        self.laser_outlier_spin.setSingleStep(1.0)
        laser_layout.addRow("Порог выброса (px):", self.laser_outlier_spin)

        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(1, 200)
        self.max_gap_spin.setValue(10)
        laser_layout.addRow("Макс. разрыв (px):", self.max_gap_spin)

        laser_group.setLayout(laser_layout)
        layout.addWidget(laser_group)

        # === Скользящее окно ===
        window_group = QGroupBox("Скользящее окно (опорная линия)")
        window_layout = QFormLayout()
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(20, 2000)
        self.window_width_spin.setValue(150)
        window_layout.addRow("Ширина окна:", self.window_width_spin)

        self.window_overlap_spin = QSpinBox()
        self.window_overlap_spin.setRange(0, 1000)
        self.window_overlap_spin.setValue(50)
        window_layout.addRow("Перекрытие:", self.window_overlap_spin)

        self.poly_degree_spin = QSpinBox()
        self.poly_degree_spin.setRange(1, 5)
        self.poly_degree_spin.setValue(2)
        window_layout.addRow("Степень полинома:", self.poly_degree_spin)
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)

        # === Пороги дефектов ===
        threshold_group = QGroupBox("Пороги дефектов")
        threshold_layout = QFormLayout()
        self.local_thresh_spin = QDoubleSpinBox()
        self.local_thresh_spin.setRange(0.1, 100.0)
        self.local_thresh_spin.setValue(4.0)
        self.local_thresh_spin.setSingleStep(0.5)
        threshold_layout.addRow("Мин. амплитуда (px):", self.local_thresh_spin)

        self.global_thresh_spin = QDoubleSpinBox()
        self.global_thresh_spin.setRange(0.1, 100.0)
        self.global_thresh_spin.setValue(8.0)
        self.global_thresh_spin.setSingleStep(0.5)
        threshold_layout.addRow("Глобальный порог (px):", self.global_thresh_spin)

        self.critical_thresh_spin = QDoubleSpinBox()
        self.critical_thresh_spin.setRange(0.1, 200.0)
        self.critical_thresh_spin.setValue(15.0)
        self.critical_thresh_spin.setSingleStep(1.0)
        threshold_layout.addRow("Критический порог (px):", self.critical_thresh_spin)

        self.min_width_spin = QSpinBox()
        self.min_width_spin.setRange(1, 100)
        self.min_width_spin.setValue(3)
        threshold_layout.addRow("Мин. ширина (px):", self.min_width_spin)
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)

        # === Временной фильтр ===
        temp_group = QGroupBox("Временная фильтрация")
        temp_layout = QFormLayout()
        self.temp_window_spin = QSpinBox()
        self.temp_window_spin.setRange(1, 30)
        self.temp_window_spin.setValue(5)
        temp_layout.addRow("Кадров подтверждения:", self.temp_window_spin)

        self.spatial_tol_spin = QSpinBox()
        self.spatial_tol_spin.setRange(1, 200)
        self.spatial_tol_spin.setValue(15)
        temp_layout.addRow("Допуск по X (px):", self.spatial_tol_spin)
        temp_group.setLayout(temp_layout)
        layout.addWidget(temp_group)

        # === Область поиска дефектов (ROI) ===
        roi_group = QGroupBox("Область поиска дефектов")
        roi_layout = QFormLayout()
        self.roi_enabled_check = QCheckBox("Ограничить область")
        self.roi_enabled_check.setChecked(False)
        roi_layout.addRow(self.roi_enabled_check)

        self.roi_x_spin = QDoubleSpinBox()
        self.roi_x_spin.setRange(0.0, 1.0)
        self.roi_x_spin.setValue(0.2)
        self.roi_x_spin.setSingleStep(0.01)
        self.roi_x_spin.setDecimals(2)
        roi_layout.addRow("X (доля ширины):", self.roi_x_spin)

        self.roi_y_spin = QDoubleSpinBox()
        self.roi_y_spin.setRange(0.0, 1.0)
        self.roi_y_spin.setValue(0.2)
        self.roi_y_spin.setSingleStep(0.01)
        self.roi_y_spin.setDecimals(2)
        roi_layout.addRow("Y (доля высоты):", self.roi_y_spin)

        self.roi_w_spin = QDoubleSpinBox()
        self.roi_w_spin.setRange(0.0, 1.0)
        self.roi_w_spin.setValue(0.6)
        self.roi_w_spin.setSingleStep(0.01)
        self.roi_w_spin.setDecimals(2)
        roi_layout.addRow("Ширина (доля):", self.roi_w_spin)

        self.roi_h_spin = QDoubleSpinBox()
        self.roi_h_spin.setRange(0.0, 1.0)
        self.roi_h_spin.setValue(0.6)
        self.roi_h_spin.setSingleStep(0.01)
        self.roi_h_spin.setDecimals(2)
        roi_layout.addRow("Высота (доля):", self.roi_h_spin)

        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        # === Визуализация ===
        viz_group = QGroupBox("Отображение")
        viz_layout = QVBoxLayout()
        self.show_ref_check = QCheckBox("Опорная линия")
        self.show_ref_check.setChecked(True)
        self.show_windows_check = QCheckBox("Границы окон")
        self.show_windows_check.setChecked(False)
        self.save_video_check = QCheckBox("Сохранять видео с разметкой")
        self.save_video_check.setChecked(True)
        viz_layout.addWidget(self.show_ref_check)
        viz_layout.addWidget(self.show_windows_check)
        viz_layout.addWidget(self.save_video_check)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # === Кнопки управления ===
        self.start_btn = QPushButton("Старт")
        self.start_btn.clicked.connect(self._start_processing)
        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.clicked.connect(self._stop_processing)
        self.stop_btn.setEnabled(False)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        # === Прогресс ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        panel.setLayout(layout)
        scroll_area.setWidget(panel)
        return scroll_area
    
    def _load_default_params(self):
        """Загружает параметры по умолчанию в поля интерфейса"""
        p = DetectorParams()
        self.file_path_edit.setText(p.video_path)
        self.laser_channel_spin.setValue(p.laser_channel)
        self.min_brightness_spin.setValue(p.laser_min_brightness)
        self.search_window_spin.setValue(p.laser_search_window)
        self.laser_outlier_spin.setValue(p.laser_outlier_threshold)
        self.max_gap_spin.setValue(p.max_gap_bridge)
        self.window_width_spin.setValue(p.window_width)
        self.window_overlap_spin.setValue(p.window_overlap)
        self.poly_degree_spin.setValue(p.poly_degree)
        self.local_thresh_spin.setValue(p.local_deviation_threshold)
        self.global_thresh_spin.setValue(p.global_deviation_threshold)
        self.critical_thresh_spin.setValue(p.critical_threshold)
        self.min_width_spin.setValue(p.min_defect_width)
        self.temp_window_spin.setValue(p.temporal_window)
        self.spatial_tol_spin.setValue(p.spatial_tolerance)
        self.roi_enabled_check.setChecked(p.roi_enabled)
        self.roi_x_spin.setValue(p.roi_x)
        self.roi_y_spin.setValue(p.roi_y)
        self.roi_w_spin.setValue(p.roi_w)
        self.roi_h_spin.setValue(p.roi_h)
        self.show_ref_check.setChecked(p.show_reference_line)
        self.show_windows_check.setChecked(p.show_window_boundaries)
        self.save_video_check.setChecked(p.save_video)
        
    def _update_params_from_ui(self):
        """Переносит значения из виджетов в объект DetectorParams"""
        p = self.current_params
        p.video_path = self.file_path_edit.text().strip()
        p.camera_calibration_path = self.calib_path_edit.text().strip()
        p.laser_channel = self.laser_channel_spin.value()
        p.laser_min_brightness = self.min_brightness_spin.value()
        p.laser_search_window = self.search_window_spin.value()
        p.laser_outlier_threshold = self.laser_outlier_spin.value()
        p.max_gap_bridge = self.max_gap_spin.value()
        p.window_width = self.window_width_spin.value()
        p.window_overlap = self.window_overlap_spin.value()
        p.poly_degree = self.poly_degree_spin.value()
        p.local_deviation_threshold = self.local_thresh_spin.value()
        p.global_deviation_threshold = self.global_thresh_spin.value()
        p.critical_threshold = self.critical_thresh_spin.value()
        p.min_defect_width = self.min_width_spin.value()
        p.temporal_window = self.temp_window_spin.value()
        p.spatial_tolerance = self.spatial_tol_spin.value()
        p.roi_enabled = self.roi_enabled_check.isChecked()
        p.roi_x = self.roi_x_spin.value()
        p.roi_y = self.roi_y_spin.value()
        p.roi_w = self.roi_w_spin.value()
        p.roi_h = self.roi_h_spin.value()
        p.show_reference_line = self.show_ref_check.isChecked()
        p.show_window_boundaries = self.show_windows_check.isChecked()
        p.save_video = self.save_video_check.isChecked()
        p.show_preview = False  # в GUI своё окно
        p.save_frames_with_defects = True  # всегда сохраняем кадры с дефектами
        p.output_dir = "output"

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видеофайл", "",
            "Видео (*.mp4 *.avi *.mov *.mkv);;Все файлы (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def _browse_calib(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите JSON калибровки", "",
            "JSON (*.json);;Все файлы (*)"
        )
        if file_path:
            self.calib_path_edit.setText(file_path)

    @Slot()
    def _start_processing(self):
        """Запускает анализ в зависимости от выбранного источника"""
        self._update_params_from_ui()

        # Проверка существования файла, если выбран файл
        if self.radio_file.isChecked():
            if not os.path.exists(self.current_params.video_path):
                QMessageBox.warning(self, "Ошибка", "Файл не найден!")
                return
            self._start_video_processing()
        else:
            self._start_camera_processing()

    def _start_video_processing(self):
        """Запуск обработки видеофайла в отдельном потоке"""
        if self.processing_thread and self.processing_thread.isRunning():
            return

        # Блокируем кнопки
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # бесконечный прогресс
        self.log_text.clear()
        self.log_text.append("Запуск обработки видео...")

        self.processing_thread = ProcessingThread(self.current_params)
        self.processing_thread.frame_processed.connect(self._on_frame_processed)
        self.processing_thread.processing_finished.connect(self._on_processing_finished)
        self.processing_thread.error_occurred.connect(self._on_error)
        self.processing_thread.start()

    def _start_camera_processing(self):
        """Захват с камеры в реальном времени (без сохранения отчёта)"""
        # Останавливаем предыдущий захват
        if self.timer.isActive():
            self.timer.stop()
        if self.video_capture:
            self.video_capture.release()

        cam_id = self.camera_id_spin.value()
        self.video_capture = cv2.VideoCapture(cam_id)
        if not self.video_capture.isOpened():
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть камеру {cam_id}")
            return

        self._update_params_from_ui()
        # Создаём пайплайн для одиночных кадров, а не для всего видео
        self.camera_pipeline = DefectDetectionPipeline(
            self.current_params,
            frame_callback=self._on_frame_processed
        )
        self.camera_pipeline.params.show_preview = False

        # Запускаем таймер с частотой кадров камеры
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        interval = int(1000 / fps)
        self.timer.start(interval)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.append("Захват с камеры запущен...")

    def _capture_camera_frame(self):
        """Обработчик таймера для захвата с камеры"""
        if not self.video_capture or not self.video_capture.isOpened():
            self.timer.stop()
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            self._on_processing_finished()
            return

        # Обрабатываем кадр с помощью тех же методов пайплайна
        try:
            # Копируем логику пайплайна для одного кадра
            # (можно вынести в отдельный метод пайплайна)
            processed = self.camera_pipeline._process_single_frame(frame)
            if processed is not None:
                annotated, defects, stats = processed
                self._on_frame_processed(annotated, defects, stats)
        except Exception as e:
            self.log_text.append(f"Ошибка: {e}")

    @Slot(np.ndarray, list, object)
    def _on_frame_processed(self, frame: np.ndarray, defects: list, stats: ProcessingStats):
        """Отображает кадр и обновляет статистику"""
        # Конвертируем OpenCV BGR -> RGB -> QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

        # Логируем дефекты
        if defects:
            for d in defects:
                self.log_text.append(
                    f"Кадр {d.frame_id}: {d.severity} отклонение {d.deviation:.1f}px X={d.x:.0f}"
                )
            # Автопрокрутка
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )

    @Slot()
    def _on_processing_finished(self):
        """Обработка завершения (из потока или камеры)"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        self.log_text.append("Обработка завершена.")
        # Показываем итоги, если есть доступ к stats (только после видео)
        if self.processing_thread:
            stats = self.processing_thread.stats if hasattr(self.processing_thread, 'stats') else None
            if stats:
                self.log_text.append(f"Всего кадров: {stats.total_frames}")
                self.log_text.append(f"Дефектов: {stats.total_defects}, критических: {stats.critical_defects}")
            self.processing_thread = None

    @Slot(str)
    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Ошибка", msg)
        self._on_processing_finished()

    @Slot()
    def _stop_processing(self):
        """Останавливает текущий процесс"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait(2000)
        self.timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self._on_processing_finished()

    def closeEvent(self, event):
        self._stop_processing()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())