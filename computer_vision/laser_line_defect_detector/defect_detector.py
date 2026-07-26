"""
Детектор дефектов поверхности по видеопотоку с лазерной линией.

Модуль содержит классы для извлечения лазерной линии, построения опорной линии
методом скользящего окна, поиска дефектов и визуализации.
Поддерживает работу как с видеофайлами, так и с отдельными кадрами.
Может использоваться совместно с GUI-приложением (defect_detector_app.py).
"""

import cv2
import numpy as np
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from collections import deque


# ============================================================
# Параметры по умолчанию
# ============================================================

@dataclass
class DetectorParams:
    """Все настраиваемые параметры детектора дефектов"""

    # --- Источник видео ---
    video_path: str = "test_laser_defects.mp4"
    output_dir: str = "output"

    # --- Параметры камеры ---
    camera_calibration_path: str = ""

    # --- Извлечение лазерной линии ---
    laser_channel: int = 2              # 0=синий, 1=зелёный, 2=красный
    laser_min_brightness: int = 50
    laser_search_window: int = 8

    # --- Скользящее окно ---
    window_width: int = 150
    window_overlap: int = 50
    poly_degree: int = 2

    # --- Пороги обнаружения ---
    local_deviation_threshold: float = 4.0
    global_deviation_threshold: float = 8.0
    critical_threshold: float = 15.0
    min_defect_width: int = 3

    # --- Временная фильтрация ---
    temporal_window: int = 5
    spatial_tolerance: int = 15

    # --- Визуализация ---
    show_preview: bool = True
    save_video: bool = True
    save_frames_with_defects: bool = True
    show_reference_line: bool = True
    show_window_boundaries: bool = False

    # --- Производительность ---
    process_every_n_frames: int = 1
    max_frames: int = 0
    
    roi_enabled: bool = False
    roi_x: float = 0.0          # левая граница (доля ширины, 0..1)
    roi_y: float = 0.0          # верхняя граница (доля высоты)
    roi_w: float = 1.0          # ширина (доля ширины)
    roi_h: float = 1.0          # высота (доля высоты)
    
    laser_outlier_threshold: float = 20.0   # макс. отклонение от соседей (пиксели)
    max_gap_bridge: int = 10   # Максимальный разрыв (в пикселях), который мы "зашиваем" при поиске компонент


# ============================================================
# Структуры данных
# ============================================================

@dataclass
class Defect:
    """Обнаруженный дефект"""
    x: float
    y: float
    deviation: float
    severity: str
    width_px: int
    frame_id: int
    timestamp: float

    def to_dict(self) -> Dict:
        return {
            'frame': self.frame_id,
            'timestamp': round(self.timestamp, 3),
            'x': round(self.x, 1),
            'y': round(self.y, 1),
            'deviation_px': round(self.deviation, 2),
            'severity': self.severity,
            'width_px': self.width_px
        }


@dataclass
class ProcessingStats:
    """Статистика обработки"""
    total_frames: int = 0
    frames_with_defects: int = 0
    total_defects: int = 0
    critical_defects: int = 0
    processing_times: deque = field(default_factory=lambda: deque(maxlen=100))
    start_time: float = 0.0

    @property
    def fps(self) -> float:
        if len(self.processing_times) < 2:
            return 0.0
        return len(self.processing_times) / max(0.001, sum(self.processing_times))

    @property
    def avg_time_ms(self) -> float:
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times) * 1000


# ============================================================
# Калибровка камеры
# ============================================================

class CameraCalibration:
    """Загрузка и применение калибровки камеры"""

    def __init__(self, calibration_path: str = ""):
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.is_calibrated: bool = False
        self._undistort_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None

        if calibration_path and os.path.exists(calibration_path):
            self._load(calibration_path)

    def _load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.camera_matrix = np.array(data['camera_matrix'], dtype=np.float64)
        self.dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float64)
        self.is_calibrated = True
        print(f"Загружена калибровка камеры из {path}")

    def init_undistort(self, width: int, height: int):
        if not self.is_calibrated:
            self._undistort_maps = None
            return
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (width, height), 1
        )
        self._undistort_maps = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None,
            new_matrix, (width, height), cv2.CV_32FC1
        )

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if self._undistort_maps is None:
            return image
        return cv2.remap(image, self._undistort_maps[0], self._undistort_maps[1],
                        cv2.INTER_LINEAR)


# ============================================================
# Извлечение лазерной линии
# ============================================================

class LaserLineExtractor:
    """Извлечение лазерной линии из изображения"""

    def __init__(self, params: DetectorParams):
        self.channel = params.laser_channel
        self.min_brightness = params.laser_min_brightness
        self.search_window = params.laser_search_window
        self.outlier_threshold = params.laser_outlier_threshold
        self.max_gap_bridge = params.max_gap_bridge

    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        laser_y = np.full(width, np.nan, dtype=np.float32)
        valid_mask = np.zeros(width, dtype=bool)

        channel_data = image[:, :, self.channel].astype(np.float32)

        for x in range(width):
            column = channel_data[:, x]
            max_val = np.max(column)
            if max_val < self.min_brightness:
                continue

            y_peak = np.argmax(column)
            y_start = max(0, y_peak - self.search_window)
            y_end = min(height, y_peak + self.search_window + 1)

            window = column[y_start:y_end]
            window_sum = np.sum(window)
            if window_sum > 0:
                y_indices = np.arange(y_start, y_end, dtype=np.float32)
                laser_y[x] = np.sum(y_indices * window) / window_sum
                valid_mask[x] = True
            else:
                laser_y[x] = float(y_peak)
                valid_mask[x] = True

        laser_y, valid_mask = self.remove_spatial_outliers(laser_y, valid_mask)
        laser_y, valid_mask = self.remove_spatial_outliers(laser_y, valid_mask)
        laser_y, valid_mask = self.keep_largest_component(laser_y, valid_mask)
        return laser_y, valid_mask

    def interpolate_and_smooth(self, laser_y: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        if np.sum(valid_mask) < 2:
            return laser_y.copy()

        x_indices = np.arange(len(laser_y))
        interpolated = np.interp(x_indices, x_indices[valid_mask], laser_y[valid_mask])
        smoothed = self._median_filter(interpolated, window=5)
        return smoothed

    def _median_filter(self, data: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return data
        result = np.zeros_like(data)
        half = window // 2
        n = len(data)
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            result[i] = np.median(data[start:end])
        return result
    
    def remove_spatial_outliers(self, laser_y: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Удаляет точки, далеко отстоящие от соседей (ложные срабатывания)"""
        if np.sum(valid_mask) < 3:
            return laser_y.copy(), valid_mask.copy()

        filtered_y = laser_y.copy()
        filtered_mask = valid_mask.copy()
        width = len(laser_y)
        half_window = 5  # по 5 соседей слева и справа

        for i in range(width):
            if not valid_mask[i]:
                continue
            # собираем индексы валидных соседей в окне ±half_window
            start = max(0, i - half_window)
            end = min(width, i + half_window + 1)
            neighbors = []
            for j in range(start, end):
                if j != i and valid_mask[j]:
                    neighbors.append(laser_y[j])
            if len(neighbors) < 2:
                continue
            median_neighbor = np.median(neighbors)
            if abs(laser_y[i] - median_neighbor) > self.outlier_threshold:
                filtered_y[i] = np.nan
                filtered_mask[i] = False
        return filtered_y, filtered_mask
    
    def keep_largest_component(self, laser_y: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Находит самую длинную непрерывную последовательность валидных точек
        (с учётом допустимых разрывов) и удаляет все остальные.
        """
        width = len(valid_mask)
        # Находим индексы валидных точек
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return laser_y, valid_mask

        # Группируем в связные компоненты с учётом max_gap_bridge
        components = []
        current_component = [valid_indices[0]]
        for i in range(1, len(valid_indices)):
            if valid_indices[i] - valid_indices[i-1] <= self.max_gap_bridge:
                current_component.append(valid_indices[i])
            else:
                components.append(current_component)
                current_component = [valid_indices[i]]
        components.append(current_component)

        # Выбираем самую длинную компоненту (по количеству точек)
        if not components:
            return laser_y, valid_mask
        largest = max(components, key=len)

        # Создаём новую маску – только точки из крупнейшей компоненты
        new_mask = np.zeros_like(valid_mask)
        new_mask[largest] = True
        new_laser_y = laser_y.copy()
        new_laser_y[~new_mask] = np.nan

        return new_laser_y, new_mask


# ============================================================
# Скользящее окно и опорная линия
# ============================================================

class SlidingWindowReference:
    """
    Построение опорной линии методом скользящего окна.
    В каждом окне строится полином, затем линии смешиваются с треугольными весами.
    """

    def __init__(self, params: DetectorParams):
        self.window_width = params.window_width
        self.overlap = params.window_overlap
        self.poly_degree = params.poly_degree

    def compute(self, laser_y: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        width = len(laser_y)
        reference_line = np.zeros(width, dtype=np.float32)
        weight_sum = np.zeros(width, dtype=np.float32)
        step = max(1, self.window_width - self.overlap)

        for window_start in range(0, width, step):
            window_end = min(width, window_start + self.window_width)
            window_center = (window_start + window_end) / 2

            x_window = np.arange(window_start, window_end, dtype=np.float32)
            y_window = laser_y[window_start:window_end]
            valid_window = valid_mask[window_start:window_end]

            min_points = max(3, self.poly_degree + 1)
            if np.sum(valid_window) < min_points:
                continue

            x_valid = x_window[valid_window]
            y_valid = y_window[valid_window]

            try:
                coeffs = np.polyfit(x_valid, y_valid, self.poly_degree)
                poly = np.poly1d(coeffs)
                fitted = poly(x_window)
            except (np.linalg.LinAlgError, ValueError):
                continue

            dist_from_center = np.abs(x_window - window_center)
            max_dist = (window_end - window_start) / 2
            if max_dist > 0:
                weight = 1.0 - dist_from_center / max_dist
            else:
                weight = np.ones_like(x_window)
            weight = np.clip(weight, 0, 1)

            reference_line[window_start:window_end] += fitted * weight
            weight_sum[window_start:window_end] += weight

        valid_weights = weight_sum > 0
        reference_line[valid_weights] /= weight_sum[valid_weights]

        if np.sum(valid_weights) > 0 and np.sum(~valid_weights) > 0:
            x_all = np.arange(width, dtype=np.float32)
            reference_line[~valid_weights] = np.interp(
                x_all[~valid_weights],
                x_all[valid_weights],
                reference_line[valid_weights]
            )

        return reference_line


# ============================================================
# Поиск дефектов
# ============================================================

class DefectFinder:
    """Поиск дефектов как отклонений от опорной линии"""

    def __init__(self, params: DetectorParams):
        self.local_threshold = params.local_deviation_threshold
        self.global_threshold = params.global_deviation_threshold
        self.critical_threshold = params.critical_threshold
        self.min_defect_width = params.min_defect_width

    def find(self, laser_y: np.ndarray, reference_line: np.ndarray,
             valid_mask: np.ndarray, frame_id: int, timestamp: float) -> List[Defect]:
        width = len(laser_y)
        deviations = np.zeros(width, dtype=np.float32)
        deviations[valid_mask] = np.abs(laser_y[valid_mask] - reference_line[valid_mask])

        above_threshold = (deviations > self.local_threshold) & valid_mask

        defects = []
        i = 0
        while i < width:
            if above_threshold[i]:
                start = i
                while i < width and above_threshold[i]:
                    i += 1
                end = i
                defect_width = end - start

                if defect_width >= self.min_defect_width:
                    segment_dev = deviations[start:end]
                    max_dev = np.max(segment_dev)
                    max_idx_local = np.argmax(segment_dev)
                    max_idx = start + max_idx_local

                    if max_dev >= self.critical_threshold:
                        severity = 'critical'
                    elif max_dev >= self.critical_threshold * 0.6:
                        severity = 'moderate'
                    else:
                        severity = 'minor'

                    defects.append(Defect(
                        x=float(max_idx),
                        y=float(laser_y[max_idx]),
                        deviation=float(max_dev),
                        severity=severity,
                        width_px=defect_width,
                        frame_id=frame_id,
                        timestamp=timestamp
                    ))
            else:
                i += 1

        return defects


# ============================================================
# Временная фильтрация
# ============================================================

class TemporalFilter:
    """Подтверждение дефектов по нескольким кадрам"""

    def __init__(self, params: DetectorParams):
        self.temporal_window = params.temporal_window
        self.spatial_tolerance = params.spatial_tolerance
        self.history: Dict[int, int] = {}

    def filter(self, defects: List[Defect], frame_id: int) -> List[Defect]:
        current_bins = set()
        for d in defects:
            spatial_bin = int(d.x // self.spatial_tolerance)
            current_bins.add(spatial_bin)
            self.history[spatial_bin] = self.history.get(spatial_bin, 0) + 1

        all_bins = list(self.history.keys())
        for spatial_bin in all_bins:
            if spatial_bin not in current_bins:
                self.history[spatial_bin] -= 1
                if self.history[spatial_bin] <= 0:
                    del self.history[spatial_bin]

        confirmed = []
        for d in defects:
            spatial_bin = int(d.x // self.spatial_tolerance)
            if self.history.get(spatial_bin, 0) >= self.temporal_window:
                confirmed.append(d)
        return confirmed


# ============================================================
# Визуализация
# ============================================================

class Visualizer:
    """Отрисовка результатов"""

    COLORS = {
        'reference_line': (0, 255, 0),
        'laser_line': (255, 0, 0),
        'minor': (0, 255, 255),
        'moderate': (0, 165, 255),
        'critical': (0, 0, 255),
        'window_boundary': (100, 100, 100),
        'text': (255, 255, 255),
        'panel_bg': (40, 40, 40),
    }

    def __init__(self, params: DetectorParams):
        self.params = params 
        self.show_reference = params.show_reference_line
        self.show_windows = params.show_window_boundaries
        self.window_width = params.window_width
        self.window_overlap = params.window_overlap

    def render(self, image: np.ndarray, laser_y: np.ndarray,
           reference_line: np.ndarray, valid_mask: np.ndarray,
           defects: List[Defect], frame_id: int,
           stats: ProcessingStats) -> np.ndarray:
        result = image.copy()
        height, width = result.shape[:2]

        # Опорная линия (только на валидных участках)
        if self.show_reference:
            self._draw_line(result, reference_line, self.COLORS['reference_line'],
                            dashed=True, valid_mask=valid_mask)

        # Границы скользящих окон
        if self.show_windows:
            self._draw_window_boundaries(result, width)

        # Лазерная линия (только валидная часть)
        self._draw_laser_line(result, laser_y, valid_mask)

        # Прямоугольник ROI, если включён
        if hasattr(self, 'params') and self.params.roi_enabled:
            self._draw_roi(result)

        # Дефекты
        for defect in defects:
            self._draw_defect(result, defect)

        # Панель статистики
        self._draw_stats(result, frame_id, stats, len(defects))

        return result

    def _draw_line(self, image: np.ndarray, y_values: np.ndarray,
                   color: Tuple[int, int, int], dashed: bool = False,
                   valid_mask: Optional[np.ndarray] = None):
        height, width = image.shape[:2]
        step = 8 if dashed else 1

        for i in range(0, width - step, step):
            x1, x2 = i, i + step
            y1, y2 = int(y_values[i]), int(y_values[i + step])

            if not ((0 <= y1 < height) and (0 <= y2 < height)):
                continue
            if valid_mask is not None:
                if not (valid_mask[i] and valid_mask[i + step]):
                    continue
            cv2.line(image, (x1, y1), (x2, y2), color, 1)

    def _draw_laser_line(self, image: np.ndarray, laser_y: np.ndarray,
                        valid_mask: np.ndarray):
        height, width = image.shape[:2]
        color = self.COLORS['laser_line']
        for i in range(width - 1):
            if valid_mask[i] and valid_mask[i + 1]:
                y1, y2 = int(laser_y[i]), int(laser_y[i + 1])
                if (0 <= y1 < height) and (0 <= y2 < height):
                    cv2.line(image, (i, y1), (i + 1, y2), color, 1)

    def _draw_window_boundaries(self, image: np.ndarray, width: int):
        height = image.shape[:2][0]
        step = max(1, self.window_width - self.window_overlap)
        color = self.COLORS['window_boundary']
        for x in range(0, width, step):
            cv2.line(image, (x, 0), (x, height), color, 1, cv2.LINE_AA)

    def _draw_defect(self, image: np.ndarray, defect: Defect):
        color = self.COLORS[defect.severity]
        x, y = int(defect.x), int(defect.y)
        cv2.circle(image, (x, y), 18, color, 2)
        cv2.line(image, (x - 12, y), (x + 12, y), color, 1)
        cv2.line(image, (x, y - 12), (x, y + 12), color, 1)

        label = f"{defect.severity[0].upper()} {defect.deviation:.1f}px"
        text_pos = (x + 22, y - 8)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(image,
                     (text_pos[0] - 3, text_pos[1] - th - 3),
                     (text_pos[0] + tw + 3, text_pos[1] + 3),
                     (0, 0, 0), -1)
        cv2.putText(image, label, text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _draw_stats(self, image: np.ndarray, frame_id: int,
                   stats: ProcessingStats, defect_count: int):
        h, w = image.shape[:2]
        panel_h = 55
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), self.COLORS['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

        lines = [
            f"Frame: {frame_id} | FPS: {stats.fps:.1f} | Time: {stats.avg_time_ms:.1f}ms",
            f"Defects: {defect_count} | Total: {stats.total_defects} | Critical: {stats.critical_defects}"
        ]
        y = 20
        for line in lines:
            cv2.putText(image, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['text'], 1)
            y += 22

    def _draw_roi(self, image: np.ndarray):
        h, w = image.shape[:2]
        p = self.params
        x1 = int(p.roi_x * w)
        y1 = int(p.roi_y * h)
        x2 = int((p.roi_x + p.roi_w) * w)
        y2 = int((p.roi_y + p.roi_h) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

# ============================================================
# Основной пайплайн
# ============================================================

class DefectDetectionPipeline:
    """
    Объединяет все компоненты.
    Может работать в двух режимах:
    - run() для обработки видеофайла
    - _process_single_frame() для покадровой обработки (камера/GUI)
    """

    def __init__(self, params: DetectorParams,
                 frame_callback: Optional[Callable[[np.ndarray, List[Defect], ProcessingStats], None]] = None):
        self.params = params
        self.frame_callback = frame_callback

        self.calibration = CameraCalibration(params.camera_calibration_path)
        self.extractor = LaserLineExtractor(params)
        self.reference_builder = SlidingWindowReference(params)
        self.defect_finder = DefectFinder(params)
        self.temporal_filter = TemporalFilter(params)
        self.visualizer = Visualizer(params)

        self.stats = ProcessingStats()
        self.all_defects: List[Defect] = []

    def run(self):
        """Обработка видеофайла"""
        params = self.params
        cap = cv2.VideoCapture(params.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {params.video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if params.max_frames > 0:
            total_frames = min(total_frames, params.max_frames)

        self.calibration.init_undistort(width, height)
        os.makedirs(params.output_dir, exist_ok=True)

        video_writer = None
        if params.save_video:
            output_video_path = os.path.join(params.output_dir, "detected_defects.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not video_writer.isOpened():
                print("Не удалось создать выходное видео")
                video_writer = None

        if params.save_frames_with_defects:
            defects_dir = os.path.join(params.output_dir, "defect_frames")
            os.makedirs(defects_dir, exist_ok=True)

        self.stats.start_time = time.time()
        frame_id = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame_id >= total_frames:
                    break
                if frame_id % params.process_every_n_frames != 0:
                    frame_id += 1
                    continue

                t_start = time.time()

                # Обработка кадра
                annotated, defects, _ = self._process_frame_logic(frame, frame_id, fps)

                processing_time = time.time() - t_start
                self.stats.processing_times.append(processing_time)

                # Колбэк для GUI
                if self.frame_callback:
                    self.frame_callback(annotated, defects, self.stats)

                # Вывод в консоль
                if defects:
                    self._log_defects(defects)

                # Сохранение
                if video_writer:
                    video_writer.write(annotated)
                if params.save_frames_with_defects and defects:
                    frame_path = os.path.join(defects_dir, f"frame_{frame_id:06d}.png")
                    cv2.imwrite(frame_path, annotated)

                if params.show_preview:
                    cv2.imshow('Defect Detection', annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break

                frame_id += 1

        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()

        self._save_report()

    def _process_frame_logic(self, frame: np.ndarray, frame_id: int,
                             fps: float) -> Tuple[np.ndarray, List[Defect], ProcessingStats]:
        """Общая логика обработки одного кадра (для видео и камеры)"""
        if self.calibration.is_calibrated:
            frame = self.calibration.undistort(frame)

        laser_y, valid_mask = self.extractor.extract(frame)
        laser_y_smooth = self.extractor.interpolate_and_smooth(laser_y, valid_mask)
        reference_line = self.reference_builder.compute(laser_y_smooth, valid_mask)

        timestamp = frame_id / fps if fps > 0 else frame_id / 30.0
        raw_defects = self.defect_finder.find(
            laser_y_smooth, reference_line, valid_mask, frame_id, timestamp
        )
        confirmed_defects = self.temporal_filter.filter(raw_defects, frame_id)

        self.stats.total_frames += 1
        if confirmed_defects:
            self.stats.frames_with_defects += 1
            self.stats.total_defects += len(confirmed_defects)
            self.stats.critical_defects += sum(
                1 for d in confirmed_defects if d.severity == 'critical'
            )
            self.all_defects.extend(confirmed_defects)

        annotated = self.visualizer.render(
            frame, laser_y_smooth, reference_line, valid_mask,
            confirmed_defects, frame_id, self.stats
        )
        return annotated, confirmed_defects, self.stats

    def _process_single_frame(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, List[Defect], ProcessingStats]]:
        """
        Обработка одного кадра без привязки к видеофайлу.
        Используется для режима камеры в GUI.
        """
        try:
            # Используем накопленный счётчик кадров
            frame_id = self.stats.total_frames
            fps = 30  # для камеры условно
            return self._process_frame_logic(frame, frame_id, fps)
        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            return None

    def _log_defects(self, defects: List[Defect]):
        critical = [d for d in defects if d.severity == 'critical']
        if critical:
            print(f"  Кадр {defects[0].frame_id}: [CRITICAL] {len(critical)} дефектов, "
                  f"макс={max(d.deviation for d in critical):.1f}px")
        else:
            print(f"  Кадр {defects[0].frame_id}: {len(defects)} дефектов "
                  f"({', '.join(d.severity for d in defects)})")

    def _save_report(self):
        """Сохраняет итоговый JSON-отчёт"""
        elapsed = time.time() - self.stats.start_time
        print(f"\nОбработка завершена. Всего кадров: {self.stats.total_frames}")
        print(f"Дефектов: {self.stats.total_defects}, критических: {self.stats.critical_defects}")

        report_path = os.path.join(self.params.output_dir, "defects_report.json")
        report = {
            'video': self.params.video_path,
            'params': {
                'window_width': self.params.window_width,
                'local_threshold': self.params.local_deviation_threshold,
                'critical_threshold': self.params.critical_threshold,
            },
            'stats': {
                'total_frames': self.stats.total_frames,
                'frames_with_defects': self.stats.frames_with_defects,
                'total_defects': self.stats.total_defects,
                'critical_defects': self.stats.critical_defects,
                'processing_time_sec': round(elapsed, 1),
                'avg_fps': round(self.stats.fps, 1),
            },
            'defects': [d.to_dict() for d in self.all_defects]
        }
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Отчёт сохранён: {report_path}")


# ============================================================
# Точка входа для автономного запуска (без GUI)
# ============================================================

def main():
    params = DetectorParams()
    # При необходимости переопределите параметры здесь
    pipeline = DefectDetectionPipeline(params)
    pipeline.run()


if __name__ == "__main__":
    main()