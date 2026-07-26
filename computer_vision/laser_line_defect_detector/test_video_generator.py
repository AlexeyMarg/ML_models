"""
Генератор тестовых видео с лазерной линией для отладки детектора дефектов.

Особенности:
- Лазерная линия может занимать не всю ширину кадра (зона задаётся в долях ширины)
- Полотно имеет плавные изгибы (неровности основы), не являющиеся дефектами
- Дефекты — резкие локальные отклонения на фоне плавных изгибов
- Исправлена отрисовка: линия всегда яркая и видимая, как в старой проверенной версии
"""

import numpy as np
import cv2
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# ============================================================
# Конфигурации
# ============================================================

@dataclass
class DefectScenario:
    """Описание одного дефекта в сценарии"""
    x_center: float          # центр дефекта по X (в долях ширины, 0..1)
    amplitude_px: float      # амплитуда отклонения в пикселях
    width_px: float          # характерная ширина дефекта в пикселях
    frame_start: int         # кадр появления
    frame_peak: int          # кадр максимального развития
    frame_end: int           # кадр исчезновения
    defect_type: str = 'dent'  # 'dent' (вмятина) или 'bump' (выступ)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DefectScenario':
        return cls(
            x_center=d.get('x_center', 0.5),
            amplitude_px=d.get('amplitude_px', 10.0),
            width_px=d.get('width_px', 15.0),
            frame_start=d.get('frame_start', 30),
            frame_peak=d.get('frame_peak', 60),
            frame_end=d.get('frame_end', 90),
            defect_type=d.get('defect_type', 'dent')
        )


# ============================================================
# Основная функция генерации (работает как старая, но с параметрами)
# ============================================================

def generate_test_video(
    output_path: str = "test_laser_defects.mp4",
    duration_sec: float = 10.0,
    fps: int = 30,
    width: int = 800,
    height: int = 600,
    base_y: float = 300.0,
    base_angle: float = 2.0,
    laser_start: float = 0.0,      # 0.0 = от левого края
    laser_end: float = 1.0,        # 1.0 = до правого края
    edge_fade: float = 0.03,       # ширина затухания в долях ширины
    laser_thickness: float = 1.5,
    laser_brightness: int = 240,
    background_brightness: int = 20,
    sensor_noise: float = 0.3,
    wave_amplitude: float = 4.0,
    wave_count: int = 3,
    ripple_amplitude: float = 1.5,
    ripple_count: int = 12,
    vertical_drift_amplitude: float = 3.0,
    vertical_drift_period: float = 5.0,
    defects: Optional[List[Dict[str, Any]]] = None,
    show_progress: bool = True,
    codec: str = 'mp4v',
) -> str:
    """
    Генерация тестового видео с лазерной линией.
    Параметры аналогичны старой версии, но расширены.
    """
    total_frames = int(duration_sec * fps)
    defects = defects or []

    # Преобразуем словари дефектов в объекты
    defect_scenarios = [DefectScenario.from_dict(d) for d in defects]

    # Создаём выходную папку при необходимости
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Выбор кодека
    codec_map = {
        'mp4v': ('.mp4', cv2.VideoWriter_fourcc(*'mp4v')),
        'avc1': ('.mp4', cv2.VideoWriter_fourcc(*'avc1')),
        'XVID': ('.avi', cv2.VideoWriter_fourcc(*'XVID')),
    }
    if codec not in codec_map:
        codec = 'mp4v'
    ext, fourcc = codec_map[codec]

    if not output_path.lower().endswith(ext):
        base = os.path.splitext(output_path)[0]
        output_path = base + ext

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Не удалось создать видеофайл {output_path}")

    if show_progress:
        print(f"Генерация видео: {output_path}")
        print(f"  Размер: {width}x{height}, {total_frames} кадров, {fps} FPS")

    # Вспомогательные функции для поверхности и дефектов
    def get_surface_profile(x_norm, t):
        """Плавные волны и рябь"""
        profile = np.zeros_like(x_norm)
        # Крупные волны
        for i in range(wave_count):
            freq = (i + 1) * 2 * np.pi
            phase_drift = 0.2 * t * (i + 1) * 0.7
            phase = i * np.pi / max(1, wave_count) + phase_drift
            amp = wave_amplitude / (i + 1) ** 0.5
            profile += amp * np.sin(x_norm * freq + phase)
        # Мелкая рябь
        for i in range(ripple_count):
            freq = (i + 1) * 2 * np.pi * 2.5
            phase_drift = 0.2 * t * (i + 1) * 1.3
            phase = i * np.pi * 0.7 + phase_drift
            amp = ripple_amplitude / (i + 1)
            profile += amp * np.sin(x_norm * freq + phase)
        # Вертикальный дрейф
        if vertical_drift_period > 0:
            drift = vertical_drift_amplitude * np.sin(2 * np.pi * t / vertical_drift_period)
            profile += drift
        return profile

    def get_defect_profile(x, t, frame_id):
        """Локальные дефекты"""
        prof = np.zeros_like(x)
        for d in defect_scenarios:
            if not (d.frame_start <= frame_id <= d.frame_end):
                continue
            # Прогресс развития
            if frame_id <= d.frame_peak:
                progress = (frame_id - d.frame_start) / max(1, d.frame_peak - d.frame_start)
            else:
                progress = 1.0 - (frame_id - d.frame_peak) / max(1, d.frame_end - d.frame_peak)
            progress = np.clip(progress, 0, 1)
            # Гауссиан
            x_center_px = d.x_center * width
            gauss = np.exp(-0.5 * ((x - x_center_px) / max(d.width_px, 1.0)) ** 2)
            sign = -1.0 if d.defect_type == 'dent' else 1.0
            prof += sign * d.amplitude_px * progress * gauss
        return prof

    # Основной цикл
    for frame_id in range(total_frames):
        t = frame_id / fps
        # Создаём фон
        frame = np.full((height, width, 3), background_brightness, dtype=np.uint8)

        # Оси X и нормализованная координата
        x = np.arange(width, dtype=np.float32)
        x_norm = x / width

        # Базовая линия (прямая с наклоном)
        angle_rad = np.radians(base_angle)
        line_y = base_y + np.tan(angle_rad) * (x - width / 2)

        # Добавляем поверхность
        line_y += get_surface_profile(x_norm, t)

        # Добавляем дефекты
        line_y += get_defect_profile(x, t, frame_id)

        # Шум сенсора
        line_y += np.random.normal(0, sensor_noise, size=width)

        # Обрезаем, чтобы не выходило за границы
        line_y = np.clip(line_y, 10, height - 10)

        # --- Отрисовка лазерной линии (как в старой функции) ---
        # Вычисляем маску видимости (плавное появление/исчезновение)
        start_px = int(laser_start * width)
        end_px = int(laser_end * width)
        fade_px = max(1, int(edge_fade * width))

        # Рисуем только в зоне видимости с плавным затуханием
        for i in range(start_px, end_px):
            yi = int(line_y[i])
            if yi < 0 or yi >= height:
                continue

            # Плавное затухание на краях зоны
            fade_factor = 1.0
            if i < start_px + fade_px:
                fade_factor = (i - start_px) / fade_px
            elif i > end_px - fade_px:
                fade_factor = (end_px - i) / fade_px
            fade_factor = max(0.0, min(1.0, fade_factor))

            # Яркость с учётом затухания
            bright = int(laser_brightness * fade_factor)

            # Основная линия (яркий пик)
            frame[max(0, yi-1):min(height, yi+2), i, 2] = bright

            # Ореол (гауссово размытие)
            for dy in range(-4, 5):
                y_pos = yi + dy
                if 0 <= y_pos < height:
                    intensity = int(bright * 0.6 * np.exp(-dy**2 / 4))
                    frame[y_pos, i, 2] = max(int(frame[y_pos, i, 2]), intensity)

        # Номер кадра
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        out.write(frame)

        if show_progress and (frame_id % 50 == 0 or frame_id == total_frames - 1):
            pct = (frame_id + 1) / total_frames * 100
            print(f"\r  Прогресс: {frame_id+1}/{total_frames} ({pct:.0f}%)", end='')

    out.release()
    if show_progress:
        print("\n  Готово!")
    return output_path


# ============================================================
# Функция-обёртка для вызова со словарём (как раньше)
# ============================================================

def generate_video(params: Dict[str, Any]) -> str:
    """
    Принимает словарь параметров и вызывает generate_test_video.
    Пример использования:
        generate_video({
            'output': 'test.mp4',
            'laser_start': 0.3,
            'laser_end': 0.7,
            'defects': [{'x_center': 0.5, 'amplitude_px': 12, ...}]
        })
    """
    # Достаём параметры из словаря, подставляя значения по умолчанию
    return generate_test_video(
        output_path=params.get('output', 'test.mp4'),
        duration_sec=params.get('duration_sec', 10.0),
        fps=params.get('fps', 30),
        width=params.get('width', 800),
        height=params.get('height', 600),
        base_y=params.get('base_y', 300.0),
        base_angle=params.get('base_angle', 2.0),
        laser_start=params.get('laser_start', 0.05),
        laser_end=params.get('laser_end', 0.95),
        edge_fade=params.get('edge_fade', 0.03),
        laser_thickness=params.get('laser_thickness', 1.5),
        laser_brightness=params.get('laser_brightness', 240),
        background_brightness=params.get('background_brightness', 20),
        sensor_noise=params.get('sensor_noise', 0.3),
        wave_amplitude=params.get('wave_amplitude', 4.0),
        wave_count=params.get('wave_count', 3),
        ripple_amplitude=params.get('ripple_amplitude', 1.5),
        ripple_count=params.get('ripple_count', 12),
        vertical_drift_amplitude=params.get('vertical_drift_amplitude', 3.0),
        vertical_drift_period=params.get('vertical_drift_period', 5.0),
        defects=params.get('defects', None),
        show_progress=params.get('show_progress', True),
        codec=params.get('codec', 'mp4v'),
    )


# ============================================================
# Точка входа для командной строки
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Генератор тестового видео с лазерной линией')
    parser.add_argument('--output', '-o', default='test_laser_defects.mp4')
    parser.add_argument('--duration', '-d', type=float, default=10.0)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--laser-start', type=float, default=0.05)
    parser.add_argument('--laser-end', type=float, default=0.95)
    parser.add_argument('--wave-amplitude', type=float, default=4.0)
    parser.add_argument('--base-angle', type=float, default=2.0)
    parser.add_argument('--no-defects', action='store_true')
    parser.add_argument('--codec', default='mp4v')

    args = parser.parse_args()

    defects = None if args.no_defects else [
        {'x_center': 0.25, 'amplitude_px': 10, 'width_px': 15, 'frame_start': 30, 'frame_peak': 60, 'frame_end': 90, 'defect_type': 'dent'},
        {'x_center': 0.45, 'amplitude_px': 8, 'width_px': 8, 'frame_start': 90, 'frame_peak': 110, 'frame_end': 140, 'defect_type': 'bump'},
        {'x_center': 0.65, 'amplitude_px': 20, 'width_px': 12, 'frame_start': 150, 'frame_peak': 175, 'frame_end': 200, 'defect_type': 'dent'},
    ]

    params = {
        'output': args.output,
        'duration_sec': args.duration,
        'fps': args.fps,
        'width': args.width,
        'height': args.height,
        'laser_start': args.laser_start,
        'laser_end': args.laser_end,
        'wave_amplitude': args.wave_amplitude,
        'base_angle': args.base_angle,
        'defects': defects,
        'codec': args.codec,
    }
    generate_video(params)


if __name__ == '__main__':
    main()