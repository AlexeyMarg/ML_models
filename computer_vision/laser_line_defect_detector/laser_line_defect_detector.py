import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import deque
import time
import threading
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import RANSACRegressor


class DefectSeverity(Enum):
    """Defect levels"""
    MINOR = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


@dataclass
class Defect:
    """Defect information"""
    x: int
    y: int
    deviation: float
    severity: DefectSeverity
    frame_id: int
    timestamp: float
    width_px: int = 0
    depth_mm: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'deviation_px': round(self.deviation, 2),
            'severity': self.severity.value,
            'frame': self.frame_id,
            'width_px': self.width_px,
            'depth_mm': self.depth_mm
        }


@dataclass
class ProcessingStats:
    """Processing statistics for performance monitoring"""
    fps: float = 0.0
    processing_time_ms: float = 0.0
    defects_detected: int = 0
    total_frames: int = 0
    critical_defects: int = 0
    
    def update(self, processing_time: float, defects: List[Defect]):
        self.total_frames += 1
        self.processing_time_ms = processing_time * 1000
        if self.processing_time_ms > 0:
            self.fps = 1000.0 / self.processing_time_ms
        self.defects_detected += len(defects)
        self.critical_defects += sum(
            1 for d in defects if d.severity == DefectSeverity.CRITICAL
        )


class LaserLineExtractor:
    """Laser line extraction with subpixel accuracy"""
    
    def __init__(self, laser_channel: int = 2, min_brightness: int = 50):
        """
        Args:
            laser_channel: 0=blue, 1=green, 2=red
            min_brightness: minimum brightness for detection
        """
        self.laser_channel = laser_channel
        self.min_brightness = min_brightness
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the laser line with subpixel accuracy.
        Uses the center of mass method to improve accuracy.
        """
        height, width = image.shape[:2]
        laser_points = np.full(width, np.nan, dtype=np.float32)
        
        channel_data = image[:, :, self.laser_channel].astype(np.float32)
        
        for x in range(width):
            column = channel_data[:, x]
            max_val = np.max(column)
            
            if max_val < self.min_brightness:
                continue
            
            y_peak = np.argmax(column)
            
            # define window for center of mass calculation
            window_half = 4
            y_start = max(0, y_peak - window_half)
            y_end = min(height, y_peak + window_half + 1)
            
            window = column[y_start:y_end]
            window_sum = np.sum(window)
            
            if window_sum > 0:
                y_indices = np.arange(y_start, y_end, dtype=np.float32)
                laser_points[x] = np.sum(y_indices * window) / window_sum
            else:
                laser_points[x] = float(y_peak)
        
        return laser_points


class LineProcessor:
    """Laser line filtering and processing"""
    
    def __init__(self, smoothing_window: int = 5, outlier_threshold: float = 3.0):
        self.smoothing_window = smoothing_window
        self.outlier_threshold = outlier_threshold
    
    def process(self, laser_points: np.ndarray) -> np.ndarray:
        """Full processing cycle for the laser line"""
        points = laser_points.copy()
        
        points = self._remove_outliers(points)
        points = self._interpolate_nans(points)
        if self.smoothing_window > 1:
            points = self._smooth(points)
        
        return points
    
    def _remove_outliers(self, points: np.ndarray) -> np.ndarray:
        """Remove statistical outliers"""
        valid_mask = ~np.isnan(points)
        if np.sum(valid_mask) < 5:
            return points
        
        valid_points = points[valid_mask]
        median = np.median(valid_points)
        mad = np.median(np.abs(valid_points - median))
        
        if mad == 0:
            return points
        
        # Modified Z-score
        modified_z = 0.6745 * (points - median) / mad
        outlier_mask = np.abs(modified_z) > self.outlier_threshold
        
        points[outlier_mask] = np.nan
        return points
    
    def _interpolate_nans(self, points: np.ndarray) -> np.ndarray:
        """Linear interpolation of missing values"""
        valid_mask = ~np.isnan(points)
        
        if np.sum(valid_mask) < 2:
            return points
        
        x = np.arange(len(points))
        return np.interp(x, x[valid_mask], points[valid_mask])
    
    def _smooth(self, points: np.ndarray) -> np.ndarray:
        """Median filtering"""
        half_window = self.smoothing_window // 2
        n = len(points)
        result = np.zeros_like(points)
        
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            result[i] = np.median(points[start:end])
        
        return result


class ReferenceLineFitter:
    """Laser line fitting using RANSAC"""
    
    def __init__(self, ransac_threshold: float = 5.0, min_samples: int = 50):
        self.ransac_threshold = ransac_threshold
        self.min_samples = min_samples
    
    def fit(self, points: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Fitting of the reference line.
        
        Returns:
            slope, intercept, inlier_mask, reference_line
        """
        x = np.arange(len(points)).reshape(-1, 1)
        y = points.reshape(-1, 1)
        
        if len(points) < self.min_samples:
            slope, intercept = np.polyfit(x.flatten(), y.flatten(), 1)
            reference_line = slope * x.flatten() + intercept
            inlier_mask = np.ones(len(points), dtype=bool)
            return slope, intercept, inlier_mask, reference_line
        
        try:
            ransac = RANSACRegressor(
                residual_threshold=self.ransac_threshold,
                max_trials=200,
                min_samples=self.min_samples,
                random_state=42
            )
            ransac.fit(x, y)
            
            slope = ransac.estimator_.coef_[0][0]
            intercept = ransac.estimator_.intercept_[0]
            inlier_mask = ransac.inlier_mask_
            
            reference_line = slope * x.flatten() + intercept
            
            return slope, intercept, inlier_mask, reference_line
            
        except Exception:
            slope, intercept = np.polyfit(x.flatten(), y.flatten(), 1)
            reference_line = slope * x.flatten() + intercept
            inlier_mask = np.ones(len(points), dtype=bool)
            return slope, intercept, inlier_mask, reference_line


class DefectFinder:
    """Defect finding based on deviations from the reference line"""
    
    def __init__(
        self,
        deviation_threshold: float = 5.0,
        critical_threshold: float = 15.0,
        min_defect_width: int = 3,
        px_to_mm_ratio: Optional[float] = None
    ):
        self.deviation_threshold = deviation_threshold
        self.critical_threshold = critical_threshold
        self.min_defect_width = min_defect_width
        self.px_to_mm_ratio = px_to_mm_ratio
    
    def find(
        self,
        points: np.ndarray,
        reference_line: np.ndarray,
        frame_id: int
    ) -> List[Defect]:
        """Defect finding"""
        timestamp = time.time()
        
        # Compute absolute deviations
        deviations = np.abs(points - reference_line)
        
        # Find regions above the threshold
        above_threshold = deviations > self.deviation_threshold
        
        defects = []
        i = 0
        
        while i < len(above_threshold):
            if above_threshold[i]:
                # Beginning of a defect
                start = i
                
                # Find the end of the defect
                while i < len(above_threshold) and above_threshold[i]:
                    i += 1
                end = i
                
                width = end - start
                
                if width >= self.min_defect_width:
                    segment_deviations = deviations[start:end]
                    max_dev = np.max(segment_deviations)
                    max_idx_local = np.argmax(segment_deviations)
                    max_idx = start + max_idx_local
                    
                    severity = self._classify_severity(max_dev)
                    
                    depth_mm = None
                    if self.px_to_mm_ratio:
                        depth_mm = max_dev * self.px_to_mm_ratio
                    
                    defects.append(Defect(
                        x=int(max_idx),
                        y=int(points[max_idx]),
                        deviation=float(max_dev),
                        severity=severity,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        width_px=width,
                        depth_mm=depth_mm
                    ))
            else:
                i += 1
        
        return defects
    
    def _classify_severity(self, deviation: float) -> DefectSeverity:
        """Classify the severity of a defect"""
        ratio = deviation / self.critical_threshold
        
        if ratio >= 1.0:
            return DefectSeverity.CRITICAL
        elif ratio >= 0.6:
            return DefectSeverity.MODERATE
        else:
            return DefectSeverity.MINOR


class TemporalFilter:
    """Temporal filter for suppressing false positives"""
    
    def __init__(self, history_size: int = 5, confirmation_frames: int = 3):
        self.history: deque = deque(maxlen=history_size)
        self.confirmation_frames = confirmation_frames
        self.defect_persistence: Dict[int, int] = {}
    
    def filter(self, defects: List[Defect], frame_id: int) -> List[Defect]:
        self.history.append((frame_id, defects))
        
        if len(self.history) < self.confirmation_frames:
            return defects
        
        # Simple version: group defects by proximity of X-coordinates
        confirmed_defects = []
        
        for defect in defects:
            # Find similar defects in the history
            key = defect.x // 10  # Grouping by 10 pixels
            self.defect_persistence[key] = self.defect_persistence.get(key, 0) + 1
            
            if self.defect_persistence.get(key, 0) >= self.confirmation_frames:
                confirmed_defects.append(defect)
        
        # Clear old keys
        if frame_id % 30 == 0:
            self.defect_persistence = {
                k: v for k, v in self.defect_persistence.items()
                if v > 0
            }
        
        return confirmed_defects


class AlertSystem:
    """Alert system"""
    
    def __init__(
        self,
        cooldown_seconds: float = 1.0,
        sound_enabled: bool = False,
        log_file: Optional[str] = None
    ):
        self.cooldown = cooldown_seconds
        self.last_alert_time = 0.0
        self.sound_enabled = sound_enabled
        self.log_file = log_file
        self.alert_history: List[str] = []
        
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write(f"=== Defect Detection Log ===\n")
                f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def check(self, defects: List[Defect], frame_id: int) -> Optional[str]:
        """Check if an alert is needed"""
        current_time = time.time()
        
        if not defects:
            return None
        
        critical = [d for d in defects if d.severity == DefectSeverity.CRITICAL]
        moderate = [d for d in defects if d.severity == DefectSeverity.MODERATE]
        
        if (current_time - self.last_alert_time) < self.cooldown:
            return None
        
        alert_msg = None
        
        if critical:
            alert_msg = (
                f"CRITICAL DEFECT! "
                f"Frame {frame_id}, "
                f"count: {len(critical)}, "
                f"max deviation: {max(d.deviation for d in critical):.1f} px"
            )
            self.last_alert_time = current_time
            
        elif len(moderate) >= 2:
            alert_msg = (
                f"MULTIPLE DEFECTS! "
                f"Frame {frame_id}, "
                f"moderate: {len(moderate)}"
            )
            self.last_alert_time = current_time
        
        if alert_msg:
            self.alert_history.append(alert_msg)
            self._log(alert_msg)
            
            if self.sound_enabled:
                self._play_alert_sound()
        
        return alert_msg
    
    def _log(self, message: str):
        """Log to file"""
        if self.log_file:
            timestamp = time.strftime('%H:%M:%S')
            with open(self.log_file, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
    
    def _play_alert_sound(self):
        """Play alert sound (platform-dependent)"""
        try:
            import sys
            if sys.platform == 'win32':
                import winsound
                winsound.Beep(1000, 200)
            elif sys.platform == 'darwin':
                import os
                os.system('afplay /System/Library/Sounds/Ping.aiff')
            else:
                print('\a') 
        except Exception:
            pass


class Visualizer:
    """Visualizer for displaying results"""
    
    # Цветовая схема
    COLORS = {
        'reference_line': (0, 255, 0),      
        'laser_line': (255, 0, 0),           
        DefectSeverity.MINOR: (0, 255, 255),     
        DefectSeverity.MODERATE: (0, 165, 255),  
        DefectSeverity.CRITICAL: (0, 0, 255),    
        'text': (255, 255, 255),             
        'panel_bg': (40, 40, 40),            
    }
    
    def __init__(self, show_details: bool = True, show_stats: bool = True):
        self.show_details = show_details
        self.show_stats = show_stats
        self.alert_message: Optional[str] = None
        self.alert_timer: float = 0.0
        self.alert_duration: float = 2.0
    
    def set_alert(self, message: str):
        """Set alert message"""
        self.alert_message = message
        self.alert_timer = time.time()
    
    def render(
        self,
        image: np.ndarray,
        laser_points: np.ndarray,
        reference_line: np.ndarray,
        defects: List[Defect],
        stats: ProcessingStats,
        frame_id: int
    ) -> np.ndarray:
        """Render all elements on the image"""
        result = image.copy()
        height, width = result.shape[:2]
        
        self._draw_reference_line(result, reference_line)
        
        self._draw_laser_line(result, laser_points)
        
        for defect in defects:
            self._draw_defect(result, defect)
        
        if self.show_stats:
            self._draw_stats_panel(result, stats, frame_id)
        
        if self.alert_message and (time.time() - self.alert_timer) < self.alert_duration:
            self._draw_alert_banner(result, self.alert_message)
        
        return result
    
    def _draw_reference_line(self, image: np.ndarray, reference_line: np.ndarray):
        height, width = image.shape[:2]
        color = self.COLORS['reference_line']
        
        step = 10  # шаг пунктира
        for i in range(0, width - step, step * 2):
            x1, x2 = i, i + step
            y1, y2 = int(reference_line[i]), int(reference_line[i + step])
            
            if (0 <= y1 < height) and (0 <= y2 < height):
                cv2.line(image, (x1, y1), (x2, y2), color, 1)
    
    def _draw_laser_line(self, image: np.ndarray, points: np.ndarray):
        height, width = image.shape[:2]
        color = self.COLORS['laser_line']
        
        for x in range(width - 1):
            y1 = int(points[x])
            y2 = int(points[x + 1])
            
            if (0 <= y1 < height) and (0 <= y2 < height):
                cv2.line(image, (x, y1), (x + 1, y2), color, 1)
    
    def _draw_defect(self, image: np.ndarray, defect: Defect):
        color = self.COLORS[defect.severity]
        
        cv2.circle(image, (defect.x, defect.y), 20, color, 2)
        
        cv2.line(image, (defect.x - 15, defect.y), (defect.x + 15, defect.y), color, 1)
        cv2.line(image, (defect.x, defect.y - 15), (defect.x, defect.y + 15), color, 1)
        
        if self.show_details:
            label = f"{defect.severity.value} {defect.deviation:.1f}px"
            text_pos = (defect.x + 25, defect.y - 10)
            
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                image,
                (text_pos[0] - 2, text_pos[1] - text_h - 2),
                (text_pos[0] + text_w + 2, text_pos[1] + 2),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                image, label, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
    
    def _draw_stats_panel(self, image: np.ndarray, stats: ProcessingStats, frame_id: int):
        h, w = image.shape[:2]
        
        panel_h = 90
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), self.COLORS['panel_bg'], -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        y_offset = 25
        lines = [
            f"Frame: {frame_id} | FPS: {stats.fps:.1f} | Proc: {stats.processing_time_ms:.1f}ms",
            f"Defects total: {stats.defects_detected} | Critical: {stats.critical_defects}",
            f"Threshold: minor>5px moderate>9px critical>15px"
        ]
        
        for line in lines:
            cv2.putText(
                image, line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1
            )
            y_offset += 25
    
    def _draw_alert_banner(self, image: np.ndarray, message: str):
        h, w = image.shape[:2]
        
        cv2.rectangle(image, (0, 0), (w, 40), (0, 0, 200), -1)
        
        cv2.putText(
            image, message,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )


class DefectDetectionPipeline:    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the defect detection pipeline.
        
        config = {
            'laser_color': 'red',
            'deviation_threshold': 5.0,
            'critical_threshold': 15.0,
            'min_defect_width': 3,
            'smoothing_window': 5,
            'temporal_frames': 3,
            'alert_cooldown': 1.0,
            'show_gui': True,
            'log_file': 'defects.log',
            'px_to_mm_ratio': None
        }
        """
        config = config or {}
        
        laser_color = config.get('laser_color', 'red')
        laser_channel = 2 if laser_color == 'red' else 1
        
        self.extractor = LaserLineExtractor(
            laser_channel=laser_channel,
            min_brightness=config.get('min_brightness', 50)
        )
        
        self.processor = LineProcessor(
            smoothing_window=config.get('smoothing_window', 5)
        )
        
        self.fitter = ReferenceLineFitter(
            ransac_threshold=config.get('deviation_threshold', 5.0)
        )
        
        self.defect_finder = DefectFinder(
            deviation_threshold=config.get('deviation_threshold', 5.0),
            critical_threshold=config.get('critical_threshold', 15.0),
            min_defect_width=config.get('min_defect_width', 3),
            px_to_mm_ratio=config.get('px_to_mm_ratio', None)
        )
        
        self.temporal_filter = TemporalFilter(
            history_size=10,
            confirmation_frames=config.get('temporal_frames', 3)
        )
        
        self.alert_system = AlertSystem(
            cooldown_seconds=config.get('alert_cooldown', 1.0),
            log_file=config.get('log_file', None)
        )
        
        self.visualizer = Visualizer(
            show_details=config.get('show_details', True),
            show_stats=config.get('show_stats', True)
        )
        
        self.stats = ProcessingStats()
        self.show_gui = config.get('show_gui', True)
        self.save_frames = config.get('save_frames', False)
        self.output_dir = config.get('output_dir', './defect_frames')
        
        if self.save_frames:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, List[Defect]]:
        """
        Process a single frame of the video stream.
        
        Returns:
            annotated_frame, defects_list
        """
        start_time = time.time()
        
        laser_points = self.extractor.extract(frame)
        
        if np.sum(~np.isnan(laser_points)) < 10:
            cv2.putText(
                frame, "NO LASER LINE DETECTED",
                (10, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
            return frame, []
        
        processed_points = self.processor.process(laser_points)
        
        slope, intercept, inlier_mask, reference_line = self.fitter.fit(processed_points)
        
        defects = self.defect_finder.find(
            processed_points, reference_line, frame_id
        )
        
        confirmed_defects = self.temporal_filter.filter(defects, frame_id)
        
        processing_time = time.time() - start_time
        self.stats.update(processing_time, confirmed_defects)
        
        alert_msg = self.alert_system.check(confirmed_defects, frame_id)
        if alert_msg:
            self.visualizer.set_alert(alert_msg)
        
        annotated_frame = self.visualizer.render(
            frame, processed_points, reference_line,
            confirmed_defects, self.stats, frame_id
        )
        
        if self.save_frames and confirmed_defects:
            filename = f"{self.output_dir}/defect_frame_{frame_id:06d}.png"
            cv2.imwrite(filename, annotated_frame)
        
        return annotated_frame, confirmed_defects
    
    def get_stats(self) -> ProcessingStats:
        return self.stats


class VideoSource:
    """Source of video stream (camera or file)"""
    
    def __init__(self, source=0, width=1280, height=720, fps=30):
        """
        Args:
            source: 0 for web camera, path to file for video
        """
        self.source = source
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        
        self.cap = None
        self.is_file = isinstance(source, str)
    
    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"Error: Failed to open source {self.source}")
            return False
        
        # Setup resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        if self.is_file:
            pass
        else:
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Source opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reading a frame"""
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """Releasing resources"""
        if self.cap:
            self.cap.release()
    
    @property
    def fps(self) -> float:
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 0.0
    
    @property
    def frame_count(self) -> int:
        if self.cap and self.is_file:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0


def generate_test_video(output_path: str, duration_sec: int = 10, fps: int = 30):
    """
    Generation of test video with laser line and defects.
    For debugging without a real camera.
    """
    width, height = 800, 600
    total_frames = duration_sec * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generation of test video: {output_path}")
    print(f"  Size: {width}x{height}, {total_frames} frames, {fps} FPS")
    
    for frame_id in range(total_frames):
        # Black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        x = np.arange(width)
        drift = 10 * np.sin(2 * np.pi * frame_id / total_frames)
        base_y = height // 2 + 50 + drift + 0.02 * x
        
        laser_y = base_y.copy()
        
        # Defect 1: (frames 30-90)
        if 30 <= frame_id <= 90:
            mask1 = (x >= 200) & (x <= 260)
            progress = min(1.0, (frame_id - 30) / 20)
            laser_y[mask1] -= 25 * progress * np.exp(-((x[mask1] - 230) ** 2) / 200)
        
        # Defect 2: (frames 100-160)
        if 100 <= frame_id <= 160:
            mask2 = (x >= 500) & (x <= 540)
            laser_y[mask2] += 35 * np.exp(-((x[mask2] - 520) ** 2) / 100)
        
        # Defect 3: (frames 180-220)
        if 180 <= frame_id <= 220:
            mask3 = (x >= 650) & (x <= 670)
            laser_y[mask3] += 20 * np.exp(-((x[mask3] - 660) ** 2) / 20)
        
        # Noise
        laser_y += np.random.normal(0, 0.3, width)
        
        # Drawing the laser line
        for i in range(width):
            yi = int(laser_y[i])
            if 0 <= yi < height:
                frame[max(0, yi-1):min(height, yi+2), i, 2] = 255
                for dy in range(-4, 5):
                    y_pos = yi + dy
                    if 0 <= y_pos < height:
                        intensity = int(150 * np.exp(-dy**2 / 4))
                        frame[y_pos, i, 2] = max(frame[y_pos, i, 2], intensity)
        
        cv2.putText(
            frame, f"Frame: {frame_id}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
        )
        
        out.write(frame)
        
        if frame_id % 50 == 0:
            print(f"  Generated {frame_id}/{total_frames} frames")
    
    out.release()
    print("  Done!")
    return output_path


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description='System for detecting defects on a laser line (video stream)'
    )
    parser.add_argument(
        '--source', '-s', type=str, default='0',
        help='Video source: 0 for camera, path to file for video'
    )
    parser.add_argument(
        '--generate-test', '-g', action='store_true',
        help='Generate test video and use it'
    )
    parser.add_argument(
        '--threshold', '-t', type=float, default=5.0,
        help='Deviation threshold for defects (pixels)'
    )
    parser.add_argument(
        '--critical', '-c', type=float, default=15.0,
        help='Critical defect threshold (pixels)'
    )
    parser.add_argument(
        '--no-gui', action='store_true',
        help='Disable display (headless mode)'
    )
    parser.add_argument(
        '--save-defects', action='store_true',
        help='Save frames with defects'
    )
    parser.add_argument(
        '--log', type=str, default='defects.log',
        help='File for defects log'
    )
    parser.add_argument(
        '--width', type=int, default=1280,
        help='Frame width'
    )
    parser.add_argument(
        '--height', type=int, default=720,
        help='Frame height'
    )
    
    args = parser.parse_args()
    
    config = {
        'laser_color': 'red',
        'deviation_threshold': args.threshold,
        'critical_threshold': args.critical,
        'min_defect_width': 3,
        'smoothing_window': 5,
        'temporal_frames': 3,
        'alert_cooldown': 1.0,
        'show_gui': not args.no_gui,
        'log_file': args.log,
        'save_frames': args.save_defects,
        'output_dir': './defect_frames',
        'show_details': True,
        'show_stats': True,
    }
    
    if args.generate_test:
        test_video_path = "test_laser_defects.mp4"
        if not os.path.exists(test_video_path):
            generate_test_video(test_video_path, duration_sec=10)
        source = test_video_path
    else:
        source = args.source
        if source.isdigit():
            source = int(source)
    
    print("=" * 60)
    print("System for detecting defects on a laser line")
    print("Real-time analysis of the laser line")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Defect threshold: {config['deviation_threshold']} px")
    print(f"  Critical threshold: {config['critical_threshold']} px")
    print(f"  Log file: {config['log_file']}")
    print(f"  GUI: {'enabled' if config['show_gui'] else 'disabled'}")
    
    video_source = VideoSource(
        source=source,
        width=args.width,
        height=args.height
    )
    
    if not video_source.open():
        return
    
    pipeline = DefectDetectionPipeline(config)
    
    print("\nControl:")
    print("  'q' or ESC - exit")
    print("  's' - save current frame")
    print("  'd' - toggle defect details")
    print("  't' - toggle statistics")
    print("  'p' - pause/resume")
    print("\nStarting processing...\n")
    
    frame_id = 0
    paused = False
    show_details = True
    show_stats = True
    
    try:
        while True:
            if not paused:
                ret, frame = video_source.read()
                
                if not ret:
                    if video_source.is_file:
                        print(f"\nEnd of video file. Processed frames: {frame_id}")
                    else:
                        print("\nVideo stream lost")
                    break
                
                annotated_frame, defects = pipeline.process_frame(frame, frame_id)
                
                frame_id += 1
                
                if defects:
                    critical = [d for d in defects if d.severity == DefectSeverity.CRITICAL]
                    if critical:
                        print(f"\rFrame {frame_id}: CRITICAL DEFECT! ({len(critical)} units)", end='')
                    else:
                        print(f"\rFrame {frame_id}: Defects: {len(defects)}", end='')
                
                if config['show_gui']:
                    cv2.imshow('Laser Defect Detection', annotated_frame)
            
            if config['show_gui']:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1
                time.sleep(0.001)
            
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('s'):
                filename = f"screenshot_{frame_id:06d}.png"
                cv2.imwrite(filename, annotated_frame)
                print(f"\nScreenshot saved: {filename}")
            elif key == ord('d'):
                show_details = not show_details
                pipeline.visualizer.show_details = show_details
                print(f"\nDefect details: {'on' if show_details else 'off'}")
            elif key == ord('t'):
                show_stats = not show_stats
                pipeline.visualizer.show_stats = show_stats
                print(f"\nStatistics: {'on' if show_stats else 'off'}")
            elif key == ord('p'):
                paused = not paused
                print(f"\n{'Pause' if paused else 'Resume'}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        stats = pipeline.get_stats()
        print("\n" + "=" * 60)
        print("Final Statistics:")
        print(f"  Total Frames: {stats.total_frames}")
        print(f"  Defects Detected: {stats.defects_detected}")
        print(f"  Critical: {stats.critical_defects}")
        print(f"  Average FPS: {stats.fps:.1f}")
        print("=" * 60)
        
        video_source.release()
        if config['show_gui']:
            cv2.destroyAllWindows()
        
        if config['log_file']:
            print(f"\nLog of defects saved to: {config['log_file']}")


if __name__ == "__main__":
    main()