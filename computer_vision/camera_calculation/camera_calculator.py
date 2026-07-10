import sys
import math
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QRadioButton, QButtonGroup,
    QGroupBox, QGridLayout, QTabWidget, QTextEdit,
    QComboBox
)
from PySide6.QtCore import Qt


class OpticsCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optics and Field of View Calculator v3.6")
        self.setMinimumSize(1050, 720)

        self.setStyleSheet("""
            QLineEdit {
                min-width: 90px;
                font-size: 14px;
                padding: 4px 6px;
            }
            QComboBox {
                min-width: 200px;
                font-size: 14px;
                padding: 4px;
            }
            QPushButton {
                min-height: 36px;
                font-size: 14px;
                padding: 6px 16px;
            }
            QLabel {
                font-size: 14px;
            }
            QGroupBox {
                font-size: 15px;
                font-weight: bold;
                padding-top: 8px;
                margin-top: 4px;
            }
            QTextEdit {
                font-size: 14px;
            }
            QRadioButton {
                font-size: 14px;
                spacing: 6px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        tab1 = QWidget()
        tabs.addTab(tab1, "📏 Task 1: Size in Pixels")
        self.setup_forward_tab(tab1)

        tab2 = QWidget()
        tabs.addTab(tab2, "📐 Task 2: Max Distance")
        self.setup_reverse_tab(tab2)

        tab3 = QWidget()
        tabs.addTab(tab3, "🔍 Task 3: Focal Length")
        self.setup_focal_length_tab(tab3)

    def create_lineedit(self, text="", min_width=90):
        le = QLineEdit(text)
        le.setMinimumWidth(min_width)
        return le

    # ----------------------------------------------------------------------
    # Task 1: Size in Pixels
    # ----------------------------------------------------------------------
    def setup_forward_tab(self, tab):
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        mode_group = QGroupBox("Calculation Mode")
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Object type:"))
        self.fwd_mode_combo = QComboBox()
        self.fwd_mode_combo.addItem("Distant object (d >> f) – simplified formula")
        self.fwd_mode_combo.addItem("Close object (d ≈ f) – exact thin lens formula")
        self.fwd_mode_combo.currentIndexChanged.connect(self.on_fwd_mode_changed)
        mode_layout.addWidget(self.fwd_mode_combo)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        matrix_group = QGroupBox("Sensor Parameters")
        matrix_vbox = QVBoxLayout()
        matrix_vbox.setSpacing(6)
        matrix_vbox.setContentsMargins(4, 4, 4, 4)

        radio_layout = QHBoxLayout()
        radio_layout.setSpacing(10)
        self.fwd_method_group = QButtonGroup()
        self.fwd_pixel_size_radio = QRadioButton("Pixel size (µm)")
        self.fwd_sensor_radio = QRadioButton("Sensor size and resolution")
        self.fwd_pixel_size_radio.setChecked(True)
        self.fwd_method_group.addButton(self.fwd_pixel_size_radio, 0)
        self.fwd_method_group.addButton(self.fwd_sensor_radio, 1)
        radio_layout.addWidget(self.fwd_pixel_size_radio)
        radio_layout.addWidget(self.fwd_sensor_radio)
        radio_layout.addStretch()
        matrix_vbox.addLayout(radio_layout)

        matrix_grid = QGridLayout()
        matrix_grid.setVerticalSpacing(8)
        matrix_grid.setHorizontalSpacing(10)

        matrix_grid.addWidget(QLabel("Pixel size (µm):"), 0, 0)
        self.fwd_pixel_size = self.create_lineedit("3.45")
        matrix_grid.addWidget(self.fwd_pixel_size, 0, 1)

        matrix_grid.addWidget(QLabel("Sensor width (mm):"), 1, 0)
        self.fwd_sensor_width = self.create_lineedit("36.0")
        matrix_grid.addWidget(self.fwd_sensor_width, 1, 1)

        matrix_grid.addWidget(QLabel("Sensor height (mm):"), 2, 0)
        self.fwd_sensor_height = self.create_lineedit("24.0")
        matrix_grid.addWidget(self.fwd_sensor_height, 2, 1)

        matrix_grid.addWidget(QLabel("Horizontal resolution:"), 3, 0)
        self.fwd_res_x = self.create_lineedit("6000")
        matrix_grid.addWidget(self.fwd_res_x, 3, 1)

        matrix_grid.addWidget(QLabel("Vertical resolution:"), 4, 0)
        self.fwd_res_y = self.create_lineedit("4000")
        matrix_grid.addWidget(self.fwd_res_y, 4, 1)

        matrix_vbox.addLayout(matrix_grid)
        matrix_group.setLayout(matrix_vbox)
        layout.addWidget(matrix_group)

        shot_group = QGroupBox("Shooting Parameters")
        shot_grid = QGridLayout()
        shot_grid.setVerticalSpacing(8)
        shot_grid.setHorizontalSpacing(10)
        shot_grid.setContentsMargins(4, 4, 4, 4)

        shot_grid.addWidget(QLabel("Focal length (mm):"), 0, 0)
        self.fwd_focal_length = self.create_lineedit("50")
        shot_grid.addWidget(self.fwd_focal_length, 0, 1)

        shot_grid.addWidget(QLabel("Distance to object:"), 1, 0)
        self.fwd_distance_label = QLabel("m")
        shot_grid.addWidget(self.fwd_distance_label, 1, 2)
        self.fwd_distance = self.create_lineedit("10")
        shot_grid.addWidget(self.fwd_distance, 1, 1)

        shot_grid.addWidget(QLabel("Object width (m):"), 2, 0)
        self.fwd_object_width = self.create_lineedit("2.0")
        shot_grid.addWidget(self.fwd_object_width, 2, 1)
        shot_grid.addWidget(QLabel("Object height (m):"), 3, 0)
        self.fwd_object_height = self.create_lineedit("1.5")
        shot_grid.addWidget(self.fwd_object_height, 3, 1)

        self.fwd_info_label = QLabel()
        self.fwd_info_label.setWordWrap(True)
        self.fwd_info_label.setStyleSheet("color: #666; font-style: italic;")
        shot_grid.addWidget(self.fwd_info_label, 4, 0, 1, 3)

        shot_group.setLayout(shot_grid)
        layout.addWidget(shot_group)

        calc_btn = QPushButton("Calculate Size in Pixels")
        calc_btn.clicked.connect(self.calculate_forward)
        layout.addWidget(calc_btn)

        self.fwd_result = QTextEdit()
        self.fwd_result.setReadOnly(True)
        layout.addWidget(self.fwd_result)

        layout.addStretch()

        self.fwd_pixel_size_radio.toggled.connect(self.toggle_fwd_fields)
        self.toggle_fwd_fields()
        self.on_fwd_mode_changed(0)

    # ----------------------------------------------------------------------
    # Task 2: Max Distance
    # ----------------------------------------------------------------------
    def setup_reverse_tab(self, tab):
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        mode_group = QGroupBox("Calculation Mode")
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Object type:"))
        self.rev_mode_combo = QComboBox()
        self.rev_mode_combo.addItem("Distant object (d >> f) – simplified formula")
        self.rev_mode_combo.addItem("Close object (d ≈ f) – exact thin lens formula")
        self.rev_mode_combo.currentIndexChanged.connect(self.on_rev_mode_changed)
        mode_layout.addWidget(self.rev_mode_combo)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        matrix_group = QGroupBox("Sensor Parameters")
        matrix_vbox = QVBoxLayout()
        matrix_vbox.setSpacing(6)
        matrix_vbox.setContentsMargins(4, 4, 4, 4)

        radio_layout = QHBoxLayout()
        radio_layout.setSpacing(10)
        self.rev_method_group = QButtonGroup()
        self.rev_pixel_size_radio = QRadioButton("Pixel size (µm)")
        self.rev_sensor_radio = QRadioButton("Sensor size and resolution")
        self.rev_pixel_size_radio.setChecked(True)
        self.rev_method_group.addButton(self.rev_pixel_size_radio, 0)
        self.rev_method_group.addButton(self.rev_sensor_radio, 1)
        radio_layout.addWidget(self.rev_pixel_size_radio)
        radio_layout.addWidget(self.rev_sensor_radio)
        radio_layout.addStretch()
        matrix_vbox.addLayout(radio_layout)

        matrix_grid = QGridLayout()
        matrix_grid.setVerticalSpacing(8)
        matrix_grid.setHorizontalSpacing(10)

        matrix_grid.addWidget(QLabel("Pixel size (µm):"), 0, 0)
        self.rev_pixel_size = self.create_lineedit("3.45")
        matrix_grid.addWidget(self.rev_pixel_size, 0, 1)

        matrix_grid.addWidget(QLabel("Sensor width (mm):"), 1, 0)
        self.rev_sensor_width = self.create_lineedit("36.0")
        matrix_grid.addWidget(self.rev_sensor_width, 1, 1)

        matrix_grid.addWidget(QLabel("Sensor height (mm):"), 2, 0)
        self.rev_sensor_height = self.create_lineedit("24.0")
        matrix_grid.addWidget(self.rev_sensor_height, 2, 1)

        matrix_grid.addWidget(QLabel("Horizontal resolution:"), 3, 0)
        self.rev_res_x = self.create_lineedit("6000")
        matrix_grid.addWidget(self.rev_res_x, 3, 1)

        matrix_grid.addWidget(QLabel("Vertical resolution:"), 4, 0)
        self.rev_res_y = self.create_lineedit("4000")
        matrix_grid.addWidget(self.rev_res_y, 4, 1)

        matrix_vbox.addLayout(matrix_grid)
        matrix_group.setLayout(matrix_vbox)
        layout.addWidget(matrix_group)

        req_group = QGroupBox("Image Requirements")
        req_grid = QGridLayout()
        req_grid.setVerticalSpacing(8)
        req_grid.setHorizontalSpacing(10)
        req_grid.setContentsMargins(4, 4, 4, 4)

        req_grid.addWidget(QLabel("Focal length (mm):"), 0, 0)
        self.rev_focal_length = self.create_lineedit("50")
        req_grid.addWidget(self.rev_focal_length, 0, 1)

        req_grid.addWidget(QLabel("Min. size (pixels) horiz.:"), 1, 0)
        self.rev_min_pixels_x = self.create_lineedit("200")
        req_grid.addWidget(self.rev_min_pixels_x, 1, 1)

        req_grid.addWidget(QLabel("Min. size (pixels) vert.:"), 2, 0)
        self.rev_min_pixels_y = self.create_lineedit("150")
        req_grid.addWidget(self.rev_min_pixels_y, 2, 1)

        req_grid.addWidget(QLabel("Real object width (m):"), 3, 0)
        self.rev_obj_width = self.create_lineedit("2.0")
        req_grid.addWidget(self.rev_obj_width, 3, 1)

        req_grid.addWidget(QLabel("Real object height (m):"), 4, 0)
        self.rev_obj_height = self.create_lineedit("1.5")
        req_grid.addWidget(self.rev_obj_height, 4, 1)

        self.rev_info_label = QLabel()
        self.rev_info_label.setWordWrap(True)
        self.rev_info_label.setStyleSheet("color: #666; font-style: italic;")
        req_grid.addWidget(self.rev_info_label, 5, 0, 1, 3)

        req_group.setLayout(req_grid)
        layout.addWidget(req_group)

        calc_btn = QPushButton("Calculate Maximum Distance")
        calc_btn.clicked.connect(self.calculate_reverse)
        layout.addWidget(calc_btn)

        self.rev_result = QTextEdit()
        self.rev_result.setReadOnly(True)
        layout.addWidget(self.rev_result)

        layout.addStretch()

        self.rev_pixel_size_radio.toggled.connect(self.toggle_rev_fields)
        self.toggle_rev_fields()
        self.on_rev_mode_changed(0)

    # ----------------------------------------------------------------------
    # Task 3: Focal Length
    # ----------------------------------------------------------------------
    def setup_focal_length_tab(self, tab):
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        mode_group = QGroupBox("Calculation Mode")
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Object type:"))
        self.fl_mode_combo = QComboBox()
        self.fl_mode_combo.addItem("Distant object – simplified formula")
        self.fl_mode_combo.addItem("Close object – exact formula")
        self.fl_mode_combo.currentIndexChanged.connect(self.on_fl_mode_changed)
        mode_layout.addWidget(self.fl_mode_combo)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        matrix_group = QGroupBox("Sensor Parameters")
        matrix_vbox = QVBoxLayout()
        matrix_vbox.setSpacing(6)
        matrix_vbox.setContentsMargins(4, 4, 4, 4)

        radio_layout = QHBoxLayout()
        radio_layout.setSpacing(10)
        self.fl_method_group = QButtonGroup()
        self.fl_pixel_size_radio = QRadioButton("Pixel size (µm)")
        self.fl_sensor_radio = QRadioButton("Sensor size and resolution")
        self.fl_pixel_size_radio.setChecked(True)
        self.fl_method_group.addButton(self.fl_pixel_size_radio, 0)
        self.fl_method_group.addButton(self.fl_sensor_radio, 1)
        radio_layout.addWidget(self.fl_pixel_size_radio)
        radio_layout.addWidget(self.fl_sensor_radio)
        radio_layout.addStretch()
        matrix_vbox.addLayout(radio_layout)

        matrix_grid = QGridLayout()
        matrix_grid.setVerticalSpacing(8)
        matrix_grid.setHorizontalSpacing(10)

        matrix_grid.addWidget(QLabel("Pixel size (µm):"), 0, 0)
        self.fl_pixel_size = self.create_lineedit("3.45")
        matrix_grid.addWidget(self.fl_pixel_size, 0, 1)

        matrix_grid.addWidget(QLabel("Sensor width (mm):"), 1, 0)
        self.fl_sensor_width = self.create_lineedit("36.0")
        matrix_grid.addWidget(self.fl_sensor_width, 1, 1)

        matrix_grid.addWidget(QLabel("Sensor height (mm):"), 2, 0)
        self.fl_sensor_height = self.create_lineedit("24.0")
        matrix_grid.addWidget(self.fl_sensor_height, 2, 1)

        matrix_grid.addWidget(QLabel("Horizontal resolution:"), 3, 0)
        self.fl_res_x = self.create_lineedit("6000")
        matrix_grid.addWidget(self.fl_res_x, 3, 1)

        matrix_grid.addWidget(QLabel("Vertical resolution:"), 4, 0)
        self.fl_res_y = self.create_lineedit("4000")
        matrix_grid.addWidget(self.fl_res_y, 4, 1)

        matrix_vbox.addLayout(matrix_grid)
        matrix_group.setLayout(matrix_vbox)
        layout.addWidget(matrix_group)

        input_group = QGroupBox("Problem Conditions")
        input_grid = QGridLayout()
        input_grid.setVerticalSpacing(8)
        input_grid.setHorizontalSpacing(10)
        input_grid.setContentsMargins(4, 4, 4, 4)

        input_grid.addWidget(QLabel("Distance to object:"), 0, 0)
        self.fl_distance_label = QLabel("m")
        input_grid.addWidget(self.fl_distance_label, 0, 2)
        self.fl_distance = self.create_lineedit("50")
        input_grid.addWidget(self.fl_distance, 0, 1)

        input_grid.addWidget(QLabel("Object width (m):"), 1, 0)
        self.fl_obj_width = self.create_lineedit("3.0")
        input_grid.addWidget(self.fl_obj_width, 1, 1)

        input_grid.addWidget(QLabel("Object height (m):"), 2, 0)
        self.fl_obj_height = self.create_lineedit("2.0")
        input_grid.addWidget(self.fl_obj_height, 2, 1)

        input_grid.addWidget(QLabel("Min. size (pixels) horiz.:"), 3, 0)
        self.fl_min_pix_x = self.create_lineedit("500")
        input_grid.addWidget(self.fl_min_pix_x, 3, 1)

        input_grid.addWidget(QLabel("Min. size (pixels) vert.:"), 4, 0)
        self.fl_min_pix_y = self.create_lineedit("400")
        input_grid.addWidget(self.fl_min_pix_y, 4, 1)

        self.fl_info_label = QLabel()
        self.fl_info_label.setWordWrap(True)
        self.fl_info_label.setStyleSheet("color: #666; font-style: italic;")
        input_grid.addWidget(self.fl_info_label, 5, 0, 1, 3)

        input_group.setLayout(input_grid)
        layout.addWidget(input_group)

        calc_btn = QPushButton("Calculate Required Focal Length")
        calc_btn.clicked.connect(self.calculate_focal_length)
        layout.addWidget(calc_btn)

        self.fl_result = QTextEdit()
        self.fl_result.setReadOnly(True)
        layout.addWidget(self.fl_result)

        layout.addStretch()

        self.fl_pixel_size_radio.toggled.connect(self.toggle_fl_fields)
        self.toggle_fl_fields()
        self.on_fl_mode_changed(0)

    # ----------------------------------------------------------------------
    # Field toggles
    # ----------------------------------------------------------------------
    def toggle_fwd_fields(self):
        is_pixel = self.fwd_pixel_size_radio.isChecked()
        self.fwd_pixel_size.setEnabled(is_pixel)
        self.fwd_sensor_width.setEnabled(not is_pixel)
        self.fwd_sensor_height.setEnabled(not is_pixel)
        self.fwd_res_x.setEnabled(not is_pixel)
        self.fwd_res_y.setEnabled(not is_pixel)

    def toggle_rev_fields(self):
        is_pixel = self.rev_pixel_size_radio.isChecked()
        self.rev_pixel_size.setEnabled(is_pixel)
        self.rev_sensor_width.setEnabled(not is_pixel)
        self.rev_sensor_height.setEnabled(not is_pixel)
        self.rev_res_x.setEnabled(not is_pixel)
        self.rev_res_y.setEnabled(not is_pixel)

    def toggle_fl_fields(self):
        is_pixel = self.fl_pixel_size_radio.isChecked()
        self.fl_pixel_size.setEnabled(is_pixel)
        self.fl_sensor_width.setEnabled(not is_pixel)
        self.fl_sensor_height.setEnabled(not is_pixel)
        self.fl_res_x.setEnabled(not is_pixel)
        self.fl_res_y.setEnabled(not is_pixel)

    # ----------------------------------------------------------------------
    # Mode change handlers
    # ----------------------------------------------------------------------
    def on_fwd_mode_changed(self, index):
        if index == 0:
            self.fwd_distance_label.setText("m")
            self.fwd_distance.setText("10")
            self.fwd_info_label.setText(
                "💡 Formula: image_size = f × object_size / distance"
            )
        else:
            self.fwd_distance_label.setText("mm")
            self.fwd_distance.setText("200")
            self.fwd_info_label.setText(
                "💡 Exact formula: 1/f = 1/d + 1/s. Distance must be > f!"
            )

    def on_rev_mode_changed(self, index):
        if index == 0:
            self.rev_info_label.setText(
                "💡 Simplified: d_max = f × object_size / min_image_size"
            )
        else:
            self.rev_info_label.setText(
                "💡 Exact: d_max = f × (1 + object_size / min_image_size)"
            )

    def on_fl_mode_changed(self, index):
        if index == 0:
            self.fl_distance_label.setText("m")
            self.fl_distance.setText("50")
            self.fl_info_label.setText(
                "💡 Formula: f = distance × min_image_size / object_size"
            )
        else:
            self.fl_distance_label.setText("mm")
            self.fl_distance.setText("500")
            self.fl_info_label.setText(
                "💡 Exact: f = distance / (1 + object_size / min_image_size)"
            )

    # ----------------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------------
    def get_pixel_sizes(self, method_radio, pixel_edit, sensor_w, sensor_h, res_x, res_y):
        if method_radio.isChecked():
            return float(pixel_edit.text()) / 1000.0
        else:
            sw = float(sensor_w.text())
            sh = float(sensor_h.text())
            rx = int(res_x.text())
            ry = int(res_y.text())
            return (sw / rx + sh / ry) / 2

    # ----------------------------------------------------------------------
    # Calculation methods
    # ----------------------------------------------------------------------
    def calculate_forward(self):
        try:
            focal_length = float(self.fwd_focal_length.text())
            is_far = self.fwd_mode_combo.currentIndex() == 0
            if is_far:
                distance = float(self.fwd_distance.text()) * 1000
            else:
                distance = float(self.fwd_distance.text())

            obj_w = float(self.fwd_object_width.text()) * 1000
            obj_h = float(self.fwd_object_height.text()) * 1000

            pixel_size = self.get_pixel_sizes(
                self.fwd_pixel_size_radio, self.fwd_pixel_size,
                self.fwd_sensor_width, self.fwd_sensor_height,
                self.fwd_res_x, self.fwd_res_y
            )

            if is_far:
                image_w_mm = focal_length * obj_w / distance
                image_h_mm = focal_length * obj_h / distance
                s = focal_length

                if distance > focal_length:
                    s_exact = focal_length * distance / (distance - focal_length)
                    M_exact = s_exact / distance
                    exact_w = M_exact * obj_w
                    exact_h = M_exact * obj_h
                    error_w = abs(image_w_mm - exact_w) / exact_w * 100
                else:
                    exact_w = None
                    error_w = None

                result_text = f"""
=== RESULTS (simplified formula) ===

📊 Image size on sensor: {image_w_mm:.3f} × {image_h_mm:.3f} mm
📊 Size in pixels: {image_w_mm/pixel_size:.1f} × {image_h_mm/pixel_size:.1f} px
"""
                if exact_w is not None:
                    result_text += f"""
=== COMPARISON WITH EXACT FORMULA ===
📊 Exact pixel size: {exact_w/pixel_size:.1f} × {exact_h/pixel_size:.1f} px
⚠️ Error of simplified formula: {error_w:.2f}%
"""
            else:
                if distance <= focal_length:
                    self.fwd_result.setText("⚠️ ERROR: d ≤ f – image is virtual!")
                    return

                s = focal_length * distance / (distance - focal_length)
                M = s / distance
                image_w_mm = M * obj_w
                image_h_mm = M * obj_h

                simple_w = focal_length * obj_w / distance
                error_w = abs(image_w_mm - simple_w) / image_w_mm * 100

                result_text = f"""
=== RESULTS (exact formula) ===

📐 Lens-to-sensor distance: {s:.2f} mm (extension: {s-focal_length:.2f} mm)
🔍 Magnification: {M:.4f}x
📊 Image size on sensor: {image_w_mm:.3f} × {image_h_mm:.3f} mm
📊 Size in pixels: {image_w_mm/pixel_size:.1f} × {image_h_mm/pixel_size:.1f} px

=== COMPARISON ===
📊 Simplified formula gives: {simple_w/pixel_size:.1f} px
⚠️ Error of simplified: {error_w:.2f}%
"""

            if self.fwd_sensor_radio.isChecked():
                sensor_w = float(self.fwd_sensor_width.text())
                sensor_h = float(self.fwd_sensor_height.text())
            else:
                sensor_w = 36.0
                sensor_h = 24.0

            fov_h = 2 * math.degrees(math.atan(sensor_w / (2 * focal_length)))
            fov_v = 2 * math.degrees(math.atan(sensor_h / (2 * focal_length)))

            result_text += f"""
=== FIELD OF VIEW ANGLES ===
🔭 Horizontal: {fov_h:.2f}°
🔭 Vertical: {fov_v:.2f}°
"""
            self.fwd_result.setText(result_text)

        except ValueError as e:
            self.fwd_result.setText(f"Input error: {e}")

    def calculate_reverse(self):
        try:
            focal_length = float(self.rev_focal_length.text())
            min_pix_x = float(self.rev_min_pixels_x.text())
            min_pix_y = float(self.rev_min_pixels_y.text())
            obj_w = float(self.rev_obj_width.text()) * 1000
            obj_h = float(self.rev_obj_height.text()) * 1000

            pixel_size = self.get_pixel_sizes(
                self.rev_pixel_size_radio, self.rev_pixel_size,
                self.rev_sensor_width, self.rev_sensor_height,
                self.rev_res_x, self.rev_res_y
            )

            min_image_w = min_pix_x * pixel_size
            min_image_h = min_pix_y * pixel_size

            is_far = self.rev_mode_combo.currentIndex() == 0

            if is_far:
                max_dist_w = focal_length * obj_w / min_image_w
                max_dist_h = focal_length * obj_h / min_image_h
                max_dist = min(max_dist_w, max_dist_h)
                formula_name = "simplified"
            else:
                max_dist_w = focal_length * (1 + obj_w / min_image_w)
                max_dist_h = focal_length * (1 + obj_h / min_image_h)
                max_dist = min(max_dist_w, max_dist_h)
                formula_name = "exact"

            result_text = f"""
=== RESULTS ({formula_name} formula) ===

📏 Maximum distance: {max_dist/1000:.2f} m

📊 Image size on sensor at this distance:
   • {min_image_w:.3f} × {min_image_h:.3f} mm
   • {min_pix_x} × {min_pix_y} pixels

💡 Beyond this distance the object will be smaller than required!
"""
            self.rev_result.setText(result_text)

        except ValueError as e:
            self.rev_result.setText(f"Input error: {e}")

    def calculate_focal_length(self):
        try:
            is_far = self.fl_mode_combo.currentIndex() == 0
            if is_far:
                distance = float(self.fl_distance.text()) * 1000
            else:
                distance = float(self.fl_distance.text())

            obj_w = float(self.fl_obj_width.text()) * 1000
            obj_h = float(self.fl_obj_height.text()) * 1000
            min_pix_x = float(self.fl_min_pix_x.text())
            min_pix_y = float(self.fl_min_pix_y.text())

            pixel_size = self.get_pixel_sizes(
                self.fl_pixel_size_radio, self.fl_pixel_size,
                self.fl_sensor_width, self.fl_sensor_height,
                self.fl_res_x, self.fl_res_y
            )

            min_image_w = min_pix_x * pixel_size
            min_image_h = min_pix_y * pixel_size

            if is_far:
                f_w = distance * min_image_w / obj_w
                f_h = distance * min_image_h / obj_h
                f_recommended = max(f_w, f_h)   # гарантирует выполнение обоих требований
                formula_details = f"""
📐 Calculation:
   f_horiz = {distance} × {min_image_w:.4f} / {obj_w} = {f_w:.2f} mm
   f_vert  = {distance} × {min_image_h:.4f} / {obj_h} = {f_h:.2f} mm
"""
            else:
                f_w = distance / (1 + obj_w / min_image_w)
                f_h = distance / (1 + obj_h / min_image_h)
                f_recommended = max(f_w, f_h)
                formula_details = f"""
📐 Calculation:
   f_horiz = {distance} / (1 + {obj_w}/{min_image_w:.4f}) = {f_w:.2f} mm
   f_vert  = {distance} / (1 + {obj_h}/{min_image_h:.4f}) = {f_h:.2f} mm
"""

            # Verification with tolerance for floating point
            if is_far:
                check_pix_w = f_recommended * obj_w / (distance * pixel_size)
                check_pix_h = f_recommended * obj_h / (distance * pixel_size)
            else:
                s = f_recommended * distance / (distance - f_recommended)
                M = s / distance
                check_pix_w = M * obj_w / pixel_size
                check_pix_h = M * obj_h / pixel_size

            # Допуск 0.01 пикселя для избежания ошибок округления
            requirements_met = (check_pix_w >= min_pix_x - 0.01) and (check_pix_h >= min_pix_y - 0.01)

            if self.fl_sensor_radio.isChecked():
                sensor_w = float(self.fl_sensor_width.text())
                sensor_h = float(self.fl_sensor_height.text())
            else:
                sensor_w = 36.0
                sensor_h = 24.0

            fov_h = 2 * math.degrees(math.atan(sensor_w / (2 * f_recommended)))
            fov_v = 2 * math.degrees(math.atan(sensor_h / (2 * f_recommended)))

            result_text = f"""
=== FOCAL LENGTH CALCULATION ===

{formula_details}

=== RECOMMENDED FOCAL LENGTH ===
🔍 f = {f_recommended:.1f} mm

📊 Verification – object size in pixels at f = {f_recommended:.1f} mm:
   • Horizontal: {check_pix_w:.1f} px (required ≥ {min_pix_x})
   • Vertical: {check_pix_h:.1f} px (required ≥ {min_pix_y})
   {'✅ Requirements met!' if requirements_met else '⚠️ Requirements NOT met!'}
"""
            if not requirements_met:
                result_text += "\n💡 Tip: Slight floating-point error may cause this; try increasing min. pixels slightly.\n"

            result_text += f"""
=== FIELD OF VIEW WITH THIS LENS ===
🔭 Horizontal angle: {fov_h:.2f}°
🔭 Vertical angle: {fov_v:.2f}°

=== FIELD OF VIEW AT {distance/1000:.2f} m ===
↔️ Width: {distance * sensor_w / f_recommended / 1000:.2f} m
↕️ Height: {distance * sensor_h / f_recommended / 1000:.2f} m

💡 Tip: choose the nearest standard focal length ≥ {f_recommended:.1f} mm
   (e.g. {self._find_nearest_standard_fl(f_recommended)})
"""
            self.fl_result.setText(result_text)

        except ValueError as e:
            self.fl_result.setText(f"Input error: {e}")

    def _find_nearest_standard_fl(self, f):
        standard_fl = [8, 10, 12, 14, 16, 18, 20, 24, 28, 35, 40, 50, 60, 70, 85,
                       100, 105, 135, 150, 180, 200, 250, 300, 400, 500, 600, 800, 1000]
        for fl in standard_fl:
            if fl >= f:
                return f"{fl} mm"
        return "1000+ mm"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OpticsCalculator()
    window.show()
    sys.exit(app.exec())