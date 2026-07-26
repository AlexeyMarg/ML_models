from test_video_generator import generate_video

path = generate_video({
    'output': 'test_laser_defects.mp4',
    'laser_start': 0.3,
    'laser_end': 0.7,
    'wave_amplitude': 6.0,
    'defects': [
        {'x_center': 0.4, 'amplitude_px': 12, 'width_px': 8,
         'frame_start': 50, 'frame_peak': 80, 'frame_end': 110,
         'defect_type': 'dent'},
        {'x_center': 0.6, 'amplitude_px': 20, 'width_px': 6,
         'frame_start': 100, 'frame_peak': 130, 'frame_end': 160,
         'defect_type': 'bump'},
    ]
})