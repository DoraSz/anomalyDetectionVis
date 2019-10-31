import numpy as np
from Time_series_animation import Time_series_animation


configuration = {
    'reconstruction_errors': np.load('example_data/reconstruction_errors.npy'),
    'threshold': np.load('example_data/threshold_history.npy'),
    'class_labels': np.load('example_data/target_labels.npy'),
    'animation_window_size': 300,
    'update_interval': 20,
    'start': 0,
    'hide_classes': False,
    'adjust_y_to_threshold': True,
    'show_std': True,
    'std_window_size': 20
}

animator = Time_series_animation(**configuration)
animator.play()

#animator.save_to_file('export.gif')

