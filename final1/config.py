"""
Cấu hình tập trung cho toàn bộ project Chess Board Detection.
Thay đổi các thông số tại đây thay vì sửa trực tiếp trong code.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # === Board Detection (board_process_en_new.py) ===
    min_contour_area: int = 5000            # Diện tích contour tối thiểu để coi là bàn cờ
    clahe_clip_limit: float = 2.0           # CLAHE clip limit
    clahe_tile_grid: tuple = (8, 8)         # CLAHE tile grid size
    side_step: int = 10                     # Bước làm tròn wrap_size
    ema_alpha: float = 0.85                 # Hệ số EMA stabilization cho perspective transform
    wrap_size_update_ratio: float = 0.1     # Cập nhật wrap_size khi thay đổi > 10%
    frame_skip_interval: int = 5            # Chỉ detect contour mỗi N frame

    # === Move Detection (move_detect.py) ===
    diff_threshold: int = 40                # Ngưỡng binary threshold cho diff ảnh
    blur_kernel: tuple = (5, 5)             # GaussianBlur kernel size
    change_ratio: float = 0.05             # Tỉ lệ % diện tích ô để coi là thay đổi
    hough_threshold: int = 110              # Ngưỡng HoughLines
    hough_cluster_gap: int = 40             # Khoảng cách tối thiểu giữa 2 đường kẻ (px)
    top_changes_count: int = 6              # Số ô thay đổi lớn nhất xét khi infer move

    # === Auto-detect (F1) ===
    auto_detect_enabled: bool = False       # Bật/tắt auto-detect nước đi
    auto_stable_frames: int = 15            # Số frame ổn định trước khi auto-confirm
    auto_change_threshold: float = 0.02     # Ngưỡng thay đổi tổng thể giữa 2 frame liên tiếp

    # === Visualizer ===
    piece_dir: str = "assets/pieces"        # Thư mục chứa asset quân cờ
    square_size: int = 62                   # Kích thước mỗi ô khi render
    board_visual_size: int = 500            # Kích thước output của visual board

    # === Display (main.py) ===
    display_width: int = 640
    display_height: int = 480
    camera_retry_limit: int = 30            # Số frame retry trước khi ngắt camera


# Singleton instance
CFG = Config()
