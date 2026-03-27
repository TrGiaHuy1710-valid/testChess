import cv2
import numpy as np
import os
from config import CFG


class ChessBoardProcessor:
    def __init__(self, side_step=None, config_path="inner_pts.npy"):
        self.inner_pts = None  # Toạ độ 4 góc chọn bằng tay
        self.side_step = side_step or CFG.side_step
        self.wrap_size = None
        self.config_path = config_path
        self.last_board_contour = None  # Store last detected contour for visualization

        # [P5] Cache CLAHE object — tạo 1 lần, dùng lại mỗi frame
        self._clahe = cv2.createCLAHE(
            clipLimit=CFG.clahe_clip_limit,
            tileGridSize=CFG.clahe_tile_grid
        )

        # [L2] EMA stabilization cho perspective transform
        self._prev_M1 = None

        # [P2] Cache combined transform matrix + frame skip
        self._cached_M_combined = None
        self._frame_count = 0

        # 👉 Auto load nếu đã có file
        if os.path.exists(self.config_path):
            self.inner_pts = np.load(self.config_path)
            print("✅ Loaded inner_pts from file")

    def calculate_optimal_side(self, board_contour):
        pts = board_contour.reshape(4, 2)

        def dist(p1, p2):
            return np.sqrt(np.sum((p1 - p2) ** 2))

        sides = [
            dist(pts[0], pts[3]), dist(pts[1], pts[2]),
            dist(pts[0], pts[1]), dist(pts[3], pts[2])
        ]
        max_side = max(sides)
        return int((max_side // self.side_step) * self.side_step)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        pts = pts.reshape(4, 2)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        return rect

    def get_board_contour_auto(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # [P5] Dùng CLAHE đã cache thay vì tạo mới
        contrast = self._clahe.apply(gray)
        _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_bit = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(thresh_bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > CFG.min_contour_area:
                peri = cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
                if len(approx) == 4:
                    self.last_board_contour = approx  # Store for visualization
                    return approx
        return None

    def select_and_save_inner_points(self, warped):
        points = []

        def mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) < 4:
                    points.append((x, y))
                    print(f"Point {len(points)}: {(x, y)}")
                    cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
                    cv2.imshow("Select Inner Corners", param)

        clone = warped.copy()
        cv2.imshow("Select Inner Corners", clone)
        cv2.setMouseCallback("Select Inner Corners", mouse_click, clone)

        print("👉 Click 4 góc: TL → TR → BR → BL")

        while len(points) < 4:
            if cv2.waitKey(1) & 0xFF == 27:
                print("❌ Cancel")
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()

        pts = np.array(points, dtype="float32")

        # 🔥 SAVE
        np.save(self.config_path, pts)
        print("✅ Saved inner_pts.npy")

        return pts

    def process_frame(self, frame):
        self._frame_count += 1
        final_board = None

        # [P2] Frame skip — dùng cached transform nếu có
        if (self._cached_M_combined is not None
                and self.wrap_size is not None
                and self._frame_count % CFG.frame_skip_interval != 0):
            return cv2.warpPerspective(frame, self._cached_M_combined,
                                       (self.wrap_size, self.wrap_size))

        # Full detection
        board_contour = self.get_board_contour_auto(frame)

        if board_contour is not None:
            # [L1] Cập nhật wrap_size khi thay đổi đáng kể
            new_size = self.calculate_optimal_side(board_contour)
            if self.wrap_size is None or abs(new_size - self.wrap_size) > self.wrap_size * CFG.wrap_size_update_ratio:
                self.wrap_size = new_size

            pts_src = self.order_points(board_contour)
            pts_dst = np.array([
                [0, 0], [self.wrap_size - 1, 0],
                [self.wrap_size - 1, self.wrap_size - 1], [0, self.wrap_size - 1]
            ], dtype="float32")

            M1 = cv2.getPerspectiveTransform(pts_src, pts_dst)

            # [L2] EMA stabilization — giảm rung lắc warped board
            if self._prev_M1 is not None:
                M1 = CFG.ema_alpha * self._prev_M1 + (1 - CFG.ema_alpha) * M1
            self._prev_M1 = M1.copy()

            warped = cv2.warpPerspective(frame, M1, (self.wrap_size, self.wrap_size))

            # 👉 Nếu chưa có inner_pts → bắt user chọn + save
            if self.inner_pts is None:
                print("inner_pts chưa có → yêu cầu chọn...")
                self.inner_pts = self.select_and_save_inner_points(warped)

                if self.inner_pts is None:
                    return None

            M2 = cv2.getPerspectiveTransform(self.inner_pts, pts_dst)
            final_board = cv2.warpPerspective(warped, M2, (self.wrap_size, self.wrap_size))

            # [P2] Cache combined transform cho frame skip
            # M_combined = M2 @ M1 (ghép 2 warp)
            self._cached_M_combined = M2 @ M1

        return final_board
