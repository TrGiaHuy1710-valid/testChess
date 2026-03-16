import cv2
import numpy as np
import matplotlib.pyplot as plt

class ChessBoardProcessor:
    def __init__(self, side_step=10):
        self.inner_pts = None  # Toạ độ 4 góc chọn bằng tay
        self.side_step = side_step
        self.wrap_size = None

    def calculate_optimal_side(self, board_contour):
        pts = board_contour.reshape(4, 2)
        def dist(p1, p2):
            return np.sqrt(np.sum((p1 - p2)**2))
        
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
        rect[0] = pts[np.argmin(s)]     # Top-left
        rect[2] = pts[np.argmax(s)]     # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        return rect

    def get_board_contour_auto(self, frame):
        # Pipeline xử lý ảnh bạn đã viết
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_bit = cv2.bitwise_not(thresh)
        
        contours, _ = cv2.findContours(thresh_bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 5000:
                peri = cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
                if len(approx) == 4:
                    return approx
        return None

    def select_inner_corners(self, warped_img):
        points = []
        def mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Select Inner Corners", param)

        clone = warped_img.copy()
        cv2.imshow("Select Inner Corners", clone)
        cv2.setMouseCallback("Select Inner Corners", mouse_click, clone)
        print("-> Click 4 góc bàn cờ (TL -> TR -> BR -> BL) trên ảnh đã Warp.")
        print("-> Nhấn ESC để thoát nếu chọn sai.")
        
        while len(points) < 4:
            if cv2.waitKey(1) & 0xFF == 27: break
        
        cv2.destroyWindow("Select Inner Corners")
        return np.array(points, dtype="float32")

    # def process_video(self, video_path, delay_ms=500, step = 5): # delay_ms=500 tương đương 2 FPS
    #     cap = cv2.VideoCapture(video_path)
    #     current_frame = 0
    #     first_frame = True
        
    #     while cap.isOpened():
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    #         ret, frame = cap.read()
    #         if not ret: break
            
    #         # frame = cv2.resize(frame, (800, 600))
            
    #         # 1. Tự động tìm khung bàn cờ
    #         board_contour = self.get_board_contour_auto(frame)
            
    #         if board_contour is not None:
    #             # Tính size và Warp lần 1
    #             if self.wrap_size is None:
    #                 self.wrap_size = self.calculate_optimal_side(board_contour)
                
    #             pts_src = self.order_points(board_contour)
    #             pts_dst = np.array([
    #                 [0, 0], [self.wrap_size-1, 0],
    #                 [self.wrap_size-1, self.wrap_size-1], [0, self.wrap_size-1]
    #             ], dtype="float32")
                
    #             M1 = cv2.getPerspectiveTransform(pts_src, pts_dst)
    #             warped = cv2.warpPerspective(frame, M1, (self.wrap_size, self.wrap_size))
                
    #             # 2. Nếu là frame đầu tiên, yêu cầu chọn 4 góc tay (Inner)
    #             if self.inner_pts is None:
    #                 self.inner_pts = self.select_inner_corners(warped)
                
    #             # 3. Warp lần 2 (Refine) dựa trên toạ độ tay đã lưu
    #             M2 = cv2.getPerspectiveTransform(self.inner_pts, pts_dst)
    #             final_board = cv2.warpPerspective(warped, M2, (self.wrap_size, self.wrap_size))
                
    #             # Hiển thị kết quả
    #             cv2.imshow("Final Refined Board", final_board)
            
    #         cv2.imshow("Original Video", frame)
            
    #         current_frame += step
    #         # Điều khiển FPS (1-2 fps)
    #         if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
    #             break
                
    #     cap.release()
    #     cv2.destroyAllWindows()

    def process_video(self, video_path, step=5):
        cap = cv2.VideoCapture(video_path)
        
        # Lấy FPS gốc của video để tính toán thời gian thực
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        # Để video chạy đúng tốc độ thực tế khi đã skip frame:
        # Nếu skip 5 frame, mỗi frame xử lý đại diện cho 5 frame gốc.
        # Thời gian chờ nên để rất nhỏ (1ms) vì việc xử lý đã chiếm thời gian rồi.
        actual_delay = 1 

        while cap.isOpened():
            # Bỏ qua (step - 1) khung hình
            for _ in range(step - 1):
                cap.grab() # Chỉ lấy gói tin, không giải mã (rất nhanh)
            
            # Đọc và giải mã khung hình thứ 'step'
            ret, frame = cap.read()
            if not ret: break
            
            # --- BẮT ĐẦU XỬ LÝ FRAME ---
            # (Giữ nguyên logic xử lý của bạn ở đây)
            board_contour = self.get_board_contour_auto(frame)
            
            if board_contour is not None:
                if self.wrap_size is None:
                    self.wrap_size = self.calculate_optimal_side(board_contour)
                
                pts_src = self.order_points(board_contour)
                pts_dst = np.array([
                    [0, 0], [self.wrap_size-1, 0],
                    [self.wrap_size-1, self.wrap_size-1], [0, self.wrap_size-1]
                ], dtype="float32")
                
                M1 = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped = cv2.warpPerspective(frame, M1, (self.wrap_size, self.wrap_size))
                
                if self.inner_pts is None:
                    self.inner_pts = self.select_inner_corners(warped)
                
                M2 = cv2.getPerspectiveTransform(self.inner_pts, pts_dst)
                final_board = cv2.warpPerspective(warped, M2, (self.wrap_size, self.wrap_size))
                
                cv2.imshow("Final Refined Board", final_board)
            
            cv2.imshow("Original Video (Skipped)", frame)
            # --- KẾT THÚC XỬ LÝ ---

            if cv2.waitKey(actual_delay) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



# --- CHẠY CHƯƠNG TRÌNH ---
processor = ChessBoardProcessor(side_step=10)
# Thay 'path_to_video.mp4' bằng file của bạn
processor.process_video(0, step=5)