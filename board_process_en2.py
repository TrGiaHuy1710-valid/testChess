import cv2
import numpy as np

class ChessBoardProcessor:
    def __init__(self, side_step=10):
        self.inner_pts = None  # Toạ độ 4 góc chọn bằng tay
        self.side_step = side_step
        self.wrap_size = None
        self.M_refine = None # Lưu ma trận tinh chỉnh từ frame đầu

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Tăng cường độ tương phản để tìm contour tốt hơn
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        blur = cv2.GaussianBlur(contrast, (5,5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
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
                cv2.imshow("Setup: Select 4 Corners", param)

        clone = warped_img.copy()
        cv2.imshow("Setup: Select 4 Corners", clone)
        cv2.setMouseCallback("Setup: Select 4 Corners", mouse_click, clone)
        
        print("-> ĐANG DỪNG VIDEO. Hãy click 4 góc (TL, TR, BR, BL).")
        while len(points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break # ESC để thoát
        
        cv2.destroyWindow("Setup: Select 4 Corners")
        return np.array(points, dtype="float32")

    def process_video(self, video_path, delay_ms=500):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Không thể mở video!")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (800, 600))
            board_contour = self.get_board_contour_auto(frame)
            
            if board_contour is not None:
                # 1. Tính toán kích thước chuẩn (chỉ làm 1 lần)
                if self.wrap_size is None:
                    # Tạm thời lấy size cố định hoặc tính từ contour frame đầu
                    self.wrap_size = 500 

                # 2. Warp thô (tự động theo contour)
                pts_src = self.order_points(board_contour)
                pts_dst = np.array([
                    [0, 0], [self.wrap_size-1, 0],
                    [self.wrap_size-1, self.wrap_size-1], [0, self.wrap_size-1]
                ], dtype="float32")
                
                M_auto = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped_raw = cv2.warpPerspective(frame, M_auto, (self.wrap_size, self.wrap_size))

                # 3. Bước quan trọng: Lấy frame đầu tiên để lấy tọa độ tay
                if self.inner_pts is None:
                    self.inner_pts = self.select_inner_corners(warped_raw)
                    # Tính ma trận tinh chỉnh M_refine cố định từ đây
                    self.M_refine = cv2.getPerspectiveTransform(self.inner_pts, pts_dst)

                # 4. Áp dụng M_refine cho mọi frame sau
                # Kết quả là bàn cờ luôn được căn lề chuẩn theo ý bạn đã chọn ở frame 1
                final_board = cv2.warpPerspective(warped_raw, self.M_refine, (self.wrap_size, self.wrap_size))
                
                cv2.imshow("Final Board (Refined)", final_board)
            
            cv2.imshow("Processing Video", frame)
            
            if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Chạy
processor = ChessBoardProcessor()
processor.process_video('chess_move.mp4', delay_ms=100)