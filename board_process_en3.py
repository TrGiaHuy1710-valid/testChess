import cv2
import numpy as np

class ChessBoardProcessor:
    def __init__(self, side_step=10):
        self.inner_pts = None  
        self.side_step = side_step
        self.wrap_size = 500 
        self.M_refine = None 

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
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        blur = cv2.GaussianBlur(contrast, (5,5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 10000:
                peri = cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
                if len(approx) == 4:
                    return approx
        return None

    def select_inner_corners(self, warped_img):
        """
        Ép buộc việc chọn điểm trên ảnh ĐÃ WARP.
        """
        points = []
        # Hàm callback xử lý click chuột
        def mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                # Vẽ trực tiếp lên ảnh hiển thị để phản hồi người dùng
                cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("SELECT ON WARPED IMAGE", param)
                print(f"Đã chọn điểm {len(points)} trên ảnh Warp: ({x}, {y})")

        img_to_show = warped_img.copy()
        window_name = "SELECT ON WARPED IMAGE"
        
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 100, 100) # Đưa cửa sổ ra vị trí dễ thấy
        cv2.setMouseCallback(window_name, mouse_click, img_to_show)
        
        print("\n[HÀNH ĐỘNG] Click 4 góc TRÊN CỬA SỔ WARP (TL -> TR -> BR -> BL)")
        
        while len(points) < 4:
            cv2.imshow(window_name, img_to_show)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break # Nhấn ESC để thoát nếu lỗi
            
        cv2.destroyWindow(window_name)
        return np.array(points, dtype="float32")

    def process_video(self, video_path, delay_ms=100):
        cap = cv2.VideoCapture(video_path)
        is_setup_done = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (800, 600))
            board_contour = self.get_board_contour_auto(frame)
            
            if board_contour is not None:
                # BƯỚC 1: WARP THÔ TỪ ẢNH GỐC (Dựa trên contour tự động)
                pts_src = self.order_points(board_contour)
                pts_dst = np.array([
                    [0, 0], [self.wrap_size-1, 0],
                    [self.wrap_size-1, self.wrap_size-1], [0, self.wrap_size-1]
                ], dtype="float32")
                
                M_auto = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped_raw = cv2.warpPerspective(frame, M_auto, (self.wrap_size, self.wrap_size))

                # BƯỚC 2: DỪNG LẠI ĐỂ CHỌN 4 GÓC TRÊN ẢNH ĐÃ WARP THÔ
                if not is_setup_done:
                    # Truyền ảnh warped_raw vào để chọn
                    self.inner_pts = self.select_inner_corners(warped_raw)
                    
                    # Tính ma trận tinh chỉnh từ các điểm đã chọn trên Warp thô 
                    # để biến nó thành ảnh Final vuông vắn
                    self.M_refine = cv2.getPerspectiveTransform(self.inner_pts, pts_dst)
                    is_setup_done = True
                    print("\n[XÁC NHẬN] Đã lấy tọa độ 4 góc từ ảnh Warp. Tiếp tục video...")

                # BƯỚC 3: HIỂN THỊ KẾT QUẢ CUỐI CÙNG
                final_board = cv2.warpPerspective(warped_raw, self.M_refine, (self.wrap_size, self.wrap_size))
                cv2.imshow("3. FINAL BOARD (REFINED)", final_board)
            
            # Hiển thị video gốc và Warp thô để so sánh
            cv2.imshow("1. ORIGINAL VIDEO", frame)
            
            current_delay = 1 if not is_setup_done else delay_ms
            if cv2.waitKey(current_delay) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = ChessBoardProcessor()
    processor.process_video('chess_move.mp4', delay_ms=100)