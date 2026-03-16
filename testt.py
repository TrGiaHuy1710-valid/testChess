import cv2
import numpy as np
import chess
import os

# --- CẤU HÌNH ---
IMG_SIZE = 600 

class ChessMoveDetector:
    def __init__(self):
        self.board = chess.Board()
        self.rect = None 
        self.h_grid = [] 
        self.v_grid = []
        self.prev_warp = None 

    def get_warped_board(self, img):
        """Tiền xử lý và cắt lấy khung hình bàn cờ"""
        img_res = cv2.resize(img, (800, 600))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(2.0, (8,8))
        contrast = clahe.apply(gray)
        _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None
        largest = max(contours, key=cv2.contourArea)
        
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) != 4:
            approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
            if len(approx) != 4: return None, None

        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] 
        rect[2] = pts[np.argmax(s)] 
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] 
        rect[3] = pts[np.argmax(diff)] 

        dst = np.array([[0,0], [IMG_SIZE-1, 0], [IMG_SIZE-1, IMG_SIZE-1], [0, IMG_SIZE-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(img_res, M, (IMG_SIZE, IMG_SIZE))
        
        return warp, rect

    def calibrate(self, frame):
        """Xác định lưới tọa độ từ frame hiện tại"""
        warp, self.rect = self.get_warped_board(frame)
        if warp is None: return False

        warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(warp_gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 120)

        h_lines, v_lines = [], []
        if lines is not None:
            for l in lines:
                rho, theta = l[0]
                if np.pi/4 < theta < 3*np.pi/4: h_lines.append(rho)
                else: v_lines.append(rho)

        def cluster_lines(data):
            data.sort()
            if not data: return []
            res = [data[0]]
            for i in range(1, len(data)):
                if abs(data[i] - data[i-1]) > 30:
                    res.append(data[i])
            return res

        self.h_grid = sorted(cluster_lines(h_lines))
        self.v_grid = sorted(cluster_lines(v_lines))

        if len(self.h_grid) < 9: self.h_grid = np.linspace(0, IMG_SIZE, 9)
        if len(self.v_grid) < 9: self.v_grid = np.linspace(0, IMG_SIZE, 9)
        
        self.prev_warp = warp 
        return True

    def draw_grid_on_frame(self, frame):
        """Vẽ trực tiếp Grid lên frame thô (sau khi đã Warp)"""
        warp, _ = self.get_warped_board(frame)
        if warp is None: return None
        
        # Vẽ các đường ngang (Xanh lá) và dọc (Đỏ)
        for h in self.h_grid:
            cv2.line(warp, (0, int(h)), (IMG_SIZE, int(h)), (0, 255, 0), 2)
        for v in self.v_grid:
            cv2.line(warp, (int(v), 0), (int(v), IMG_SIZE), (0, 0, 255), 2)
        return warp

    def process_move(self, current_frame):
        """So sánh sự thay đổi giữa frame mốc và frame hiện tại"""
        if self.rect is None or self.prev_warp is None:
            return None

        dst = np.array([[0,0], [IMG_SIZE-1, 0], [IMG_SIZE-1, IMG_SIZE-1], [0, IMG_SIZE-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(self.rect, dst)
        
        warp_c = cv2.warpPerspective(cv2.resize(current_frame, (800,600)), M, (IMG_SIZE, IMG_SIZE))
        
        diff = cv2.absdiff(cv2.cvtColor(self.prev_warp, cv2.COLOR_BGR2GRAY), 
                           cv2.cvtColor(warp_c, cv2.COLOR_BGR2GRAY))
        _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

        scores = []
        for r in range(8):
            for c in range(8):
                y1, y2 = int(self.h_grid[r]), int(self.h_grid[r+1])
                x1, x2 = int(self.v_grid[c]), int(self.v_grid[c+1])
                roi = thresh[y1:y2, x1:x2]
                score = cv2.countNonZero(roi)
                sq_idx = chess.square(c, 7 - r)
                scores.append((sq_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_squares = [s[0] for s in scores[:5]]

        detected_move = None
        for move in self.board.legal_moves:
            if move.from_square in top_squares and move.to_square in top_squares:
                detected_move = move
                break
        
        self.prev_warp = warp_c 
        return detected_move

# --- CHƯƠNG TRÌNH CHÍNH ---
cap = cv2.VideoCapture(0)
detector = ChessMoveDetector()
show_grid = False

print("Hướng dẫn điều chỉnh:")
print(" - Nhấn 'g': Bật/Tắt chế độ Preview Grid (Dùng để chỉnh camera)")
print(" - Nhấn 'i': Chốt nước đi hiện tại")
print(" - Nhấn 'q': Thoát")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    display_frame = frame.copy()
    key = cv2.waitKey(1) & 0xFF
    
    # 1. Nhấn 'g' để bật/tắt hiển thị Grid liên tục
    if key == ord('g'):
        show_grid = not show_grid
        if show_grid:
            # Mỗi lần bật 'g', tính toán lại Grid theo vị trí camera mới
            success = detector.calibrate(frame)
            if not success:
                print("⚠️ Không tìm thấy bàn cờ để vẽ Grid!")
                show_grid = False
            else:
                print("🔍 Chế độ Preview Grid: ON")
        else:
            print("⏹️ Chế độ Preview Grid: OFF")
        
    # 2. Nhấn 'i' để phát hiện nước đi
    if key == ord('i'):
        # Tự động calibrate nếu chưa từng làm hoặc để cập nhật frame mốc mới nhất
        if detector.rect is None:
            detector.calibrate(frame)
        
        move = detector.process_move(frame)
        if move:
            print(f"✅ Nước đi: {move}")
            detector.board.push(move)
            print(detector.board)
        else:
            print("❌ Không tìm thấy nước đi hợp lệ.")

    # Xử lý hiển thị
    if show_grid:
        # Vẽ Grid liên tục lên một cửa sổ riêng để soi camera
        grid_view = detector.draw_grid_on_frame(frame)
        if grid_view is not None:
            cv2.imshow("Adjust Camera - Grid View", grid_view)
    else:
        # Đóng cửa sổ Grid khi tắt chế độ preview
        if cv2.getWindowProperty("Adjust Camera - Grid View", cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow("Adjust Camera - Grid View")

    # Hiển thị thông tin lên frame chính
    status = "PREVIEW MODE" if show_grid else "NORMAL MODE"
    cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("Main Camera", display_frame)
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
