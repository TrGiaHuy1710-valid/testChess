import cv2
import numpy as np
import chess
import chess.pgn
import os

# --- CẤU HÌNH ---
PIECE_MAP = {
    'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk',
    'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk',
}

# Đổi lại đường dẫn của bạn ở đây (sử dụng r"" để tránh lỗi ký tự đặc biệt)
ASSET_PATH = r"assets\pieces"


# --- PHẦN 1: HELPER CLASSES & FUNCTIONS ---

class ChessVisualizer:
    def __init__(self, piece_dir, square_size=60):
        self.square = square_size
        self.pieces = {}
        self.has_assets = True

        if not os.path.exists(piece_dir):
            print(f"ERROR: Không tìm thấy thư mục: {piece_dir}")
            self.has_assets = False
            return

        for k, v in PIECE_MAP.items():
            path = os.path.join(piece_dir, f"{v}.png")
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                self.has_assets = False
            else:
                self.pieces[k] = cv2.resize(img, (square_size, square_size))

    def overlay(self, bg, png, x, y):
        if png.shape[2] == 4:
            alpha = png[:, :, 3] / 255.0
            for c in range(3):
                bg[y:y + self.square, x:x + self.square, c] = (
                        alpha * png[:, :, c] +
                        (1 - alpha) * bg[y:y + self.square, x:x + self.square, c]
                )
        else:
            bg[y:y + self.square, x:x + self.square] = png

    def draw_board(self, board: chess.Board, last_move=None):
        img = np.zeros((8 * self.square, 8 * self.square, 3), dtype=np.uint8)

        # 1. Vẽ ô bàn cờ
        for r in range(8):
            for c in range(8):
                color = (240, 217, 181) if (r + c) % 2 == 0 else (181, 136, 99)

                # Highlight nước đi vừa rồi (nếu có)
                if last_move:
                    sq_idx = chess.square(c, 7 - r)
                    if sq_idx == last_move.from_square or sq_idx == last_move.to_square:
                        color = (100, 200, 100)  # Màu xanh nhạt

                cv2.rectangle(
                    img,
                    (c * self.square, r * self.square),
                    ((c + 1) * self.square, (r + 1) * self.square),
                    color, -1
                )

        # 2. Vẽ quân cờ
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                symbol = piece.symbol()
                if self.has_assets and symbol in self.pieces:
                    self.overlay(img, self.pieces[symbol], c * self.square, r * self.square)
                else:
                    text_color = (0, 0, 0) if piece.color == chess.BLACK else (255, 255, 255)
                    cv2.putText(img, symbol, (c * self.square + 15, r * self.square + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        return img


def nothing(x): pass


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect



# --- LOGIC PHÁT HIỆN NƯỚC ĐI TỪ ẢNH ---

def detect_changes(prev_img, curr_img, h_grid, v_grid):
    """
    So sánh 2 ảnh dựa trên tọa độ lưới h_grid và v_grid đã được calibrate.
    """
    if prev_img is None or curr_img is None:
        return []

    # Chuyển xám
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    # Làm mờ để giảm nhiễu do rung lắc camera (jitter/noise)
    prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # Tính hiệu ảnh
    diff = cv2.absdiff(prev_blur, curr_blur)
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY) 

    changes = []
    
    # Đảm bảo chỉ duyệt trong phạm vi số đường kẻ hiện có
    rows = min(8, len(h_grid) - 1)
    cols = min(8, len(v_grid) - 1)

    for r in range(rows):
        for c in range(cols):
            y_start, y_end = int(h_grid[r]), int(h_grid[r+1])
            x_start, x_end = int(v_grid[c]), int(v_grid[c+1])

            # Đếm số pixel trắng (thay đổi) trong ô này
            roi = thresh[y_start:y_end, x_start:x_end]
            non_zero = cv2.countNonZero(roi)

            # Nếu lượng pixel thay đổi lớn hơn ngưỡng nhất định (noise filter)
            if non_zero > 100:
                # Mapping sang Chess Coordinate
                # Ảnh: r=0 là trên cùng. Chess: rank=7 là trên cùng.
                sq_idx = chess.square(c, 7 - r)
                changes.append((sq_idx, non_zero))

    # Sắp xếp các ô thay đổi nhiều nhất lên đầu
    changes.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in changes]


def infer_move(board, changed_squares):
    """
    Suy luận nước đi dựa trên các ô bị thay đổi và luật cờ vua.
    """
    if len(changed_squares) < 2:
        return None, "Not enough changes detected"

    # Lấy 2 ô thay đổi nhiều nhất (Thông thường là ô đi và ô đến)
    # Lưu ý: Nhập thành (Castling) sẽ thay đổi 4 ô. Ăn quân cũng thay đổi 2 ô.
    top_changes = changed_squares[:4]

    legal_moves = list(board.legal_moves)
    possible_moves = []

    for move in legal_moves:
        # Kiểm tra xem nước đi này có nằm trong số các ô bị thay đổi không
        # Logic đơn giản: Cả ô đi (from) và ô đến (to) đều phải có sự thay đổi hình ảnh
        if move.from_square in top_changes and move.to_square in top_changes:
            possible_moves.append(move)

    if len(possible_moves) == 1:
        return possible_moves[0], "Success"
    elif len(possible_moves) > 1:
        # Nếu có nhiều khả năng (hiếm gặp), chọn cái có độ thay đổi cao nhất (đã sort ở bước trước)
        return possible_moves[0], "Ambiguous, picked best match"

    return None, "No matching legal move found"

def draw_grid(img, h_grid=None, v_grid=None, grid_size = 500, cells = 8):
    if h_grid is None or v_grid is None or len(h_grid) == 0 or len(v_grid) == 0:
        step = grid_size // cells
        h_grid = np.linspace(0, grid_size, cells + 1)
        v_grid = np.linspace(0, grid_size, cells + 1)

    # Dọc (v_grid)
    for x in v_grid:
        cv2.line(img, (int(x), 0), (int(x), grid_size), (255, 0, 0), 1)

    # Ngang (h_grid)
    for y in h_grid:
        cv2.line(img, (0, int(y)), (grid_size, int(y)), (0, 255, 0), 1)
    return img

calibration_points = []
manual_mode = False

def mouse_callback(event, x, y, flags, param):
    global calibration_points, manual_mode
    if manual_mode and event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append((x, y))
            print(f"-> Point {len(calibration_points)} captured: ({x}, {y})")

# --- MAIN ---
def main():
    global manual_mode, calibration_points
    # Đổi 0 thành tên video file (ví dụ: "chess_move.mp4") nếu muốn test bằng video có sẵn
    VIDEO_PATH = 0
    # cap = cv2.VideoCapture(0) # Dùng cho Camera
    cap = cv2.VideoCapture(VIDEO_PATH) # Dùng cho Video File

    # Set up logic variables

    # PGN Game tracking
    game = chess.pgn.Game()
    game.headers["Event"] = "CV Chess Game"
    game.headers["White"] = "Player (Camera)"
    game.headers["Black"] = "Player (Camera)"
    node = game
    # Setup Windows
    cv2.namedWindow("Settings")
    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", mouse_callback)
    cv2.createTrackbar("Threshold", "Settings", 70, 300, nothing)
    cv2.createTrackbar("AngleDelta", "Settings", 20, 100, nothing)
    cv2.createTrackbar("InnerPad", "Settings", 5, 100, nothing)

    # Chess Logic
    board = chess.Board()
    visualizer = ChessVisualizer(piece_dir=ASSET_PATH, square_size=60)

    WARPED_SIZE = 500
    dst_pts = np.array([
        [0, 0], [WARPED_SIZE - 1, 0],
        [WARPED_SIZE - 1, WARPED_SIZE - 1], [0, WARPED_SIZE - 1]
    ], dtype="float32")

    prev_warped_img = None  # Lưu trạng thái bàn cờ trước khi đi
    current_warped_img = None  # Lưu trạng thái hiện tại (để hiển thị)

    h_grid = np.linspace(0, 500, 9) 
    v_grid = np.linspace(0, 500, 9)

    print("=== HƯỚNG DẪN ===")
    print("1. Chỉnh camera sao cho toàn bộ bàn cờ nằm trong khung hình (hệ thống sẽ bắt góc tự động).")
    print("2. Đổi qua lại chế độ tự động và thủ công bằng phím 'm'. (Nếu thủ công: dùng chuột click 4 góc, 'r' để reset)")
    print("3. Nhấn phím 'i' (Init) để lưu trạng thái bàn cờ ban đầu và Calibrate khung cờ.")
    print("4. Đi quân cờ thật.")
    print("5. Rút tay ra, nhấn phím 'SPACE' để cập nhật.")
    print("6. Nhấn phím 's' để lưu trận đấu thành file PGN.")
    print("7. Nhấn 'q' để thoát.")
    print("=================")

    prev_M = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- 1. Xử lý ảnh để tìm bàn cờ (Tiền xử lý nâng cao) ---
        img_res = cv2.resize(frame, (800, 600))
        frame_resized = img_res.copy()
        board_found = False

        if not manual_mode:
            gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
            # Tăng tương phản bằng CLAHE
            clahe = cv2.createCLAHE(2.0, (8,8))
            contrast = clahe.apply(gray)
            # Sử dụng OTSU để tự động tìm Threshold tốt cho viền
            _, thresh_img = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_img = cv2.bitwise_not(thresh_img)
            
            contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 5000:
                    peri = cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
                    
                    if len(approx) == 4:
                        cv2.drawContours(frame_resized, [approx], -1, (0, 255, 0), 2)
                        rect = order_points(approx.reshape(4, 2))
                        M = cv2.getPerspectiveTransform(rect, dst_pts)

                        # Stabilize transform matrix
                        if prev_M is None:
                            prev_M = M
                        else:
                            alpha = 0.85
                            M = alpha * prev_M + (1 - alpha) * M
                            prev_M = M

                        current_warped_img = cv2.warpPerspective(img_res, M, (WARPED_SIZE, WARPED_SIZE))
                        board_found = True
        else:
            # Chế độ thủ công
            for i, pt in enumerate(calibration_points):
                cv2.circle(frame_resized, pt, 5, (0, 0, 255), -1)
                cv2.putText(frame_resized, str(i + 1), (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if len(calibration_points) == 4:
                pts = np.array(calibration_points, dtype=np.float32)
                rect = order_points(pts)
                M = cv2.getPerspectiveTransform(rect, dst_pts)
                
                prev_M = M # Update prev_M to maintain stabilization if switching back
                current_warped_img = cv2.warpPerspective(img_res, M, (WARPED_SIZE, WARPED_SIZE))
                board_found = True

        if board_found:
            pad_percent = cv2.getTrackbarPos("InnerPad", "Settings") / 100.0
            pad = int(WARPED_SIZE * pad_percent / 2)
            inner = current_warped_img[pad: WARPED_SIZE - pad, pad: WARPED_SIZE - pad]
            inner = cv2.resize(inner, (WARPED_SIZE, WARPED_SIZE))

            # thể hiện ra grid view
            grid_view = draw_grid(inner.copy(), h_grid, v_grid, WARPED_SIZE)
            cv2.imshow("Warped View", grid_view)

            current_warped_img = inner

        # --- 2. Xử lý phím bấm ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('m'):
            manual_mode = not manual_mode
            calibration_points = []
            prev_M = None
            print(f"-> Đã chuyển sang chế độ: {'THỦ CÔNG (Click 4 điểm)' if manual_mode else 'TỰ ĐỘNG'}")
        elif key == ord('r'):
            if manual_mode:
                calibration_points = []
                print("-> Đã reset điểm chọn thủ công.")

        # Phím 'i': Khởi tạo (Lưu ảnh ban đầu và Calibrate Grid)
        elif key == ord('i'):
            if board_found:
                prev_warped_img = current_warped_img.copy()
                
                # Thực hiện Calibrate Grid
                warp_gray = cv2.cvtColor(prev_warped_img, cv2.COLOR_BGR2GRAY)
                edges_w = cv2.Canny(warp_gray, 50, 150)
                lines_w = cv2.HoughLines(edges_w, 1, np.pi/180, 110) # Dò lại đường thẳng bên trong bàn cờ phẳng
                
                h_lines, v_lines = [], []
                if lines_w is not None:
                    for l in lines_w:
                        rho, theta = l[0]
                        # Chú ý: rho có thể âm nếu theta > pi/2
                        if np.pi/4 < theta < 3*np.pi/4: h_lines.append(abs(rho))
                        else: v_lines.append(abs(rho))

                def cluster_lines(data, max_val):
                    if not data: return np.linspace(0, max_val, 9).tolist()
                    data.sort()
                    res = []
                    current_cluster = [data[0]]
                    for i in range(1, len(data)):
                        if abs(data[i] - data[i-1]) > 40: # Khoảng cách tối thiểu giữa 2 đường kẻ phải > 40px
                            res.append(sum(current_cluster)/len(current_cluster))
                            current_cluster = [data[i]]
                        else:
                            current_cluster.append(data[i])
                    res.append(sum(current_cluster)/len(current_cluster))
                    
                    # Nếu thuật toán Calibrate không tìm đủ 8 ô (9 đường viền), fall back về chia đều tránh lỗi.
                    if len(res) != 9:
                        return np.linspace(0, max_val, 9).tolist()
                    return res

                # Gom nhóm lưới thực tế
                h_grid = cluster_lines(h_lines, WARPED_SIZE)
                v_grid = cluster_lines(v_lines, WARPED_SIZE)
                
                print(f"-> Đã khởi tạo và Calibrate đường kẻ (H={len(h_grid)}, V={len(v_grid)}). (Ready)")
            else:
                print("-> Lỗi: Không tìm thấy bàn cờ để khởi tạo.")

        # Phím SPACE: Xác nhận nước đi
        elif key == 32:  # 32 là mã ASCII của Space
            if not board_found:
                print("-> Lỗi: Không nhìn thấy bàn cờ.")
            elif prev_warped_img is None:
                print("-> Lỗi: Chưa khởi tạo (Nhấn 'i' trước).")
            else:
                print("Processing move...")
                # 1. Tìm các ô thay đổi
                changes = detect_changes(prev_warped_img, current_warped_img, h_grid, v_grid)

                # 2. Suy luận nước đi
                move, msg = infer_move(board, changes)

                if move:
                    # Lấy tên quân cờ để in ra
                    piece = board.piece_at(move.from_square)
                    piece_name = piece.symbol() if piece else "Unknown"
                    color_name = "White" if piece and piece.color == chess.WHITE else "Black"

                    # Cập nhật bàn cờ ảo và Game Node
                    board.push(move)
                    node = node.add_variation(move)

                    # In thông tin Nước đi và PGN
                    print(f"-> MOVE DETECTED: {color_name} {piece_name} moved {chess.square_name(move.from_square)} -> {chess.square_name(move.to_square)}")
                    
                    # Cập nhật lịch sử ra terminal
                    exporter = chess.pgn.StringExporter(headers=False, variations=True, comments=True)
                    pgn_string = game.accept(exporter)
                    print(f"[PGN Current]: {pgn_string}\n")

                    # Kiểm tra game over
                    if board.is_game_over():
                        print("-> GAME OVER:", board.result())

                    # Cập nhật ảnh tham chiếu mới
                    prev_warped_img = current_warped_img.copy()
                    print("-> Board updated. Ready for next move.")
                else:
                    print(f"-> Cannot detect move: {msg}")
                    print(f"   Changed squares indices: {changes}")

        # Phím 's': Lưu file PGN
        elif key == ord('s'):
            try:
                with open("game_recorded.pgn", "w", encoding="utf-8") as f:
                    exporter = chess.pgn.FileExporter(f)
                    game.accept(exporter)
                print("-> SUCCESS: Đã lưu ván cờ vào file 'game_recorded.pgn'!")
            except Exception as e:
                print(f"-> ERROR: Không thể lưu file PGN: {e}")

        # --- 3. Hiển thị ---
        last_move = board.peek() if len(board.move_stack) > 0 else None
        virtual_img = visualizer.draw_board(board, last_move)

        cv2.imshow("Camera", frame_resized)
        cv2.imshow("Virtual Board", virtual_img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()