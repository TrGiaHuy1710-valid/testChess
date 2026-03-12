import cv2
import numpy as np
import chess
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

def detect_changes(prev_img, curr_img, grid_size=500):
    """
    So sánh 2 ảnh, trả về danh sách chỉ số ô (0-63) có sự thay đổi lớn nhất.
    """
    if prev_img is None or curr_img is None:
        return []

    # Chuyển xám và làm mờ
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # Tính hiệu ảnh
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    square_w = grid_size / 8
    changes = []

    # Duyệt qua 64 ô
    for r in range(8):  # 0-7 (Top-Down in Image)
        for c in range(8):  # 0-7 (Left-Right)
            x_start = int(c * square_w)
            y_start = int(r * square_w)
            x_end = int((c + 1) * square_w)
            y_end = int((r + 1) * square_w)

            # Đếm số pixel trắng (thay đổi) trong ô này
            roi = thresh[y_start:y_end, x_start:x_end]
            non_zero = cv2.countNonZero(roi)

            # Nếu lượng pixel thay đổi lớn hơn ngưỡng nhất định (noise filter)
            if non_zero > 100:
                # Mapping sang Chess Coordinate
                # Ảnh: r=0 là trên cùng. Chess: rank=7 là trên cùng.
                rank = 7 - r
                file = c
                sq_idx = chess.square(file, rank)
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


# --- MAIN ---
def main():
    cap = cv2.VideoCapture(0)

    # Set up logic variables

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

    print("=== HƯỚNG DẪN ===")
    print("1. Chỉnh camera sao cho toàn bộ bàn cờ nằm trong khung hình (hệ thống sẽ tự động bắt góc).")
    print("2. Nhấn phím 'i' (Init) để lưu trạng thái bàn cờ ban đầu.")
    print("3. Đi quân cờ thật.")
    print("4. Rút tay ra, nhấn phím 'SPACE' để cập nhật.")
    print("5. Nhấn 'q' để thoát.")
    print("=================")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- 1. Xử lý ảnh để tìm bàn cờ ---
        scale_percent = 60
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame_resized = cv2.resize(frame, (width, height))  # Resize để hiển thị

        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Dùng Canny và morphology để lấy đường viền liền mạch
        edges = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        board_found = False
        if contours:
            # Lọc các contour lớn, sắp xếp theo diện tích giảm dần
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 5000:  # Bỏ qua các hình quá nhỏ
                    break
                
                peri = cv2.arcLength(c, True)
                # Xấp xỉ đa giác
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # Nếu hình xấp xỉ có đúng 4 đỉnh, ta coi là bàn cờ
                if len(approx) == 4:
                    cv2.drawContours(frame_resized, [approx], -1, (0, 255, 0), 2)
                    rect = order_points(approx.reshape(4, 2))
                    M = cv2.getPerspectiveTransform(rect, dst_pts)
                    current_warped_img = cv2.warpPerspective(frame_resized, M, (WARPED_SIZE, WARPED_SIZE))
                    board_found = True
                    cv2.imshow("Warped View", current_warped_img)
                    break

        # --- 2. Xử lý phím bấm ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Phím 'i': Khởi tạo (Lưu ảnh ban đầu)
        elif key == ord('i'):
            if board_found:
                prev_warped_img = current_warped_img.copy()
                print("-> Đã lưu trạng thái bàn cờ ban đầu (Ready).")
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
                changes = detect_changes(prev_warped_img, current_warped_img)

                # 2. Suy luận nước đi
                move, msg = infer_move(board, changes)

                if move:
                    # Lấy tên quân cờ để in ra
                    piece = board.piece_at(move.from_square)
                    piece_name = piece.symbol() if piece else "Unknown"
                    color_name = "White" if piece and piece.color == chess.WHITE else "Black"

                    # Cập nhật bàn cờ ảo
                    board.push(move)

                    # In thông tin
                    print(
                        f"-> MOVE DETECTED: {color_name} {piece_name} moved {chess.square_name(move.from_square)} -> {chess.square_name(move.to_square)}")

                    # Kiểm tra game over
                    if board.is_game_over():
                        print("-> GAME OVER:", board.result())

                    # Cập nhật ảnh tham chiếu mới
                    prev_warped_img = current_warped_img.copy()
                    print("-> Board updated. Ready for next move.")
                else:
                    print(f"-> Cannot detect move: {msg}")
                    print(f"   Changed squares indices: {changes}")

        # --- 3. Hiển thị ---
        last_move = board.peek() if len(board.move_stack) > 0 else None
        virtual_img = visualizer.draw_board(board, last_move)

        cv2.imshow("Camera", frame_resized)
        cv2.imshow("Virtual Board", virtual_img)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
