import cv2
import numpy as np
import chess
import os

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục chứa ảnh quân cờ (bạn kiểm tra lại đường dẫn này trên máy bạn nhé)
ASSET_PATH = r"E:\Python_Project\Realsense\CVchess\G7_CVChess\Test_Chess\Chess_CV\assets\pieces"

PIECE_MAP = {
    'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk',
    'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk',
}

# Biến toàn cục để lưu 4 điểm click
calibration_points = []


class ChessVisualizer:
    def __init__(self, piece_dir, square_size=60):
        self.square = square_size
        self.pieces = {}
        if os.path.exists(piece_dir):
            for k, v in PIECE_MAP.items():
                path = os.path.join(piece_dir, f"{v}.png")
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.pieces[k] = cv2.resize(img, (square_size, square_size))

    def draw_board(self, board, last_move=None):
        img = np.zeros((8 * self.square, 8 * self.square, 3), dtype=np.uint8)
        for r in range(8):
            for c in range(8):
                color = (240, 217, 181) if (r + c) % 2 == 0 else (181, 136, 99)
                if last_move:
                    if last_move.from_square == chess.square(c, 7 - r) or \
                            last_move.to_square == chess.square(c, 7 - r):
                        color = (100, 200, 100)
                cv2.rectangle(img, (c * self.square, r * self.square),
                              ((c + 1) * self.square, (r + 1) * self.square), color, -1)

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
                if piece.symbol() in self.pieces:
                    self._overlay(img, self.pieces[piece.symbol()], c * self.square, r * self.square)
                else:
                    cv2.putText(img, piece.symbol(), (c * self.square + 20, r * self.square + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return img

    def _overlay(self, bg, fg, x, y):
        if fg.shape[2] == 4:
            alpha = fg[:, :, 3] / 255.0
            for c in range(3):
                bg[y:y + self.square, x:x + self.square, c] = \
                    alpha * fg[:, :, c] + (1 - alpha) * bg[y:y + self.square, x:x + self.square, c]


# --- XỬ LÝ CLICK CHUỘT ---
def mouse_callback(event, x, y, flags, param):
    global calibration_points  # Sử dụng biến toàn cục
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append((x, y))
            print(f"-> Point {len(calibration_points)} captured: ({x}, {y})")


# --- XỬ LÝ ẢNH & LOGIC WARP ---
def get_square_roi(img, row, col, grid_size, margin=0.2):
    sq_h, sq_w = grid_size / 8, grid_size / 8
    y_start = int((row + margin) * sq_h)
    y_end = int((row + 1 - margin) * sq_h)
    x_start = int((col + margin) * sq_w)
    x_end = int((col + 1 - margin) * sq_w)
    return img[y_start:y_end, x_start:x_end]


def detect_move_robust(prev, curr, grid_size=400):
    if prev is None or curr is None: return []

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    changes = []
    CHANGE_THRESHOLD = 30
    MIN_PIXELS_CHANGED = 50

    for r in range(8):
        for c in range(8):
            roi_prev = get_square_roi(prev_gray, r, c, grid_size)
            roi_curr = get_square_roi(curr_gray, r, c, grid_size)

            diff = cv2.absdiff(roi_prev, roi_curr)
            _, thresh = cv2.threshold(diff, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)

            count = cv2.countNonZero(thresh)
            if count > MIN_PIXELS_CHANGED:
                # Mapping: r=0 (ảnh) -> rank=7 (chess)
                sq_idx = chess.square(c, 7 - r)
                changes.append((sq_idx, count))

    changes.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in changes]


def main():
    global calibration_points  # <--- SỬA LỖI Ở ĐÂY: Khai báo sử dụng biến toàn cụcq

    cap = cv2.VideoCapture("chess_move.mp4")
    # Tùy chỉnh độ phân giải camera nếu cần
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Camera Setup")
    cv2.setMouseCallback("Camera Setup", mouse_callback)

    visualizer = ChessVisualizer(ASSET_PATH, square_size=60)
    board = chess.Board()

    WARPED_SIZE = 400
    dst_pts = np.array([
        [0, 0],
        [WARPED_SIZE, 0],
        [WARPED_SIZE, WARPED_SIZE],
        [0, WARPED_SIZE]
    ], dtype="float32")

    M = None
    prev_warped = None

    print("=== BƯỚC 1: CALIBRATION ===")
    print("Click chuột lần lượt vào 4 góc vùng chơi theo thứ tự:")
    print("1. Trái-Trên (a8) -> 2. Phải-Trên (h8) -> 3. Phải-Dưới (h1) -> 4. Trái-Dưới (a1)")
    print("Nhấn 'r' để reset điểm nếu click sai.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()

        # --- Giai đoạn 1: Chọn góc ---
        if len(calibration_points) < 4:
            for i, pt in enumerate(calibration_points):
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, str(i + 1), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Click point {len(calibration_points) + 1}/4", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # --- Giai đoạn 2: Đã có 4 góc ---
        else:
            if M is None:
                src_pts = np.array(calibration_points, dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                print("-> Calibration Complete! Press 'i' to initialize board.")

            warped = cv2.warpPerspective(frame, M, (WARPED_SIZE, WARPED_SIZE))

            # Vẽ lưới kiểm tra
            debug_warped = warped.copy()
            step = WARPED_SIZE // 8
            for i in range(1, 8):
                cv2.line(debug_warped, (i * step, 0), (i * step, WARPED_SIZE), (255, 0, 0), 1)
                cv2.line(debug_warped, (0, i * step), (WARPED_SIZE, i * step), (255, 0, 0), 1)

            cv2.imshow("Warped Grid Check", debug_warped)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('i'):
                prev_warped = warped.copy()
                print("-> Initialized. Move a piece and press SPACE.")

            elif key == 32:  # SPACE
                if prev_warped is not None:
                    print("Scanning...")
                    changed_squares = detect_move_robust(prev_warped, warped, WARPED_SIZE)
                    candidates = changed_squares[:4]

                    if candidates:
                        print(f"Detected changes at: {[chess.square_name(s) for s in candidates]}")

                        found_move = None
                        for move in board.legal_moves:
                            if move.from_square in candidates and move.to_square in candidates:
                                found_move = move
                                break

                        if found_move:
                            board.push(found_move)
                            print(f"-> MOVE CONFIRMED: {board.peek()}")
                            prev_warped = warped.copy()
                        else:
                            print("-> No legal move matches the changes.")
                    else:
                        print("-> No changes detected.")

        # Reset calibration
        if cv2.waitKey(1) & 0xFF == ord('r'):
            calibration_points = []  # Reset list
            M = None
            print("-> Reset calibration points.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow("Camera Setup", display_frame)
        virtual_board = visualizer.draw_board(board, last_move=board.peek() if board.move_stack else None)
        cv2.imshow("Virtual Game", virtual_board)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()