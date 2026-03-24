import cv2
import numpy as np
import chess
from board_process_en_new import ChessBoardProcessor
from test_chess_thaihung2 import ChessVisualizer, detect_changes, infer_move, draw_grid, PIECE_MAP, ASSET_PATH

def cluster_lines(data, max_val):
    if not data: return np.linspace(0, max_val, 9).tolist()
    data.sort()
    res = []
    current_cluster = [data[0]]
    for i in range(1, len(data)):
        if abs(data[i] - data[i-1]) > 40:  # Khoảng cách tối thiểu giữa 2 đường kẻ phải > 40px
            res.append(sum(current_cluster)/len(current_cluster))
            current_cluster = [data[i]]
        else:
            current_cluster.append(data[i])
    res.append(sum(current_cluster)/len(current_cluster))
    
    # Nếu không tìm đủ 9 đường, fall back về chia đều
    if len(res) != 9:
        return np.linspace(0, max_val, 9).tolist()
    return res


def calibrate_grid(final_board, wrap_size=500):
    warp_gray = cv2.cvtColor(final_board, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(warp_gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 110)
    
    h_lines, v_lines = [], []
    if lines is not None:
        for l in lines:
            rho, theta = l[0]
            if np.pi/4 < theta < 3*np.pi/4: 
                h_lines.append(abs(rho))
            else: 
                v_lines.append(abs(rho))
    
    h_grid = cluster_lines(h_lines, wrap_size)
    v_grid = cluster_lines(v_lines, wrap_size)
    
    return h_grid, v_grid

def main():
    # Sử dụng camera
    cap = cv2.VideoCapture("chess_move.mp4")  # Thay bằng 0 nếu dùng webcam
    if not cap.isOpened():
        print("Không thể mở camera")
        return
    
    processor = ChessBoardProcessor()
    visualizer = ChessVisualizer(piece_dir=ASSET_PATH, square_size=60)
    
    board = chess.Board()
    prev_final_board = None
    h_grid = None
    v_grid = None
    board_initialized = False
    
    print("=== HƯỚNG DẪN ===")
    print("1. Định vị camera sao cho nhìn thấy toàn bộ bàn cờ")
    print("2. Nhấn 'i' để khởi tạo và lưu trạng thái bàn cờ ban đầu")
    print("3. Đi quân cờ")
    print("4. Nhấn SPACE để phát hiện nước đi")
    print("5. Nhấn 'q' để thoát")
    print("=================\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi khi đọc frame từ camera")
            break
        
        # Resize frame
        frame_resized = cv2.resize(frame, (800, 600))
        
        # Tự động phát hiện bàn cờ
        board_contour = processor.get_board_contour_auto(frame_resized)
        
        if board_contour is not None and not board_initialized:
            if processor.wrap_size is None:
                processor.wrap_size = processor.calculate_optimal_side(board_contour)
            
            if processor.inner_pts is None:
                pts_src = processor.order_points(board_contour)
                pts_dst = np.array([
                    [0, 0], [processor.wrap_size-1, 0],
                    [processor.wrap_size-1, processor.wrap_size-1], [0, processor.wrap_size-1]
                ], dtype="float32")
                
                M1 = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped = cv2.warpPerspective(frame_resized, M1, (processor.wrap_size, processor.wrap_size))
                processor.inner_pts = processor.select_inner_corners(warped)
        
        # Xử lý ảnh hiện tại
        if board_contour is not None:
            pts_src = processor.order_points(board_contour)
            pts_dst = np.array([
                [0, 0], [processor.wrap_size-1, 0],
                [processor.wrap_size-1, processor.wrap_size-1], [0, processor.wrap_size-1]
            ], dtype="float32")
            
            M1 = cv2.getPerspectiveTransform(pts_src, pts_dst)
            warped = cv2.warpPerspective(frame_resized, M1, (processor.wrap_size, processor.wrap_size))
            
            if processor.inner_pts is not None:
                M2 = cv2.getPerspectiveTransform(processor.inner_pts, pts_dst)
                curr_final_board = cv2.warpPerspective(warped, M2, (processor.wrap_size, processor.wrap_size))
                
                # Hiển thị bàn cờ đã xử lý
                cv2.imshow("Processed Board", curr_final_board)
        else:
            curr_final_board = None
        
        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        # Phím 'i': Khởi tạo bàn cờ
        elif key == ord('i'):
            if curr_final_board is not None:
                print("Khởi tạo bàn cờ...")
                prev_final_board = curr_final_board.copy()
                h_grid, v_grid = calibrate_grid(prev_final_board, processor.wrap_size)
                board_initialized = True
                print("✓ Bàn cờ đã được khởi tạo. Sẵn sàng phát hiện nước đi!")
            else:
                print("✗ Lỗi: Không tìm thấy bàn cờ để khởi tạo")
        
        # Phím SPACE: Phát hiện nước đi
        elif key == 32:
            if not board_initialized:
                print("✗ Lỗi: Chưa khởi tạo bàn cờ (nhấn 'i' trước)")
            elif prev_final_board is None or curr_final_board is None:
                print("✗ Lỗi: Không tìm thấy bàn cờ")
            else:
                print("Phát hiện nước đi...")
                
                # Phát hiện thay đổi
                changes = detect_changes(prev_final_board, curr_final_board, h_grid, v_grid)
                
                # Suy luận nước đi
                move, msg = infer_move(board, changes)
                
                if move:
                    piece = board.piece_at(move.from_square)
                    piece_name = piece.symbol() if piece else "Unknown"
                    color_name = "White" if piece and piece.color == chess.WHITE else "Black"
                    
                    board.push(move)
                    print(f"✓ NƯỚC ĐI: {color_name} {piece_name}: {chess.square_name(move.from_square)} → {chess.square_name(move.to_square)}")
                    
                    if board.is_game_over():
                        print(f"✓ KẾT THÚC TRẬN: {board.result()}")
                    
                    prev_final_board = curr_final_board.copy()
                else:
                    print(f"✗ Không phát hiện nước đi: {msg}")
                    print(f"  Ô thay đổi phát hiện: {changes}")
        
        # Hiển thị frame gốc
        cv2.imshow("Camera Feed", frame_resized)
        
        # Hiển thị bàn cờ ảo
        last_move = board.peek() if len(board.move_stack) > 0 else None
        virtual_img = visualizer.draw_board(board, last_move)
        cv2.imshow("Virtual Chess Board", virtual_img)
        
        # Hiển thị grid nếu đã khởi tạo
        if board_initialized and curr_final_board is not None:
            grid_view = draw_grid(curr_final_board.copy(), h_grid, v_grid, processor.wrap_size)
            cv2.imshow("Board with Grid", grid_view)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()