# Project Summary: CV Chess (Real-Time Chess Move Detection)

## 📌 Giới thiệu chung
**CV Chess** là hệ thống nhận diện nước đi cờ vua theo thời gian thực từ camera, sử dụng kỹ thuật xử lý ảnh (Computer Vision) kết hợp với thư viện `python-chess` để quản lý và kiểm tra tính hợp lệ của nước đi.

## 🚀 Các tính năng chính

Hệ thống cung cấp 2 chế độ nhận diện và hoạt động qua hai file chính:

1. **`test_chess.py` (Chế độ tự động):** Tự động phát hiện góc của bàn cờ sử dụng thuật toán Contour Analysis (tìm viền ngoài lớn nhất).
2. **`test_chess_manualConner.py` (Chế độ thủ công):** Cho phép người dùng chọn thủ công 4 góc của bàn cờ để warp ảnh, giúp tăng độ ổn định và chính xác trong các điều kiện ánh sáng hoặc góc máy khó.

## 🧠 Kiến trúc hệ thống & Luồng xử lý (Pipeline)

Hệ thống hoạt động theo 2 giai đoạn chính:

### 1. Phát hiện bàn cờ (Board Detection & Perspective Warp)

- Sử dụng **Canny Edge Detection** kết hợp morphology (gộp nét) để tìm các cạnh trong ảnh camera.
- Áp dụng **Contour Analysis (`cv2.findContours`)** để tìm các đường viền bao khép kín.
- Lọc contour lớn nhất và dùng `cv2.approxPolyDP` để xấp xỉ lấy ra chính xác 4 góc tọa độ của bàn cờ.
- Áp dụng **Perspective Transform** (`cv2.getPerspectiveTransform` và `cv2.warpPerspective`) để bẻ cong (warp) góc nhìn của camera về dạng 2D top-down (nhìn từ trên xuống) với kích thước cố định (chia làm lưới 8x8).

### 2. Phát hiện & Suy luận nước đi (Move Detection & Inference)

- **Chụp ảnh tham chiếu & Cân chỉnh (Calibrate):** Khi nhấn 'i' (Init), hệ thống lưu lại ảnh gốc và dò lại chính xác các đường thẳng ngang dọc (`HoughLines`) của mặt phẳng bàn cờ để tạo ra bộ lưới `h_grid`, `v_grid`. Bộ lưới này giúp khắc phục hoàn toàn sự méo mó quang học (perspective distortion) của camera.
- **So sánh ảnh:** Sau khi đi quân thực tế, lấy ảnh bàn cờ mới, tính độ chênh lệch tuyệt đối (**absdiff**) giữa 2 ảnh sau khi chuyển sang ảnh xám.
- **Phân tích từng ô:** Chia bức ảnh thành lưới không đều dựa trên `h_grid`, `v_grid` tạo vòng trước, đếm số lượng pixel thay đổi (sau khi thresholding + OTSU) ở mỗi ô. Hệ thống tìm ra các ô có sự thay đổi hình ảnh đáng kể nhất.
- **Suy luận logic:** Dùng thư viện `python-chess` sinh ra tất cả các nước đi hợp lệ (legal moves) ở trạng thái hiện tại. So khớp xem nước đi nào có điểm bắt đầu (from_square) và điểm kết thúc (to_square) trùng khớp với các ô.
- **Lưu diễn biến chuẩn PGN:** Nước đi hợp lệ sẽ được ghi nhận vào Node (thử viện `chess.pgn`), cho phép theo dõi log và xuất file ván đấu toàn thư (`.pgn`).
- **Cập nhật giao diện:** Thể hiện nước đi lên bàn cờ ảo UI (Virtual Board).

## 🛠️ Công nghệ sử dụng

- **Ngôn ngữ:** Python
- **Xử lý ảnh:** OpenCV (`cv2`)
- **Toán học & Mảng:** Numpy
- **Logic kiểm tra cờ vua:** `python-chess`

## ⚙️ Hướng dẫn cài đặt & Chạy dự án

**1. Cài đặt thư viện:**

```bash
pip install opencv-python numpy python-chess
```

**2. Khởi chạy hệ thống:**

```bash
python test_chess.py
# Hoặc chế độ căn góc thủ công:
python test_chess_manualConner.py
```

**3. Thao tác điều khiển (trong quá trình chạy):**

- **Phím 'i':** Khởi tạo (Init) - Lưu lại hình ảnh trạng thái bàn cờ ban đầu và tự động *Calibrate* lưới ô cờ bên trong để triết tiêu sự phân sai phối cảnh.
- **Đi quân:** Thực hiện nước đi trên bàn cờ thật, sau đó rút tay ra khỏi khung hình.
- **Phím `SPACE`:** Xác nhận hệ thống phân tích và cập nhật nước đi vừa diễn ra.
- **Phím 's':** Lưu lại lịch sử ván cờ vào file dạng văn bản chuẩn `game_recorded.pgn` trong thư mục project.
- **Phím 'q':** Thoát chương trình.

## 📝 Lịch sử thay đổi & Tối ưu hóa

Gần đây hệ thống đã được tích hợp một số nâng cấp quan trọng để hoạt động mượt mà và chính xác hơn ở môi trường thực tế:

**1. Thay thế `HoughLines` bằng `Contour Analysis`:**
- Ở phiên bản cũ, việc dùng HoughLines bị giới hạn do tìm thấy quá nhiều đường thẳng gây rối tọa độ giao điểm (nhất là ảnh bị nhiễu).
- Hệ thống hiện dùng `cv2.findContours` kếp hợp `cv2.approxPolyDP` để bắt bao quát đường viền ngoài hình chữ nhật lớn nhất (bàn cờ) để tự động Warp ảnh. 

**2. Cân chỉnh lưới quang học (Calibration):**
- Hiện tượng quang học khiến các ô vuông bàn cờ bị méo khi đưa lên camera ở góc chéo (ô ở xa sẽ nhỏ hơn ô ở gần).
- Giải pháp: Khởi tạo ảnh với phím 'i', tái sử dụng `HoughLines` quét trên ảnh đã phẳng hóa (Warp) và gom nhóm các khoảng cách đường kẻ thực tế (Hàm `cluster_lines`). Tạo ra lưới viền cong ảo `h_grid` & `v_grid` bám sát mọi ô cờ ở bất kì góc độ nào.

**3. Tiền xử lý ánh sáng thông minh (CLAHE & OTSU):**
- Khắc phục lỗi sai viền do chói sáng cục bộ hoặc bóng đổ từ tay người chơi. Giải pháp sử dụng `cv2.createCLAHE` ép cân bằng sáng theo từng mảng nhỏ (tăng độ tương phản vạch kẻ) và `cv2.THRESH_OTSU` tìm khoảng phân tách màu tự động thông minh.

**4. Xuất mảng Game PGN (Portable Game Notation):**
- Nâng cấp `python-chess` từ thư viện bắt lỗi luật chơi thông thường thành máy lưu trữ "Node Tree".
- In thông báo realtime trên Terminal theo định dạng SAN (Standard Algebraic Notation, vd: `1. e4 e5`) và trích xuất ván cờ hoàn chỉnh dạng `game_recorded.pgn` bằng phím tắt `S`. File này tương thích mọi bộ đếm PGN PGN chuẩn toàn cầu (Chess.com, Lichess).
