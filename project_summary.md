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
- **Chụp ảnh tham chiếu:** Nhấn một phím để lưu trạng thái bàn cờ hiện tại trước khi đi quản (`prev_warped_img`).
- **So sánh ảnh:** Sau khi đi quân thực tế, lấy ảnh bàn cờ mới, tính độ chênh lệch tuyệt đối (**absdiff**) giữa 2 ảnh sau khi chuyển sang ảnh xám và làm mờ (Gaussian Blur).
- **Phân tích từng ô:** Chia bức ảnh thành lưới 8x8, đếm số lượng pixel thay đổi (sau khi thresholding) ở mỗi ô. Hệ thống tìm ra các ô có sự thay đổi hình ảnh đáng kể nhất.
- **Suy luận logic:** Dùng thư viện `python-chess` sinh ra tất cả các nước đi hợp lệ (legal moves) ở trạng thái hiện tại. So khớp xem nước đi nào có điểm bắt đầu (from_square) và điểm kết thúc (to_square) trùng khớp với các ô có hình ảnh thay đổi.
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

- **Phím 'i':** Khởi tạo (Init) - Lưu lại hình ảnh trạng thái bàn cờ ban đầu.
- **Đi quân:** Thực hiện nước đi trên bàn cờ thật, sau đó rút tay ra khỏi khung hình.
- **Phím `SPACE`:** Xác nhận hệ thống phân tích và cập nhật nước đi vừa diễn ra.
- **Phím 'q':** Thoát chương trình.
