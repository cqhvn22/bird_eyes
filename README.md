# Bird Eyes — Perch Stereo

Ứng dụng thị giác stereo để phân đoạn cành/điểm đậu (*perch*) và ước lượng khoảng cách, chiều dài, chiều rộng từ hai camera hoặc hai video trái–phải. Project dùng model YOLOv8 segmentation đã huấn luyện (`best-seg.pt`), xuất sang **NCNN FP16**, suy luận trên CPU và không cần PyTorch/Ultralytics ở runtime.

Hai chương trình dùng chung pipeline:

- `perch_detector.py`: nhận hình từ hai camera trên Raspberry Pi 5 qua Picamera2.
- `perch_video_seg.py`: xử lý cặp video stereo trên Windows hoặc Raspberry Pi.

## Tính năng

- Phân đoạn trên **ảnh trái**; ảnh phải dùng để hiển thị và tính stereo depth.
- Hai profile model: `detail` (960, mặc định) và `fast` (640).
- Gợi ý căn giữa và góc căn chỉnh theo ảnh 2D.
- Chỉ tính độ sâu SGBM/WLS khi nhấn **T**; chế độ LIVE không tính depth.
- Theo dõi đối tượng giữa các frame, lấy trung vị phép đo từ một đợt chụp.
- Lưu ảnh xem trước, mask, depth NumPy, dữ liệu JSON và video xem trước MP4.
- Có chế độ `--headless` để chạy không cần cửa sổ giao diện.

## Cấu trúc repository

```text
bird_eyes/
├── perch_detector.py          # Chạy camera stereo
├── perch_video_seg.py         # Chạy hai video
├── pyproject.toml             # Package, dependency và console commands
├── src/perch/                 # Pipeline dùng chung
│   ├── backend.py            # NCNN segmentation và kiểm tra model
│   ├── depth.py              # Rectification, SGBM/WLS
│   ├── measurement.py        # Đo khoảng cách và kích thước
│   ├── workflow.py           # LIVE → CAPTURING → RESULT
│   ├── sources/              # Nguồn camera/video
│   └── ...                   # CLI, tracking, giao diện và lưu kết quả
├── assets/
│   ├── calibration/          # stereo_calibration.npz
│   └── models/               # Hai bản NCNN FP16: 640 và 960
├── data/videos/              # Tự thêm video cục bộ; không đưa lên Git
├── outputs/                  # Kết quả chạy; không đưa lên Git
├── scripts/setup_pi5.sh       # Cài môi trường Raspberry Pi 5
├── requirements/pi5.txt       # Dependency NCNN cho Pi
├── tests/                    # Kiểm thử hồi quy
├── docs/architecture.md       # Chi tiết kiến trúc
└── LICENSE                   # GNU GPL v3
```

Model NCNN và calibration có sẵn trong repository. **Video thử nghiệm không được đưa lên GitHub**; sau khi clone, bạn cần chuẩn bị cặp video của mình để chạy chế độ video. Checkpoint `.pt` và công cụ train/export không thuộc bản runtime này.

## Cài đặt và chạy trên Windows

Cần Git, Python **64-bit từ 3.10 trở lên** và các dependency trong `pyproject.toml`. Mở PowerShell:

```powershell
git clone https://github.com/cqhvn22/bird_eyes.git
cd bird_eyes
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
New-Item -ItemType Directory -Force data/videos
```

Đặt hai video tương ứng vào `data/videos/l1.mp4` và `data/videos/r1.mp4`, rồi chạy:

```powershell
.\.venv\Scripts\python.exe perch_video_seg.py --left data/videos/l1.mp4 --right data/videos/r1.mp4
```

Có thể dùng đường dẫn tới video ở nơi khác. Hai video cần đúng thứ tự mắt trái–phải và được đồng bộ; dùng `--left-start` / `--right-start` để chọn frame bắt đầu cho từng video.

Nếu không truyền `--left` / `--right`, chương trình tìm hai file sau trong `data/videos/`:

```text
left_20261026_172822.mp4
right_20261026_172822.mp4
```

Thêm `--profile fast` để dùng model 640. Sau khi cài package, `.\.venv\Scripts\perch-video-seg.exe` là lệnh tương đương với launcher Python.

## Cài đặt và chạy trên Raspberry Pi 5

Cần Raspberry Pi OS **64-bit**, Python từ 3.10 và hai camera được Picamera2 nhận diện. Clone repository rồi chạy:

```bash
git clone https://github.com/cqhvn22/bird_eyes.git
cd bird_eyes
bash scripts/setup_pi5.sh
.venv-pi5/bin/python perch_detector.py
```

Script cài OpenCV, NumPy, PyYAML và Picamera2 qua `apt`, tạo môi trường `.venv-pi5` dùng được package hệ thống, rồi cài NCNN qua `pip`. Script cũng kiểm tra OpenCV có `ximgproc.createRightMatcher` cần cho stereo depth.

Mặc định: **CAM0 trái, CAM1 phải, 1920 × 1080, 30 FPS**. Chọn profile 640 bằng:

```bash
.venv-pi5/bin/python perch_detector.py --profile fast
```

Chạy video trên Pi:

```bash
.venv-pi5/bin/python perch_video_seg.py --left data/videos/l1.mp4 --right data/videos/r1.mp4
```

Console commands tương ứng là `.venv-pi5/bin/perch-detector` và `.venv-pi5/bin/perch-video-seg`. Tạo môi trường riêng trên Pi; không sao chép virtual environment từ Windows.

## Quy trình đo và điều khiển

1. **LIVE**: xem ảnh stereo và segmentation, căn chỉnh theo gợi ý trên màn hình.
2. Nhấn **T**: chuyển sang **CAPTURING**, xử lý mặc định **3 cặp ảnh mới** sau thời điểm kích hoạt.
3. **RESULT**: giữ kết quả và ảnh xem trước. Cần đủ **3 mẫu hợp lệ** để đánh giá; kết quả dùng trung vị các phép đo của cùng một đối tượng.
4. Nhấn **R** để xóa phép đo và quay về LIVE, hoặc **T** để đo lại.

| Phím | Chức năng |
|---|---|
| `T` | Bắt đầu một đợt đo depth |
| `R` | Xóa kết quả và quay về LIVE |
| `Space` | Tạm dừng / tiếp tục; ở RESULT thì quay về LIVE |
| `G` | Bật/tắt gợi ý căn chỉnh |
| `D` | Hiện/ẩn depth; không kích hoạt tính depth |
| `S` | Lưu snapshot của frame hiện tại |
| `B` | Tua lại cả hai video về frame bắt đầu đã cấu hình; chỉ dùng cho video |
| `Q` / `Esc` | Thoát |

Ngưỡng đánh giá hiện tại: khoảng cách **0,20–5,00 m**, chiều dài **0,10–2,00 m**, chiều rộng **0,02–0,20 m**. Thiếu mẫu depth hợp lệ hoặc mất đối tượng ở cuối đợt đo sẽ không cho kết luận đạt/không đạt.

## Tham số thường dùng

| Tham số | Mặc định / ý nghĩa |
|---|---|
| `--profile detail` / `--profile fast` | Model 960 / 640; mặc định `detail` |
| `--model PATH` | Thư mục model NCNN tùy chọn, ghi đè profile |
| `--calib PATH` | File calibration; mặc định `assets/calibration/stereo_calibration.npz` |
| `--unit mm` | Đơn vị calibration: `mm`, `cm` hoặc `m` |
| `--conf 0.35` | Ngưỡng confidence segmentation |
| `--downscale 0.5` | Tỉ lệ ảnh dùng cho depth |
| `--burst-frames 3 --min-samples 3` | Số cặp mỗi đợt và số mẫu hợp lệ tối thiểu |
| `--left-camera 0 --right-camera 1` | Chỉ số camera trái/phải |
| `--left PATH --right PATH` | Hai video đầu vào |
| `--left-start 0 --right-start 0` | Frame bắt đầu của từng video |
| `--loop` | Lặp cả hai video khi hết file |
| `--headless` | Không mở cửa sổ giao diện |
| `--trigger-frame N` | Tự kích hoạt đo sau khi xử lý frame có chỉ số ít nhất N; chỉ số bắt đầu từ 0 |
| `--max-frames N` | Giới hạn tổng số cặp xử lý |
| `--save-last` | Lưu snapshot cuối trước khi thoát |
| `--snapshot-dir PATH` | Ghi đè thư mục lưu snapshot |
| `--output PATH.mp4` | Ghi video xem trước bố cục 2 × 2 |

Các lệnh dùng `python` bên dưới giả định bạn đã kích hoạt virtual environment; nếu chưa, thay bằng đường dẫn Python của môi trường đã cài ở trên.

Xem toàn bộ tham số:

```bash
python perch_detector.py --help
python perch_video_seg.py --help
```

Ví dụ chạy headless, kích hoạt đo sau frame đầu tiên, xử lý tối đa 4 cặp và lưu kết quả:

```bash
python perch_video_seg.py --left data/videos/l1.mp4 --right data/videos/r1.mp4 --headless --trigger-frame 0 --max-frames 4 --save-last
```

Frame kích hoạt không nằm trong đợt đo; cần thêm 3 cặp mới cho cấu hình mặc định. Khi chạy headless, chương trình thoát khi có RESULT hoặc đạt giới hạn frame. `--headless --loop` cần thêm `--max-frames`.

## Dữ liệu đầu ra

Nhấn `S` hoặc dùng `--save-last` để lưu vào `outputs/camera/` hay `outputs/video/` tùy chương trình:

- `*_seg.png`, `*_right.png`, `*_view.png`: ảnh segmentation, ảnh phải và bố cục xem trước.
- `*_mask_<track_id>.png`: mask theo từng đối tượng.
- `*_depth.png`, `*_depth_m.npy`: ảnh depth và ma trận độ sâu theo mét, khi có dữ liệu tương ứng.
- `*.json`: chỉ số cặp ảnh, trạng thái quy trình, confidence, khoảng cách, kích thước, số mẫu và lý do đánh giá.

Mỗi snapshot dùng tên có thời gian và chỉ số frame. Thư mục `outputs/` được Git bỏ qua để tránh đưa kết quả chạy lên repository.

## Calibration và giới hạn phép đo

Calibration phải khớp với cặp camera, thứ tự trái–phải, chế độ chụp và đơn vị thực tế. File có sẵn thuộc cấu hình của project; khi thay camera hoặc cấu hình quang học, cần cung cấp calibration phù hợp qua `--calib`.

Gợi ý căn chỉnh dựa trên ảnh 2D. Chiều dài và chiều rộng là ước lượng chiếu theo độ sâu đại diện, không phải phép tái dựng đầy đủ hình dạng 3D. Chất lượng kết quả phụ thuộc vào calibration, đồng bộ ảnh, segmentation và lượng pixel depth hợp lệ.

Các đường dẫn mặc định được tính từ thư mục project. Đường dẫn tương đối truyền qua CLI được tính từ thư mục làm việc hiện tại. Nên cài bằng `pip install -e .`; nếu cài wheel riêng, đặt biến môi trường `PERCH_HOME` tới thư mục chứa `assets/`, `data/` và `outputs/`, vì các tài nguyên này không được nhúng vào wheel.

## Kiểm thử

Sau khi cài package và chuẩn bị dữ liệu cục bộ:

```bash
python -m unittest discover -s tests -v
```

Lưu ý: test `test_defaults_resolve_to_standalone_resources` kiểm tra cả sự tồn tại của hai video mặc định nêu trên. Vì video không được đưa lên GitHub, cần bổ sung chúng để chạy đầy đủ bộ test hiện tại.

Kiểm tra pipeline với cặp video `l1.mp4` / `r1.mp4` có đủ frame:

```bash
python perch_video_seg.py --headless --left data/videos/l1.mp4 --right data/videos/r1.mp4 --left-start 188 --right-start 188 --trigger-frame 0 --max-frames 4
```

Xem [kiến trúc và trách nhiệm từng module](docs/architecture.md). [Báo cáo kiểm chứng refactor đã lưu](docs/refactor_validation.json) ghi nhận 49 test đạt và đối chiếu trên video; báo cáo này chưa bao gồm kiểm thử trên phần cứng Raspberry Pi thực tế.

## Xử lý lỗi thường gặp

| Hiện tượng | Cách kiểm tra |
|---|---|
| Không mở được video | Chuẩn bị file cục bộ và truyền đúng `--left` / `--right`; video không có trong bản clone |
| `Calibration missing` | Kiểm tra `assets/calibration/`, `--calib` hoặc `PERCH_HOME` |
| Thiếu `cv2.ximgproc` | Windows dùng `opencv-contrib-python` theo dependency; trên Pi chạy script setup và kiểm tra OpenCV hệ thống |
| Thiếu Picamera2 hoặc không nhận camera | Chạy chế độ camera trên Pi với môi trường `.venv-pi5`, kiểm tra kết nối và chỉ số hai camera |
| Không đủ mẫu depth | Kiểm tra thứ tự/đồng bộ ảnh, calibration, vùng đối tượng và số frame còn lại; căn chỉnh rồi đo lại |

## Giấy phép

Project đi kèm giấy phép [GNU General Public License v3.0](LICENSE).
