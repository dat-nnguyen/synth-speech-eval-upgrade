# PDSM thích nghi: triển khai và cách kiểm chứng

Tài liệu yêu cầu gốc: `docs/HUONG_DAN_CAI_TIEN.md` (được giữ nguyên).
Các đường dẫn `../thuc-nghiem/...` trong tài liệu gốc thuộc một bố trí thư mục
khác. Cấu hình được triển khai ở đây dùng dữ liệu hiện có trong `inputs/` và
tách output `outputs/adaptive/` với `outputs/baseline/`.

## 1. Quy tắc đã triển khai

Sau khi gộp saliency của mỗi khoảng âm vị thành điểm `s_i`, trong từng nhóm
`(dataset, file, kênh, đoạn, chiều chất lượng)`:

```text
cutoff = mean(scores) + adaptive_lambda * std(scores, ddof=0)
selected = score > cutoff
```

- So sánh nghiêm ngặt `>`; không lấy các điểm bằng cutoff.
- Không ép chọn ít nhất một âm vị. Nhóm chỉ một âm vị hoặc các điểm bằng nhau
  không chọn âm vị nào khi lambda không âm.
- Xếp theo điểm giảm dần; điểm bằng nhau giữ thứ tự thời gian.
- Lambda phải hữu hạn, không âm. Giá trị ví dụ mặc định là 1.0.
- Default của project vẫn là `k_method: percent`, `k: 0.1`.

Hai sửa lỗi được dùng chung cho baseline và adaptive:

1. `end_frame` là khung cuối dạng bao gồm. Pooling và mask nay dùng
   `start:end+1`, nên không mất khung cuối hoặc làm rỗng âm vị một khung.
2. Top phần trăm dùng `floor(N*k)`. Khi số lượng bằng 0, chọn 0 âm vị;
   không còn lát cắt `[-0:]` vô tình lấy tất cả.

Vì vậy baseline là **top 10% đã sửa lỗi**, không phải nguyên trạng mọi lỗi
của repo gốc. Không quy lợi ích của hai sửa lỗi chung cho quy tắc adaptive.
Không huấn luyện lại SQ-AST/WhisperX và không thay trọng số.

## 2. Các file chính

| File | Chức năng |
|---|---|
| `src/process/pdsm.py` | Tính điểm tất cả âm vị, chọn percent/count/adaptive; giữ API hai giá trị cũ |
| `src/process/phoneme_export.py` | Schema CSV và đổi mốc frame sang giây trên file gốc |
| `src/process/proc.py` | Nối adaptive vào pipeline, ghi điểm, đếm từng dòng, lưu score sớm |
| `src/compare_pdsm.py` | So sánh sáu quy tắc từ cùng một CSV; chỉ cần NumPy và thư viện chuẩn |
| `configs/adaptive.yaml` | MOS, lambda=1.0, threshold chất lượng=3 |
| `configs/baseline.yaml` | Cùng dữ liệu/mô hình/pooling, chọn top 10% |
| `tests/test_adaptive_pdsm.py` | Kiểm thử thuật toán, chỉ số và rò rỉ dữ liệu |
| `tests/test_pipeline_export.py` | Kiểm thử pipeline với đầu ra mô hình giả lập |
| `scripts/cache_alignment_models.py` | Chuẩn bị cache ASR/alignment dùng bởi wrapper hiện có |

## 3. Chạy kiểm thử

Trên máy này, dùng image Docker đã chuẩn bị:

```bash
docker run --rm --init \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -v "$PWD:/work" -w /work synth-speech-eval-cpu:local \
  python -m unittest discover -s tests -v
```

Trong môi trường Python có NumPy, có thể dùng:

```bash
python -m unittest discover -s tests -v
```

Các kiểm thử tích hợp được bỏ qua nếu thiếu bộ thư viện pipeline. Trong
image đầy đủ, bộ kiểm thử có 41 ca gồm 35 ca nhẹ và 6 ca tích hợp. Dữ liệu mô hình
trong kiểm thử tích hợp được mock; chúng không chứng minh hiệu quả trên audio.
Log kiểm thử được lưu tại `outputs/improvement/tests_final.log`.

## 4. Chạy audio để lấy điểm âm vị

```bash
# Dung lai mos.pth da tai trong src/models/weights/.
# Chuan bi cache neu may chua co ASR/alignment:
docker run --rm --init -e HF_HUB_DISABLE_XET=1 \
  -v "$PWD:/work" -v synth-speech-eval-cache:/root/.cache \
  -w /work synth-speech-eval-cpu:local \
  python scripts/cache_alignment_models.py

bash scripts/run_cpu.sh adaptive
```

Input hiện tại: `inputs/input_sample.wav`, stereo, 44.100 Hz, 15,708 giây.
Hai cấu hình chỉ dùng `sq_ast_dims: [mos]`, cùng GradCAM, pooling L2, batch 1
và `score_threshold: 3`. Ngưỡng chất lượng này quyết định đoạn nào được phân
tích; nó khác ngưỡng adaptive dùng chọn âm vị. Nếu cần phân tích tất cả,
đổi threshold thành 5 trong cùng thiết lập của mọi phương pháp.

Output của một lượt adaptive:

```text
outputs/adaptive/inputs/<timestamp>/
  output_sq_ast.tsv
  config_used.yaml
  logs.log
  phoneme_scores.csv       # khi ASR/alignment/PDSM chay duoc
  output_thresholded.tsv   # khi toan bo nhanh phan tich hoan tat
  individual_plots/
  system_plots/
```

Lệnh `bash scripts/run_cpu.sh baseline` cũng được cung cấp, nhưng không cần
chạy mô hình thêm sáu lần để so sánh. Dùng cùng `phoneme_scores.csv` làm đầu
vào cho cả sáu quy tắc là cách so sánh trực tiếp.

Score SQ-AST được ghi trước bước alignment. Nếu bước sâu lỗi, kiểm tra log
và exit code; không coi bảng score có sẵn là bằng chứng nhánh âm vị đã hoàn tất.
CLI hiện trả exit code khác 0 khi `run_eval` báo lỗi.

Image hiện tại được sửa TorchCodec về 0.7.0 cho torch 2.8; `pip check` và
load VAD có sẵn đã qua. Image cũ được giữ ở tag
`synth-speech-eval-cpu:pre-adaptive-20261107`.

Nếu mạng gián đoạn, Hugging Face có thể trả lại thư mục snapshot đã có
metadata nhưng chưa đủ trọng số. Script cache kiểm tra file thực và dung
lượng so với metadata trước khi ghi thành công; không chỉ tin tên thư mục
hoặc dòng tiến độ 100%. File `.incomplete` không phải trọng số sẵn dùng.
Sau khi đã xác minh đủ cache SQ-AST, ASR và alignment trên máy này, có thể
chạy `HF_HUB_OFFLINE=1 bash scripts/run_cpu.sh adaptive` để tránh truy cập
metadata mạng và tải bản chuyển đổi safetensors tự động không cần thiết.
Nếu cache thiếu file, chế độ offline sẽ báo lỗi; chạy lại lệnh chuẩn để tải.

## 5. Nội dung CSV

Mỗi dòng là **một khoảng âm vị, một đoạn/kênh, một chiều chất lượng**,
kể cả khi không được chọn. Các cột chính:

| Cột | Diễn giải |
|---|---|
| `dataset`, `file_path`, `wav_channel`, `wav_start`, `wav_end`, `dimension` | Khóa nhóm |
| `phoneme_index`, `phoneme_count` | Thứ tự theo thời gian và tổng số âm vị của nhóm |
| `phoneme_id`, `phoneme` | Nhãn từ bộ căn chỉnh; không tự lọc im lặng/ký hiệu đặc biệt |
| `start_frame`, `end_frame` | Frame địa phương, khung cuối dạng bao gồm |
| `start_seconds`, `end_seconds` | Mốc trên file gốc, khoảng `[start,end)` |
| `score` | Điểm saliency sau pooling, không phải xác suất có lỗi |
| `selected`, `selection_rank`, `cutoff` | Lựa chọn của cấu hình đang chạy; cutoff chỉ có cho adaptive |
| `saliency_method`, `pdsm_preprocess`, `pdsm_pool` | Thông tin để đối chiếu phép tính |
| `label`, `split` | Để trống khi xuất; chỉ điền nhãn độc lập và dev/test khi có |

Công thức thời gian:

```text
base = wav_start / file_fs
start_seconds = base + start_frame * 0.01
end_seconds = min(wav_end / file_fs, base + (end_frame + 1) * 0.01)
```

Đây là quy đổi theo lưới frame của backend hiện tại, không phải xác nhận
độ chính xác alignment bằng ground truth. Bản cải tiến giữ nguyên wrapper
WhisperX; các vấn đề căn chỉnh, ánh xạ silence và hard-code ngôn ngữ đã biết
không được coi là đã sửa bởi thay đổi ngưỡng.

## 6. So sánh sáu quy tắc, không cần chạy lại mô hình

Sau khi có CSV thực tế, thay `<timestamp>` trong lệnh:

```bash
docker run --rm --init \
  -v "$PWD:/work" -w /work synth-speech-eval-cpu:local \
  python -m src.compare_pdsm \
  "outputs/adaptive/inputs/<timestamp>/phoneme_scores.csv" \
  --output-dir outputs/improvement/comparison_audio
```

Trong Python chỉ có NumPy:

```bash
python -m src.compare_pdsm "DUONG_DAN/phoneme_scores.csv" \
  --output-dir outputs/improvement/comparison_audio
```

Sáu quy tắc là top 5%, 10%, 20% và adaptive lambda 0.5, 1.0, 1.5.

- `comparison.csv`: số file/nhóm/âm vị, số chọn; nếu có nhãn thì thêm confusion
  matrix, Precision, Recall, F1, FPR; tách theo split và chiều.
- `selections.csv`: giữ dữ liệu gốc, thêm lựa chọn/rank/cutoff của từng quy tắc.
- `selected_on_dev.json`: thông tin nguồn, SHA256 CSV và lựa chọn dựa trên dev.
  Nếu không có nhãn dev hoặc dev không có lỗi, không chọn phương pháp.

Dòng `ALL` cộng đếm micro trên các chiều, không phải trung bình F1.
Không cộng dòng ALL với các dòng chiều để tránh đếm hai lần.
Nếu thiếu nhãn, TP/FP/FN/TN và các chỉ số chất lượng để trống, không tự gán 0.
Khi đã có nhãn, Precision/Recall/F1 có mẫu số 0 được ghi 0; FPR khi không có
âm tính để trống.

Để thử CLI ngay bằng dữ liệu giả lập có đánh dấu rõ:

```bash
docker run --rm --init -v "$PWD:/work" -w /work \
  synth-speech-eval-cpu:local python -m src.compare_pdsm \
  examples/phoneme_scores_synthetic.csv \
  --output-dir outputs/improvement/synthetic_comparison
```

Đây là 20 điểm nhân tạo 0-19, không có WAV hoặc nhãn tương ứng. Nó chỉ
kiểm tra thao tác CSV/quy tắc; kết quả không phải thực nghiệm chất lượng audio.

## 7. Gán nhãn và chọn tham số sau này

1. Sao chép CSV thành `phoneme_scores_labelled.csv`, giữ đủ điểm và các dòng.
2. Nghe và đánh dấu lỗi độc lập, không xem saliency/selected khi gán nhãn.
   Có thể chốt trước quy tắc giao nhau ít nhất 50% thời lượng để gán nhãn âm vị.
3. Điền `label=0/1` và `split=dev/test` cho mọi âm vị của một nhóm được đánh giá.
   Nhóm chưa gán nhãn có thể giữ trống cả hai cột.
4. Chia theo file gốc. Mọi đoạn/kênh/chiều của một file phải cùng tập.
   Một WAV hiện tại không đủ để tạo cả dev và test độc lập theo file.
5. Chạy công cụ so sánh; chọn theo micro F1 trên dev, hòa thì Precision,
   Recall và cuối cùng thứ tự sáu quy tắc cố định. Không dùng test để chọn.
6. Đọc test của lựa chọn đã khóa, so với top 10% và baseline percent tốt nhất
   trên dev. Nếu chưa có test, không kết luận khả năng tổng quát hóa.

Công cụ từ chối nhóm gán nhãn một phần, chỉ số âm vị trùng/thiếu, score không
hữu hạn và cùng định danh file xuất hiện ở cả dev và test. Nó không tự nhận
biết hai file đổi tên là bản sao cùng một câu; cần quản lý nguồn dữ liệu.

Chỉ thay ngưỡng không đổi thứ hạng điểm trong cùng CSV. Chưa có cơ sở khẳng
định tăng AUC, F1 hoặc độ chính xác trên giọng nói thật khi chưa có nhãn.

## 8. Kết quả audio thật quan sát ngày 07/11/2026

Lượt `outputs/adaptive/inputs/20261107_1510/` đã ghi đủ TSV, CSV và biểu đồ,
với dòng `Completed analysis in 0:13:30.031566`. Thời gian này gồm cả tải
alignment qua mạng. Sau khi ghi xong output, tiến trình còn chờ tải bản
safetensors chuyển đổi ở luồng nền; đã dừng tiến trình đó (exit 143) và chạy
lại toàn bộ bằng cache offline để kiểm tra CLI thoát bình thường. Trạng thái
lượt offline được ghi trong `outputs/improvement/implementation_audit.json`.

Lượt xác minh offline **`outputs/adaptive/inputs/20261107_1526/` hoàn tất và
thoát với exit code 0**, thời gian xử lý 6 phút 15.040 giây (22:26:15 đến
22:32:30 giờ Việt Nam). CSV 139 dòng có SHA256 trùng hoàn toàn lượt online.
Bảng `comparison_audio` cuối cùng dùng CSV của lượt offline này.

CSV có **139 khoảng PPG, 4 nhóm đoạn/kênh, 1 file gốc**, chưa có nhãn:

| Quy tắc | Số khoảng được chọn |
|---|---:|
| Top 5% | 4 |
| Top 10% | 12 |
| Top 20% | 27 |
| Adaptive lambda 0.5 | 25 |
| Adaptive lambda 1.0 | 17 |
| Adaptive lambda 1.5 | 15 |

Top phần trăm làm tròn xuống riêng từng nhóm, nên tổng số chọn không nhất
thiết đúng tỉ lệ phần trăm trên toàn bộ 139 dòng. Bảng so sánh nằm tại
`outputs/improvement/comparison_audio/`; bảng lựa chọn có 834 dòng = 139 × 6.

**Cả 17 khoảng adaptive lambda=1 chọn đều mang ký hiệu đặc biệt `<s>` của
backend. Không được diễn giải chúng thành 17 âm vị phát âm sai.** Đây là
giới hạn thực tế của alignment/PPG và có thể chịu ảnh hưởng pooling L2 trên
khoảng dài. Phạm vi cải tiến giữ nguyên wrapper; cần kiểm tra căn chỉnh và
quản lý silence riêng trước khi đánh giá chất lượng định vị.

Số chọn lambda=1 theo `(kênh, đoạn)` là `(0,1): 2/15`, `(0,2): 5/52`,
`(1,1): 5/35`, `(1,2): 5/37`. Cột `pdsm_sq_mos_num` khớp CSV từng nhóm.
MOS khớp hoàn toàn bốn dòng của lượt score cũ, max absolute delta bằng 0.
WhisperX tự nhận diện tiếng Anh với xác suất 0.25; đây không phải xác nhận
ngôn ngữ hoặc độ chính xác bản chép lời bằng người nghe.

Các cột Precision/Recall/F1/FPR để trống và JSON là `not_selected` do chưa
có nhãn dev. Hai trọng số ASR/alignment đã được kiểm tra file thực, dung
lượng và SHA256, không chỉ metadata snapshot. Các kết quả trên chỉ xác nhận
hành vi và khả năng chạy; chưa xác nhận cải thiện định vị lỗi.
