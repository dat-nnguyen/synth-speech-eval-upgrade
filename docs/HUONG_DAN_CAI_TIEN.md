Cải tiến XSQ-AST bằng ngưỡng chọn âm vị thích nghi

## Mục tiêu và phạm vi

Thay quy tắc chọn top 10% âm vị bằng quy tắc chọn điểm saliency lớn hơn
`trung bình + lambda × độ lệch chuẩn` trong từng câu, từng chiều chất lượng.
Không huấn luyện lại SQ-AST hoặc WhisperX. Hai cấu hình mẫu chỉ đánh giá MOS
(chất lượng tổng thể) để giảm công sức chạy và gán nhãn. Có thể mở rộng năm
chiều sau, nhưng phải giữ cùng cấu hình cho mọi phương pháp so sánh.

Đây là giả thuyết cần kiểm chứng, chưa có kết quả chứng minh tăng độ chính xác.
Saliency thể hiện tầm quan trọng với mô hình, không phải xác suất có lỗi.
Ngưỡng thích nghi có thể bỏ sót lỗi khi cả câu xuống chất lượng đồng đều hoặc
khi có rất ít âm vị. Không chọn âm vị nào không đồng nghĩa câu không có lỗi.

## Phần đã triển khai

- `src/process/pdsm.py`: thêm `k_method: adaptive`, hệ số `adaptive_lambda`,
  dùng độ lệch chuẩn tổng thể (`ddof=0`) và so sánh nghiêm ngặt `score > cutoff`.
  Điểm bằng nhau hoặc chỉ một âm vị thì không chọn âm vị nào. Không ép số
  lượng tối thiểu. Điểm bằng nhau được xếp theo thứ tự thời gian.
- `src/process/proc.py`: nối thuật toán vào luồng chạy, xuất mọi điểm âm vị
  ra `phoneme_scores.csv`, lưu số âm vị được chọn đúng cho từng câu.
- `configs/adaptive.yaml` và `configs/baseline.yaml`: cấu hình chạy tương ứng.
  Mặc định gốc vẫn chọn top 10%; phải dùng cấu hình adaptive để bật cải tiến.
- `src/compare_pdsm.py`: so sánh sáu quy tắc từ cùng một bộ điểm đã lưu,
  không cần chạy mô hình sáu lần. Chỉ cần NumPy để dùng công cụ này.
- `tests/test_adaptive_pdsm.py`: kiểm tra thuật toán, trường hợp biên và
  công cụ so sánh. Các dữ liệu trong kiểm thử là giả lập, không phải kết quả
  thực nghiệm trên giọng nói thật.

Hai sửa lỗi được áp dụng chung cho cả baseline và adaptive:

1. Khoảng âm vị lưu khung kết thúc dạng bao gồm, nên phép gộp và tô màu phải
   bao gồm khung cuối. Trước đây khung cuối bị bỏ, âm vị một khung bị rỗng.
2. Khi `floor(số âm vị × tỉ lệ)` bằng 0, cách cũ dùng lát cắt `[-0:]` và vô
   tình chọn tất cả. Nay chọn 0 âm vị. Chế độ phần trăm vẫn làm tròn xuống.

Baseline ở đây là cách chọn top phần trăm đã sửa lỗi, không phải bản tái tạo
nguyên trạng mọi lỗi của mã gốc. Không quy lợi ích từ hai sửa lỗi này cho
ngưỡng thích nghi. Hai phương pháp phải được chạy trên cùng mã đã sửa.

## Bước 1: Chuẩn bị môi trường và dữ liệu

Dùng môi trường Python 3.12 đã cài được các thư viện của dự án. Mô hình đầy
đủ cần các thư viện trong `requirements.txt`, công cụ espeak-ng và các mô
hình tải sẵn mà SQ-AST/WhisperX yêu cầu. Cách cài đầy đủ theo README của dự án;
không cần cài toàn bộ các thư viện này nếu chỉ so sánh bộ điểm đã xuất.

Trọng số SQ-AST thực tế được mã nguồn đọc tại `src/models/weights/<dimension>.pth`.
Với hai cấu hình mẫu chỉ MOS, cần `src/models/weights/mos.pth` từ dự án SQ-AST
được liên kết trong README. WhisperX còn cần tải được mô hình nhận dạng và
căn chỉnh. Nhật ký có sẵn ở `thuc-nghiem/outputs/test/20261102_2253/logs.log`
cho thấy lần chạy trước dừng khi tải mô hình căn chỉnh. Cần xử lý việc tải
mô hình ở môi trường chạy trước khi có thể thu điểm âm vị; thay ngưỡng không
giải quyết được lỗi tải này.

Đặt các WAV vào `../thuc-nghiem/inputs/test/`, tính từ thư mục mã nguồn này.
Hai cấu hình mới đã trỏ đến thư mục đó và tách đầu ra để giữ kết quả cũ.
Dùng một tập thí điểm nhiều câu, có câu ít lỗi và nhiều lỗi. Ví dụ 30–50 câu
phù hợp để bắt đầu khảo sát; đây không phải bảo đảm về độ mạnh thống kê.
Một file WAV hiện có chỉ đủ thử luồng xử lý, chưa đủ đánh giá cải tiến.

## Bước 2: Chạy mô hình một lần để lấy điểm

Mở terminal tại thư mục `synth-speech-eval`, trong môi trường đã cài mô hình:

```bash
python -m src.eval adaptive
```

Đầu ra nằm dưới `../thuc-nghiem/outputs/adaptive/test/<thời điểm chạy>/`.
Nếu chạy thành công và có câu qua bước lọc chất lượng, thư mục này có thêm
`phoneme_scores.csv` cùng các biểu đồ và `config_used.yaml`.

Có thể chạy `python -m src.eval baseline` để xem hình của top 10%, nhưng
không cần chạy thêm lần này để lấy bảng so sánh sáu quy tắc.

`score_threshold` mặc định bằng 3: chỉ những câu có điểm chất lượng dưới hoặc
bằng ngưỡng này mới được phân tích âm vị. Ngưỡng này khác với ngưỡng thích
nghi. Nếu không có câu nào qua bước lọc thì không có CSV điểm âm vị.
Khi đánh giá phải giữ bộ câu và ngưỡng chất lượng cố định; kết quả chỉ nói
về định vị trên bộ câu đã lọc, không đại diện cho phát hiện mọi câu có lỗi.
Nếu muốn phân tích tất cả các câu hợp lệ, đặt `score_threshold: 5` ở cả hai
cấu hình; quá trình sẽ lâu hơn.

## Bước 3: Gán nhãn lỗi và chia tập

Sao chép `phoneme_scores.csv` thành `phoneme_scores_labelled.csv`, giữ nguyên
các dòng và điểm. Mỗi dòng là một khoảng âm vị của một câu và một chiều.

- `start_seconds`, `end_seconds`: thời gian trên file gốc; khoảng là
  `[start, end)`, không bao gồm thời điểm kết thúc.
- `label`: điền `1` nếu âm vị có lỗi nghe được, `0` nếu không.
- `split`: điền `dev` để chọn tham số hoặc `test` để đánh giá sau cùng.

Người gán nhãn nên nghe và đánh dấu khoảng lỗi mà không xem saliency hay cột
`selected`. Có thể quy đổi khoảng lỗi sang nhãn âm vị bằng quy tắc định trước,
ví dụ giao nhau ít nhất 50% thời lượng âm vị. Dùng cùng quy tắc cho toàn bộ
dữ liệu. Nếu có nhiều người nghe, chốt quy tắc đồng thuận trước khi so sánh.
Nhãn tự suy ra từ saliency sẽ không kiểm chứng được cải tiến.

Chia theo **file gốc**, không chia ngẫu nhiên từng dòng âm vị. Mọi đoạn,
kênh và chiều của cùng file phải thuộc cùng tập. Ví dụ chia 60% file cho dev,
40% cho test; cả hai nên có ví dụ lỗi và không lỗi. Nếu có đủ dữ liệu,
kiểm tra thêm trên người nói hoặc hệ TTS chưa có trong dev.

Điền đủ nhãn và split cho mọi âm vị của mỗi nhóm câu/chiều được đánh giá.
Có thể để nguyên cả hai cột trống cho một nhóm chưa gán nhãn; nhóm đó chỉ
được thống kê số lượng chọn, không được tính độ chính xác. Công cụ không
chấp nhận chia cùng file sang cả dev và test.

Giữ các ký hiệu trống/đặc biệt từ bộ căn chỉnh nhất quán giữa các phương
pháp. Bản cải tiến này chưa thêm cơ chế lọc im lặng hoặc sửa căn chỉnh.

## Bước 4: So sánh và đọc kết quả

Thay đường dẫn CSV bên dưới bằng đường dẫn thực tế; nếu có dấu cách thì giữ
dấu ngoặc kép. Thư mục hiện tại vẫn là `synth-speech-eval`:

```bash
python -m src.compare_pdsm "DUONG_DAN/phoneme_scores_labelled.csv" --output-dir ../thuc-nghiem/outputs/so-sanh-pdsm
```

Sáu lựa chọn được tính trên cùng điểm: top 5%, 10%, 20% và adaptive với
lambda 0.5, 1.0, 1.5. Đầu ra:

- `comparison.csv`: số âm vị chọn, TP/FP/FN/TN, Precision, Recall, F1 và
  tỉ lệ báo nhầm; tách theo dev/test và từng chiều. Dòng `ALL` gộp đếm
  trên toàn bộ âm vị (micro). Không cộng các dòng ALL với dòng từng chiều.
- `selections.csv`: âm vị nào được từng lựa chọn đánh dấu, dùng để đối chiếu.
- `selected_on_dev.json`: lựa chọn có micro F1 cao nhất trên dev, gồm cả
  adaptive tốt nhất, tỉ lệ phần trăm tốt nhất và lựa chọn tốt nhất toàn bộ;
  mỗi lựa chọn kèm kết quả test của chính nó. Hòa F1 thì ưu tiên Precision rồi Recall.
  Nếu vẫn hòa, thứ tự sáu lựa chọn ở trên quyết định; không dùng test để chọn.
  Nếu thiếu nhãn dev hoặc dev không có lỗi, công cụ không tự chọn phương pháp.

Có thể chạy công cụ trên CSV chưa gán nhãn để xem số âm vị được chọn;
các chỉ số Precision/Recall/F1 khi đó được để trống, không tự tạo nhãn.
Precision/Recall/F1 có mẫu số bằng 0 được ghi bằng 0; FPR thiếu ví dụ âm tính
được để trống. Các chỉ số nhị phân ở đây khác phép đánh giá tương quan PDSM
trong paper, không so trực tiếp với số liệu trong bảng của tác giả.

Chọn lambda trên dev, khóa lựa chọn rồi mới diễn giải kết quả test. So sánh
adaptive đã chọn với top 10% và baseline phần trăm tốt nhất trên dev.
Nêu cả Recall và số báo nhầm vì giảm số âm vị chọn có thể tăng Precision
nhưng làm bỏ sót lỗi. Nếu adaptive không tốt hơn baseline thì báo cáo đúng
kết quả, không điều chỉnh tiếp theo test. Chỉ thay ngưỡng chọn không thay
đổi thứ hạng điểm saliency, nên không kỳ vọng tăng AUC của cùng bộ điểm.

## Kiểm tra phần mã mà không tải mô hình

Chỉ cần Python 3.12 và NumPy. Chạy từ thư mục mã nguồn:

```bash
python -m unittest discover -s tests -v
```

Các kiểm thử kiểm tra hành vi và cách tính chỉ số trên dữ liệu giả lập.
Đây không phải kiểm chứng hiệu quả định vị trên dữ liệu âm thanh thật.

## Khung nội dung báo cáo

1. Vấn đề: tỉ lệ cố định chưa thích nghi với mức phân bố lỗi của từng câu.
2. Đề xuất: chọn điểm âm vị vượt trung bình cộng lambda lần độ lệch chuẩn.
3. Thiết lập: dữ liệu, nhãn, chia dev/test, mô hình và pooling giữ cố định;
   công khai hai sửa lỗi chung của baseline và adaptive.
4. Kết quả: bảng trên test của top 10%, baseline tốt nhất và adaptive đã
   chọn từ dev; kèm vài ví dụ đúng, báo nhầm, bỏ sót.
5. Hạn chế: dữ liệu nhỏ, sai căn chỉnh, thiên lệch độ dài âm vị từ pooling
   và trường hợp lỗi phân bố đồng đều.
