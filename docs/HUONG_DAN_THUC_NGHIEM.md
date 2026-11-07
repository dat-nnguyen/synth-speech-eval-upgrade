# Phân tích project và hướng dẫn thực nghiệm XSQ-AST

Tài liệu dựa trên mã nguồn trong workspace, kiểm tra ngày 07/11/2026.
Các lệnh bên dưới được chạy từ thư mục gốc `synth-speech-eval`.
Đã có audio `inputs/input_sample.wav`, kiểm tra sau khi người dùng thêm file.
Đã tải đủ năm checkpoint cải tiến từ nguồn chính thức và chạy thành công
nhánh chấm điểm trên audio này. Kết quả thực tế ở mục 0; các ví dụ giả định
ở phần khác vẫn được ghi rõ. Mã chạy đã tối ưu khởi tạo/cache và bộ nhớ khi
chỉ lấy điểm; nhánh giải thích chưa được sửa/kiểm chứng đầy đủ.

## 0. Chạy với audio thực tế vừa được thêm

### Khắc phục `ModuleNotFoundError: No module named torch` trên máy hiện tại

Python macOS đang dùng là 3.15, chưa cài dependency của project. Để chạy
bộ version hiện có, dùng môi trường Linux/Python 3.12 trong `Dockerfile.cpu`.
Không cần cài torch vào Python hệ thống của macOS.

```bash
# Chạy ở thư mục gốc project, Docker Desktop cần đang hoạt động.
# Image và weights trên máy hiện tại đã được chuẩn bị; không cần build lại.
bash scripts/run_cpu.sh input_sample_scores
```

Script tự build nếu chưa có image, kiểm tra weights theo các chiều trong
config và giữ model cache trong Docker volume `synth-speech-eval-cache`.
Input và output được mount vào `/work` từ chính thư mục repo trên máy.
Cần đặt các weights vào `src/models/weights/` trước khi chấm điểm; nếu thiếu,
script sẽ liệt kê file cần bổ sung rồi dừng trước khi khởi tạo mô hình.
Khi đã sửa requirements hoặc Dockerfile, chạy lại lệnh build trước khi dùng script.

Đã kiểm chứng image `synth-speech-eval-cpu:local`: Python 3.12.16,
`torch==2.8.0+cpu`, `torchaudio==2.8.0+cpu`, import `src.process.proc`
thành công, đọc audio thực tế thành đúng 4 dòng metadata. `pip check`
không báo xung đột. Python hệ thống macOS vẫn là môi trường riêng;
dùng script Docker bên trên để chạy project.

Lưu ý trạng thái: image đang có chưa áp dụng được bản pin TorchCodec 0.7
vì BuildKit gặp lỗi snapshot khi bổ sung layer. Phần import và chia đoạn
ở trên đã qua; chưa xác nhận chạy WhisperX/alignment. Dockerfile bên dưới
đã khai báo pin cho lần build mới; không cần build lại để dùng bước score
đã kiểm tra. Suy luận SQ-AST đã chạy thành công trên cả năm checkpoint.

Dockerfile CPU cố định `torchcodec==0.7.0` vì binary TorchCodec cần khớp
với torch 2.8 theo [bảng tương thích chính thức](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions).
Trong quá trình cài, pip cũng báo WhisperX 3.8.2 đã bị yanked do lỗi
word timestamp cho chữ số/ký hiệu và tương thích faster-whisper cũ.
Môi trường giữ version của repo để kiểm tra phần score; cần đánh giá nâng
WhisperX trước khi dùng alignment trong thí nghiệm B.


Đã đọc header và sample PCM của `inputs/input_sample.wav` bằng thư viện
chuẩn Python, không cần cài torch:

| Thuộc tính | Giá trị thực tế |
|---|---|
| Định dạng | WAV PCM 16-bit |
| Sample rate | 44.100 Hz |
| Số kênh | 2 (stereo), cả hai đều có sample khác 0 |
| Số frame audio | 692.729 |
| Thời lượng | 15,70814059 giây |
| Kích thước file | 2.770.960 byte |

File đáp ứng kiểm tra định dạng/độ dài của pipeline; chưa xác minh bằng
nghe về ngôn ngữ và nội dung giọng nói. SQ-AST tự resample lên 48 kHz nên
không bắt buộc chuyển file trước. Chạy công thức segmentation của chính
repo cho thấy mỗi kênh có hai đoạn:

| Kênh | Đoạn | Sample bắt đầu | Sample kết thúc (exclusive) | Khoảng giây |
|---|---|---|---|---|
| 0 | 1 | 0 | 379439 | 0–8,604059 |
| 0 | 2 | 313289 | 692728 | 7,104059–15,708118 |
| 1 | 1 | 0 | 379439 | 0–8,604059 |
| 1 | 2 | 313289 | 692728 | 7,104059–15,708118 |

Hai đoạn chồng lấn 1,5 giây. Công thức làm tròn hiện tại bỏ một sample
cuối ở file này (~22,7 micro giây); bảng trên phản ánh đúng code hiện tại.

Đã tạo `configs/input_sample_scores.yaml`, dùng directory mode để đọc WAV
ngay trong `inputs/`. Với môi trường và weights đã chuẩn bị trên máy này:

```bash
bash scripts/run_cpu.sh input_sample_scores
```

Input là file hiện có, không cần di chuyển. Nếu thêm WAV khác ở ngay trong
`inputs/`, config này sẽ đọc cả những file đó. Output thành công dự kiến:

```text
outputs/scores/inputs/<YYYYMMDD_HHMM>/
├── output_sq_ast.tsv     # 4 dòng cho file hiện tại, 5 cột điểm
├── config_used.yaml
└── logs.log
```

Các thư mục plot/sorted cũng được tạo nhưng rỗng do cấu hình chỉ lấy điểm.
Không có `output_thresholded.tsv` trong chế độ này. `dataset_name: inputs`
là tên thư mục dữ liệu, không phải tên file. Hướng dẫn bốn biến thể `demo_en`
ở các mục sau là một thực nghiệm
bổ sung riêng, không phải yêu cầu để chấm file vừa thêm.

### Kết quả thực tế đã kiểm tra

Lượt thành công: `outputs/scores/inputs/20261107_1339/`.
Log ghi thời gian xử lý 8 phút 44 giây trên CPU. Đã kiểm tra bốn dòng có
khóa file/kênh/đoạn riêng biệt và cả 20 điểm đều hữu hạn, trong thang 1–5.

| Kênh | Đoạn | MOS | NOI | DIS | COL | LOUD |
|---|---|---|---|---|---|---|
| 0 | 1 | 1.1165 | 1.1957 | 3.5562 | 2.0066 | 2.2176 |
| 0 | 2 | 1.1739 | 1.1173 | 3.5966 | 2.0904 | 2.3057 |
| 1 | 1 | 1.1338 | 1.2386 | 3.6405 | 2.0648 | 2.1120 |
| 1 | 2 | 1.1572 | 1.0942 | 3.5485 | 2.1521 | 2.2603 |

Đây là điểm mô hình dự đoán trên file hiện tại, không phải điểm người nghe.
Đọc giá trị đầy đủ trong `output_sq_ast.tsv`; `logs.log` có dòng
`Completed analysis in` và `config_used.yaml` lưu cấu hình thực tế.
Nguồn/checksum năm weights ở `src/models/weights/weights_manifest.json`.
Script CPU mặc định giới hạn bốn luồng; có thể đổi bằng
`SQ_AST_CPU_THREADS=8 bash scripts/run_cpu.sh input_sample_scores` nếu muốn
thực nghiệm hiệu năng trên máy khác. Khi threshold <1, code bỏ gradient
vì không có điểm nào được yêu cầu giải thích. Metadata AST ưu tiên cache.

## 1. Project giải quyết việc gì?

Project nhận audio giọng nói và dự đoán chất lượng, sau đó giải thích vùng
thời gian/tần số và các âm vị được mô hình chú ý ở những đoạn có điểm thấp.
Không cần audio tham chiếu hay transcript do người dùng nhập.

| Điểm đầu ra | Ý nghĩa | Chiều tốt |
|---|---|---|
| `sq_mos` | Chất lượng tổng thể dự đoán | Cao hơn tốt hơn |
| `sq_noi` | Chất lượng xét theo nhiễu | Cao hơn tốt hơn |
| `sq_dis` | Chất lượng xét theo gián đoạn | Cao hơn tốt hơn |
| `sq_col` | Chất lượng xét theo biến đổi màu âm | Cao hơn tốt hơn |
| `sq_loud` | Chất lượng xét theo độ lớn âm thanh | Cao hơn tốt hơn |

Cả năm điểm ở thang 1–5, tính bằng `4 * sigmoid(logit) + 1`.
MOS ở đây là dự đoán của mô hình, không phải điểm trung bình của người nghe.
Tên `Noisiness` không có nghĩa điểm cao là nhiều nhiễu hơn.

Đây là pipeline inference và phân tích; repo không có vòng lặp training,
optimizer, train/validation split, loss hay công cụ fine-tune. Repo cũng
không xuất nhãn thật/giả, xác suất deepfake hoặc audio đã được sửa lỗi.
Nhận định về chất lượng và saliency cần được đối chiếu bằng nghe thử;
saliency biểu thị liên hệ với quyết định mô hình, chưa chứng minh nguyên nhân lỗi.

## 2. Cấu trúc và luồng dữ liệu

```text
configs/default.yaml + configs/<tên>.yaml
                  |
inputs/<dataset>/*.wav
                  |
audio_segmentation.get_input_dataset
  -> metadata: một dòng cho một (file, kênh, đoạn)
                  |
sq_ast_mod.ASTVal
  -> lấy kênh, cắt đoạn, resample 48 kHz
  -> mel features: [1024, 128]
                  |
5 SQ-AST checkpoints độc lập
  -> điểm: [N, D], D tối đa 5
  -> saliency thô: [N, D, 12, 101]
                  |
lọc: có ít nhất một điểm <= score_threshold
                  |
WhisperX small + wav2vec2 + eSpeak
  -> transcript, thời gian từ, alignment âm vị
                  |
PDSM + KDE + biểu đồ
                  |
TSV + PNG + config_used.yaml + logs.log
```

| File | Vai trò |
|---|---|
| `src/eval.py` | Entry point CLI |
| `src/process/proc.py` | Đọc config, chọn thiết bị, điều phối, ghi kết quả |
| `src/util/audio_segmentation.py` | Liệt kê WAV, bỏ file quá ngắn, tách kênh/đoạn |
| `src/models/sq_ast_mod.py` | Tiền xử lý, SQ-AST, score, saliency, lọc threshold |
| `src/models/ppgs_wrapper.py` | WhisperX, alignment và biểu diễn âm vị theo thời gian |
| `src/process/pdsm.py` | Gộp saliency theo âm vị, chọn top-k, thống kê âm vị |
| `src/util/kde_tools.py` | Mật độ saliency theo thời gian/tần số, flatness |
| `src/util/plot_helper_ind.py` | Hình cho từng đoạn/kênh |
| `src/util/plot_helper_sys.py` | Hình tổng hợp toàn dataset |
| `app.py` | UI Streamlit upload ZIP và tải báo cáo ZIP |

`12 × 101` là lưới patch saliency; trong mảng kết quả này không phải
12 transformer layers. Saliency được nội suy thành `128 × 1024`, sau đó
cắt phần padding theo độ dài đoạn. Với 8 giây/48 kHz, số frame tính bởi
code là `floor((384000 - 1200) / 480) + 1 = 798`.

PDSM lấy nhãn âm vị có giá trị lớn nhất tại từng frame, gom các frame liên
tiếp cùng nhãn, rồi gộp saliency trên vùng mỗi âm vị bằng L2/mean/max...
Vì vậy L2/sum có thể ưu tiên âm vị dài hơn. PPG trong repo được dựng từ
alignment thành các giá trị 0/1; không phải posterior xác suất mềm trực tiếp.

KDE gom saliency theo hai trục. `kde_flatness_sq_*` là tỷ lệ trung bình
nhân/trung bình cộng của KDE theo thời gian: gần 1 là phân bố đều hơn,
nhỏ hơn là tập trung hơn. Đây không phải một điểm chất lượng bổ sung.

## 3. Điều kiện chạy và tình trạng workspace

Đã kiểm tra:

- Runtime hiện tại báo `Darwin x86_64`, Python `3.15.2`.
- Chưa tìm thấy các package torch, torchaudio, WhisperX, phonemizer,
  transformers, NumPy, pandas, SciPy, PyYAML, Streamlit trong Python đó.
- Không tìm thấy `conda`, `ffmpeg`, `espeak-ng` trên PATH; có `uv` và Homebrew.
- Sau khi người dùng thêm dữ liệu, `inputs/` có `input_sample.wav` stereo
  44,1 kHz, dài 15,708 giây; xem mục 0. Đã có đủ năm `.pth` trong
  `src/models/weights/`, tổng 1.724.085.225 byte.
- Nhánh score đã import, nạp strict đủ năm checkpoint và chạy inference
  thành công trong Docker. Python host macOS là môi trường riêng.

Repo yêu cầu Python 3.12 theo `environment.yml`. WhisperX 3.8.2 yêu cầu
Python >=3.10,<3.14 và torch/torchaudio ~=2.8.0.
Nguồn: [metadata chính thức WhisperX 3.8.2](https://raw.githubusercontent.com/m-bain/whisperX/v3.8.2/pyproject.toml).

Môi trường macOS x86_64 hiện tại không phù hợp với bộ pin: PyTorch đã dừng
phát hành binary macOS x86 sau dòng 2.2. Không nên chỉ hạ torch vì sẽ lệch
dependency của WhisperX. Nguồn: [thông báo PyTorch](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690).
Nếu máy là Apple Silicon nhưng đang chạy Rosetta, cần kiểm tra lại kiến trúc
Python; repo vẫn có đường dẫn eSpeak Linux cần điều chỉnh khi chạy native macOS.

Lộ trình thực tế: Linux x86_64/Python 3.12, CPU cho thử ít file hoặc NVIDIA
GPU cho tập lớn. Code chỉ tự chọn CUDA/CPU; không tự dùng Apple MPS và sẽ
ghi đè giá trị `device` người dùng thêm trong YAML.

### 3.1 Cài trên Ubuntu 24.04 x86_64

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv ffmpeg espeak-ng libsndfile1 libgomp1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Chọn **một** trong hai lệnh cài torch dưới đây.

CPU:

```bash
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU, sau khi hệ thống có driver tương thích CUDA 12.8:

```bash
nvidia-smi
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Các binary tương ứng được liệt kê trong
[hướng dẫn PyTorch phiên bản cũ](https://pytorch.org/get-started/previous-versions/).
WhisperX 3.8.2 hướng dẫn CUDA 12.8 cho GPU; CPU không cần CUDA.
Nguồn: [WhisperX 3.8.2](https://pypi.org/project/whisperx/3.8.2/).

Sau đó:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch, torchaudio, whisperx, transformers, phonemizer, yaml; print(torch.__version__, torchaudio.__version__, "cuda:", torch.cuda.is_available())'
ffmpeg -version
espeak-ng --version
```

Đây là quy trình cài đề xuất theo các version trong repo, chưa được cài và
kiểm chứng end-to-end trong phiên này. Nếu resolver báo xung đột, cần xử lý
xung đột trước khi inference; không dùng `--no-deps` để bỏ qua.
Code dưới `src/` không import ESPnet hoặc pyarmor; chưa cần thực hiện bước
ESPnet riêng trong README để chạy CLI nguồn. pyarmor trong requirements
phục vụ quy trình đóng gói, không phải thành phần mô hình chấm điểm.

### 3.2 Chạy từ Mac x86_64 bằng Linux container (CPU)

Nếu đã cài Docker Desktop, có thể mở shell Linux từ thư mục repo:

```bash
docker run --platform linux/amd64 -it --rm -v "$PWD":/work -w /work python:3.12-slim bash
```

Trong container:

```bash
apt-get update
apt-get install -y ffmpeg espeak-ng libsndfile1 libgomp1
python -m venv .venv-linux
source .venv-linux/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

Sau đó chạy các bước weights/input/CLI bên dưới trong shell đó. `.venv-linux`
và output nằm trên thư mục mount; không dùng venv Linux từ Python macOS.
Model cache trong container sẽ mất sau khi thoát nếu không mount cache riêng.
Lần đầu cần mạng để tải các tài nguyên chưa có trong cache. Lượt score
đã kiểm chứng trên máy này mất khoảng 8 phút 44 giây; xem mục 0.
Đây là phương án riêng; Dockerfile sẵn có trong repo cần `dist/` và
`pyarmor_runtime_000000/` chưa được tạo nên chưa build trực tiếp được.

### 3.3 Tải đúng weights

[Repo SQ-AST chính thức](https://github.com/WafaaWardah/SQ-AST) có hai bộ
weights, bản ban đầu và bản cải tiến. Chọn một bộ và ghi lại nguồn/checksum,
không trộn tùy tiện giữa các lần so sánh. Đường dẫn thực tế mà code local đọc:

```text
src/models/weights/
├── mos.pth
├── noi.pth
├── dis.pth
├── col.pth
└── loud.pth
```

```bash
mkdir -p src/models/weights
```

Tải/giải nén và chép năm file vào thư mục trên. README local ghi
`./app/models/weights/`, nhưng code đọc `src/models/weights/<dim>.pth`.
Chỉ khi đã ở trong Docker WORKDIR `/app` thì đường dẫn tuyệt đối tương ứng
là `/app/src/models/weights/`.

Có thể tải tự động bộ cải tiến từ link chính thức bằng Python chuẩn trên
máy host, không cần cài torch vào Python macOS:

```bash
python3 scripts/download_weights.py
```

Script tải theo nhiều phần, tiếp tục các phần còn thiếu nếu mạng ngắt,
kiểm tra kích thước và lưu SHA256 cùng nguồn ở
`src/models/weights/weights_manifest.json`. Tổng năm checkpoint khoảng
1,724 GB (1.724.085.225 byte).

```bash
python - <<'PY'
from pathlib import Path
import hashlib
for dim in ['mos', 'noi', 'dis', 'col', 'loud']:
    p = Path('src/models/weights') / f'{dim}.pth'
    assert p.is_file() and p.stat().st_size > 0, f'Missing: {p}'
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    print(p, p.stat().st_size, h.hexdigest())
PY
```

Ngoài năm file này, code cần cấu hình AST và feature extractor của
`MIT/ast-finetuned-audioset-10-10-0.4593` qua Hugging Face; các file nhỏ này
được cache trong Docker volume. Đã sửa khởi tạo `ASTXL` để tạo kiến trúc
từ config rồi nạp toàn bộ checkpoint SQ-AST với kiểm tra strict. Cách này
bỏ lượt tải weights AudioSet bị ghi đè ngay sau đó. Nhánh giải thích còn
tải Whisper small, wav2vec2 và các tài nguyên VAD/alignment liên quan.
Để chạy offline cần có cả weights và các file cấu hình/feature extractor
trong cache; riêng nhánh giải thích cần cache thêm các model tương ứng.

## 4. Input rõ ràng

Đầu vào khuyên dùng cho lần đầu: **WAV mono, PCM 16-bit, 48 kHz,
tiếng Anh, dài 2–10 giây và thực sự có giọng nói**.
Không cần nhãn, CSV hoặc transcript. 48 kHz là lựa chọn ổn định;
SQ-AST có resample nội bộ và WhisperX dùng 16 kHz.

```text
inputs/demo_en/
├── utt001_clean.wav
├── utt001_noise.wav
├── utt001_dropout.wav
└── utt001_quiet.wav
```

Quy tắc của code cần lưu ý:

- CLI chỉ đọc `.wav` chữ thường ở ngay thư mục dataset, không đệ quy.
- File ngắn hơn 2 giây bị bỏ; file đúng 2 giây được nhận.
- Mỗi kênh được phân tích riêng; kênh toàn số 0 bị bỏ. Kênh chỉ có nhiễu
  nhưng khác 0 vẫn qua bước kiểm tra này.
- File >10 giây được chia thành các đoạn chồng lấn 1,5 giây.
  Kiểm tra công thức trực tiếp: file 12 giây -> đoạn [0,6.75] và [5.25,12].
- `wav_start`, `wav_end` là chỉ số sample trên audio đầu vào; end dùng
  trong lát cắt Python nên là mốc loại trừ. Thời gian = chỉ số / `file_fs`.
- `wav_channel` bắt đầu từ 0. File không chia đoạn dùng `wav_segment=0`;
  file chia đoạn dùng 1,2,...
- Dùng thư mục kể cả khi chỉ có một WAV: nhánh single-file hiện ghép
  đường dẫn file với chuỗi rỗng có thể tạo dấu `/` sau tên file.

### 4.1 Chuẩn bị một câu thật

Đặt audio nguồn vào `raw/utt001.wav`; chọn câu có ít nhất 8 giây giọng nói.

```bash
mkdir -p inputs/demo_en
ffmpeg -i raw/utt001.wav -t 8 -ac 1 -ar 48000 -c:a pcm_s16le inputs/demo_en/utt001_clean.wav
ffprobe -v error -show_entries stream=sample_rate,channels,duration -of default=noprint_wrappers=1 inputs/demo_en/utt001_clean.wav
```

`-t 8` không kéo dài audio ngắn; cần kiểm tra file thực tế. Ví dụ mong muốn:
`sample_rate=48000`, `channels=1`, `duration=8.000000`.

### 4.2 Tạo thực nghiệm có vị trí lỗi biết trước

Sau khi cài NumPy/SciPy, chạy:

```bash
python - <<'PY'
from pathlib import Path
import csv
import numpy as np
from scipy.io import wavfile

root = Path('inputs/demo_en')
fs, pcm = wavfile.read(root / 'utt001_clean.wav')
assert fs == 48000 and pcm.ndim == 1 and pcm.dtype == np.int16
x = pcm.astype(np.float64) / 32768.0
assert 3.0 <= len(x) / fs <= 10.0
assert np.sqrt(np.mean(x*x)) > 1e-5, 'Need actual audible speech'
a, b = int(2.0 * fs), int(2.5 * fs)
rng = np.random.default_rng(42)

# Thêm nhiễu SNR 10 dB vào khoảng [2.0, 2.5) giây.
z = rng.standard_normal(b-a)
noise = x.copy()
target_rms = np.sqrt(np.mean(x[a:b]**2)) / (10 ** (10/20))
noise[a:b] += z * target_rms / np.sqrt(np.mean(z*z))
dropout = x.copy()
dropout[a:b] = 0
quiet = x * 0.1  # giảm biên độ 20 dB toàn câu

def save(name, y):
    assert np.max(np.abs(y)) <= 32767/32768, 'Clipping: choose lower-level source audio'
    wavfile.write(root / name, fs, np.round(y * 32768).astype(np.int16))

save('utt001_noise.wav', noise)
save('utt001_dropout.wav', dropout)
save('utt001_quiet.wav', quiet)
with (root / 'manifest.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['file', 'condition', 'artifact_start_s', 'artifact_end_s'])
    w.writerow(['utt001_clean.wav', 'clean', '', ''])
    w.writerow(['utt001_noise.wav', 'local_noise_snr10', 2.0, 2.5])
    w.writerow(['utt001_dropout.wav', 'dropout', 2.0, 2.5])
    w.writerow(['utt001_quiet.wav', 'gain_minus20dB', 0, len(x)/fs])
print('Prepared four WAV inputs and manifest.csv')
PY
```

Nghe trước khoảng 2–2,5 giây phải có giọng nói; nếu là im lặng, chọn khoảng
khác và cập nhật manifest. `manifest.csv` là ground truth thực nghiệm do ta
tạo, pipeline chỉ đọc bốn WAV. Nên chạy trong thư mục chưa có các WAV cũ
để số dòng đầu ra dễ kiểm tra. Audio biến đổi có thể là giọng tổng hợp
hoặc giọng thật; nếu muốn đánh giá TTS, audio clean nên là đầu ra TTS.

## 5. Thực nghiệm A: lấy điểm chất lượng trước

Đã tạo `configs/experiment_scores.yaml`. Nó merge với config default.
`score_threshold: 0` là cách dùng CLI hiện tại để chỉ lấy điểm: mọi score
đều >=1 nên không có mẫu đi vào WhisperX/PDSM/KDE. Đây là thủ thuật cấu hình,
không phải một chế độ scores-only chính thức. Nó vẫn import WhisperX nên
vẫn cần cài dependency. Code hiện tại bỏ gradient khi threshold <1 để
giảm bộ nhớ của nhánh chỉ lấy điểm.

```bash
python -m src.eval experiment_scores
```

Không truyền `.yaml` hoặc `configs/` vào đối số: CLI tự ghép hai phần này.
Input = bốn WAV trong `inputs/demo_en/` + năm weights + config đã merge.
Nếu chỉ muốn kiểm tra nhanh MOS, đổi `sq_ast_dims` trong config thành `[mos]`;
khi đó chỉ cần `mos.pth` và chỉ có cột `sq_mos`.

Output thành công:

```text
outputs/scores/demo_en/<YYYYMMDD_HHMM>/
├── output_sq_ast.tsv
├── config_used.yaml
├── logs.log
├── individual_plots/       # rỗng trong cấu hình này
├── system_plots/           # rỗng trong cấu hình này
└── sorted_tsvs/            # rỗng trong cấu hình này
```

Không có `output_thresholded.tsv` là đúng với thí nghiệm A.
Với đúng bốn WAV mono, mỗi WAV <=10 giây, output phải có bốn dòng.
Code đặt timestamp theo phút của máy chạy; hai lần cùng dataset/output_dir
trong một phút có thể ghi chung thư mục. Chờ sang phút khác hoặc đổi
`output_dir` trước lần chạy tiếp theo.

**Kiểm chứng kết quả:** CLI bắt nhiều exception nhưng không truyền lỗi ra
exit code, vì `run_evaluation()` bỏ qua return code. Vì vậy `$? = 0` chưa
chứng minh thành công. Kiểm tra cả file và dòng log `Completed analysis in`.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path('outputs/scores/demo_en')
runs = sorted(p for p in root.iterdir() if p.is_dir())
assert runs, 'No run directory'
run = runs[-1]
log = (run / 'logs.log').read_text()
assert 'Completed analysis in' in log and '| Error:' not in log, log
df = pd.read_csv(run / 'output_sq_ast.tsv', sep='\t')
cols = ['sq_mos', 'sq_noi', 'sq_dis', 'sq_col', 'sq_loud']
assert len(df) == 4, f'Expected 4 mono segments; got {len(df)}'
assert df[cols].notna().all().all()
assert ((df[cols] >= 1) & (df[cols] <= 5)).all().all()
assert df.file_path.is_unique
print('Run:', run)
print(df[['file_path'] + cols].to_string(index=False))

by_file = df.set_index('file_path')[cols]
delta = by_file.subtract(by_file.loc['utt001_clean.wav'])
delta.to_csv(run / 'delta_vs_clean.tsv', sep='\t')
print('\nDelta versus clean (negative = predicted quality decreased):')
print(delta.to_string())
PY
```

Điểm thực tế chỉ biết sau inference; không có giá trị MOS cố định cho bốn
file ví dụ. Giả thuyết kiểm tra: noise làm giảm `sq_noi`, dropout làm giảm
`sq_dis`, quiet làm giảm `sq_loud`. Chúng có thể làm giảm nhiều chiều cùng
lúc và mô hình có thể không phản ứng đúng; cần báo cáo cả trường hợp đó.

### Các cột trong `output_sq_ast.tsv`

| Cột | Diễn giải |
|---|---|
| `db` | Tên dataset; code hiện có thể ghi rỗng vì lấy basename đường dẫn có `/` cuối |
| `file_num` | Số thứ tự file theo `os.listdir`, không bảo đảm thứ tự cố định |
| `file_path` | Tên WAV tương đối trong thư mục dataset |
| `file_fs` | Sample rate của WAV đầu vào |
| `total_wav_channels`, `wav_channel` | Tổng số kênh và kênh đang chấm |
| `total_wav_segments`, `wav_segment` | Số đoạn và mã đoạn |
| `wav_start`, `wav_end` | Sample bắt đầu/kết thúc trên file gốc |
| `sq_mos`, `sq_noi`, `sq_dis`, `sq_col`, `sq_loud` | Điểm dự đoán của các chiều đã chọn |

Khóa ghép kết quả nên là `(file_path, wav_channel, wav_start, wav_end)`,
không dùng `file_num` để ghép hai lần chạy.

## 6. Thực nghiệm B: giải thích và định vị lỗi

**Nhánh này cần sửa/kiểm chứng các vấn đề ở mục 8 trước khi dùng kết quả
âm vị và alignment làm bằng chứng nghiên cứu.** Config được cung cấp ở
`configs/experiment_explain_template.yaml` là template, chưa chứng nhận
pipeline giải thích chạy đúng trên dữ liệu thực.

Sau khi xử lý các vấn đề tương ứng:

```bash
python -m src.eval experiment_explain_template
```

Input vẫn là bốn WAV; threshold mặc định template là 3; batch WhisperX =1;
PDSM lấy top 3 vùng âm vị để tránh lỗi tỷ lệ làm tròn về 0. GradCAM luôn
ép batch SQ-AST về 1 bất kể config khai báo lớn hơn.

Điều kiện vào nhánh giải thích là **ít nhất một chiều có score <=3**.
Với một dòng được giữ lại, chỉ các chiều đạt điều kiện mới có saliency/PDSM.
Ví dụ minh họa: MOS=3.6, NOI=2.4, DIS=4.1, COL=3.8, LOUD=4.0 thì dòng được
giữ do NOI; chỉ NOI được giải thích. Đây không phải số đo thật.

Nếu muốn kiểm tra nhánh giải thích trên mọi mẫu sau khi sửa code, dùng
threshold 5 trên tập rất nhỏ. Nó không có nghĩa mọi audio đều kém; chỉ
là yêu cầu phân tích toàn bộ. Tắt mọi cờ plot cũng không bỏ bước WhisperX
khi còn dòng dưới ngưỡng.

Output dự kiến khi có mẫu được giữ và xử lý hoàn tất:

```text
outputs/explain/demo_en/<YYYYMMDD_HHMM>/
├── output_sq_ast.tsv
├── output_thresholded.tsv
├── config_used.yaml
├── logs.log
├── sorted_tsvs/
│   ├── thredholded_sorted_sq_mos.tsv
│   └── ...                         # chỉ chiều có mẫu dưới ngưỡng
├── system_plots/
│   ├── sq_violin.png
│   ├── perc_under_thresh.png
│   ├── asr_conf_violin.png
│   ├── frequency_KDE_for_dims.png
│   ├── sys_single_hist_<dim>.png
│   └── sys_double_hist_<dim>.png    # chỉ khi có cặp đủ điều kiện
└── individual_plots/<tên_wav>/
    ├── 0_<dim>_Phoneme.png
    ├── 0_<dim>_KDE_Word.png
    ├── 0_KDE_Time_Word.png
    └── 0_ASR_Confidence.png
```

`thredholded` là lỗi chính tả thật trong tên file code xuất ra.
Audio nhiều kênh có thêm thư mục `ch0`, `ch1`,...; số 0 ở tên hình thay
bằng mã segment nếu audio được chia. `wav_chan_perc.png` chỉ xuất khi có
nhiều kênh. Nếu bật correlation, có thêm `sq_metric_corr.png` và
`sq_metric_corr_thresh.png`; template tắt vì chỉ vài mẫu chưa đủ ý nghĩa.
Các histogram có thể vắng khi không có âm vị/cặp hợp lệ.

`output_thresholded.tsv` có thêm:

| Cột | Nội dung và cách đọc |
|---|---|
| `pre_threshold_index` | Chỉ số dòng trước lọc |
| `index` | Chỉ số sau lọc |
| `mean_asr_conf`, `median_asr_conf` | Trung bình/trung vị score từ alignment; không phải WER |
| `transcription` | Danh sách từ cùng start/end/score được serialize vào ô TSV |
| `pdsm_sq_<dim>` | Các tuple `(id, âm vị, start_frame, end_frame)` quan trọng |
| `pdsm_sq_<dim>_num` | Hiện ghi sai; chưa dùng để đếm âm vị |
| `kde_flatness_sq_<dim>` | Độ phẳng KDE thời gian |

Các cột âm vị/KDE cho chiều không dưới ngưỡng chứa `n/a`.
Frame PDSM có bước 0,01 giây theo ý đồ code; để quy về file gốc cần cộng
`wav_start/file_fs`. Tuy nhiên repo cắt `1:-1` của PPG làm lệch trục thời
gian cần được kiểm chứng trước khi báo cáo mốc chính xác. Cột chứa list/dict
là chuỗi Python, không phải JSON; dùng `ast.literal_eval` khi cần đọc,
không dùng `eval`.

Nếu không có mẫu dưới ngưỡng: chỉ có bảng score cùng các output tổng quát
được bật; không có TSV thresholded và hình cá nhân. Nếu log có lỗi giữa
pipeline, ngay cả bảng score cũng có thể chưa được ghi vì code ghi cuối.

## 7. Thiết kế thực nghiệm có thể đưa vào báo cáo

### A. Độ nhạy của điểm với loại lỗi

Dùng 20–50 câu tiếng Anh ban đầu cho một thử nghiệm nhỏ, mỗi câu có clean,
noise, dropout, quiet. Đây là quy mô đề xuất, không phải mức tối thiểu
được repo chứng minh. Giữ nguyên câu, giọng, độ dài và định dạng giữa các
điều kiện. Lưu source/câu/giọng/điều kiện trong manifest và dùng seed cố định.

Tính cho từng câu và từng chiều: `delta = score_degraded - score_clean`.
Báo cáo trung bình, trung vị delta và tỷ lệ câu có delta <0. Lập bảng:

```text
condition | n_utterances | mean_delta_mos | mean_delta_noi |
mean_delta_dis | mean_delta_col | mean_delta_loud
```

Muốn đánh giá coloration, thêm biến thể lọc tần số được kiểm soát; ba
biến thể minh họa ở mục 4 không đủ để kết luận riêng về chiều này.
Không mặc định mọi loại lỗi chỉ tác động một chiều.

### B. Khả năng định vị vùng lỗi

Với noise/dropout biết trước ở [2.0,2.5), dùng nhánh giải thích đã sửa và
kiểm chứng; đo tỷ lệ khối lượng KDE/saliency nằm trong khoảng đó, so với
clean và tỷ lệ độ dài khoảng lỗi trên toàn câu. Chọn một quy tắc chuẩn hóa
và threshold cố định trước khi so sánh; bổ sung nghe thử các vùng nổi bật.
Không dùng một ảnh heatmap đơn lẻ để kết luận khả năng định vị.

### C. So sánh TTS A và B

Tách `inputs/tts_a/` và `inputs/tts_b/`, dùng cùng danh sách câu, cố gắng
kiểm soát người nói và các tham số sinh. Tạo hai config kế thừa default,
đổi dataset và giữ nguyên weights/config khác. Chạy score trước, ghép
theo mã câu, rồi so sánh cặp score và tỷ lệ dưới cùng threshold.
Đây là đánh giá bằng mô hình proxy; để kết luận chất lượng nghe cần
thêm đánh giá người nghe hoặc ground truth phù hợp.

### D. So sánh phương pháp giải thích

Chỉ so GradCAM/Raw/Flow/Rollout sau khi sửa attention ở mục 8. Các phương
pháp phải dùng cùng audio, checkpoint, threshold và phép đo định vị.
PDSM `k`, pool, KDE bandwidth ảnh hưởng giải thích, không thay đổi
định nghĩa score SQ-AST. Tăng threshold thường tăng chi phí giải thích.

Audio dài có overlap nên các segment không độc lập; đừng coi chúng là
nhiều câu độc lập khi tính độ tin cậy. Cần công bố cách gộp theo utterance:
trung bình, có trọng số độ dài, hoặc lấy đoạn tệ nhất phục vụ QA; overlap
cần được tính đến nếu gộp theo thời gian.

Lưu cùng báo cáo: commit repo, `pip freeze`, cấu hình đã dùng, SHA256
weights, manifest/checksum WAV, thiết bị, thời gian, số file/kênh/đoạn,
ngôn ngữ và các sửa đổi code. Repo chưa ghi đủ những metadata này tự động.

## 8. Những lỗi và hạn chế cần xử lý

Các mục dưới là kết quả đọc code; segmentation/top-k được kiểm tra bằng
thực thi công thức nhẹ. Nhánh score đã chạy thực tế; chưa chạy nhánh giải
thích để đo tác động của các lỗi âm vị/attention.

| Vị trí | Phát hiện | Hệ quả / việc cần làm |
|---|---|---|
| `app.py:14-15` | Import `dist.src...` nhưng không có `dist/` | UI chưa chạy trực tiếp từ bản source; dùng CLI hoặc sửa import sang `src` rồi kiểm tra UI |
| `Dockerfile:45-46` | COPY artifacts pyarmor chưa có | Build hiện chưa đầy đủ đầu vào; không coi Dockerfile là đường chạy sẵn |
| `ppgs_wrapper.py:16-21` | Mọi OS không phải Windows bị gán đường dẫn eSpeak Linux x86_64 | macOS/native ARM Linux cần tìm library đúng; biến môi trường bên ngoài hiện bị code ghi đè |
| `ppgs_wrapper.py:95-99,151-152` | `silences` bị ép int và dùng nhầm trục trong `ppgs_out[index,0,silences,sil_idx]` | Không gán nhãn silence đúng cho các frame; cần mask bool và gán theo trục thời gian |
| `ppgs_wrapper.py:108-118` | Cùng model từ điển âm vị được dùng align chữ viết và align âm vị | Alignment từ/confidence cần model CTC phù hợp chữ viết riêng; kiểm chứng tiếng Anh trước |
| `ppgs_wrapper.py:96-99` | Tạo id âm vị bằng enumerate dictionary nhưng silence dùng id từ dictionary gốc | Cần thống nhất mapping nội bộ và id silence, không dựa vào thứ tự dict |
| `ppgs_wrapper.py:162` | PPG đã có trục 1000 frame nhưng bị cắt `1:-1` | Kiểm tra lý do dịch/mất frame và đồng bộ với saliency; không mặc định đã đúng |
| `ppgs_wrapper.py:114` | `phonemize(..., language='en-us')` hard-code | Đổi YAML thành `vi` không tạo ra pipeline âm vị tiếng Việt đúng |
| `sq_ast_mod.py:94,363-389` | Chỉ trả `attentions[0]` (layer đầu), phía sau coi head là layer | Raw/Flow/Rollout không đúng mô tả nhiều layer; phải lấy đủ layers, giảm heads có chủ đích và chọn backend hỗ trợ attention |
| `pdsm.py:114-116` | `int(n_phonemes * k)` có thể bằng 0, slice `[-0:]` lấy tất cả | Đã tái hiện với 3 và 9 âm vị, k=0.1: lấy cả 3/9; cần quy tắc min/max k rõ ràng |
| `pdsm.py:108,125` | End được lưu kiểu inclusive nhưng slicing bỏ end | Mất frame cuối mỗi vùng, vùng 1 frame có thể rỗng; cần thống nhất quy ước interval |
| `proc.py:317` | Cột `_num` lấy len danh sách của toàn dataset | Giá trị lặp cho mọi dòng, không phải số âm vị của từng dòng; cần đếm theo từng list |
| `proc.py:252-254,319-321` | Raw KDE chỉ append cho chiều dưới ngưỡng | Bật `output_raw_kde=true` có thể lỗi độ dài cột khi có dòng chỉ đạt một số chiều; bổ sung placeholder |
| `proc.py:345,349-381` | Score ghi cuối, lỗi được bắt, CLI bỏ return code | Phân tích sâu lỗi có thể làm mất cả bảng score và vẫn exit 0; cần ghi score sớm, traceback và exit code phù hợp |
| `audio_segmentation.py:66-67` | basename đường dẫn có `/` cuối; `os.listdir` không sort | `db` có thể rỗng và thứ tự thay đổi; sửa metadata trước benchmark lớn |

Đối chiếu cấu trúc attention với
[mã AST của Transformers 4.57.6](https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py),
và dictionary CTC với
[mã alignment WhisperX 3.8.2](https://raw.githubusercontent.com/m-bain/whisperX/v3.8.2/whisperx/alignment.py).

Gợi ý sửa tối thiểu về ý tưởng cho silence: giữ mask bool cho các frame
chưa có nhãn, đặt giá trị vào `ppgs_out[index, 0, silence_id, mask]`, với
`silence_id` lấy từ chính mapping dùng dựng PPG. Sau đó phải kiểm tra
mỗi frame chỉ có một nhãn và khoảng im lặng vào đúng lớp. Đây là định hướng
sửa, chưa phải bản vá hoàn chỉnh cho toàn bộ vấn đề alignment.

GradCAM mặc định tránh nhánh Raw/Flow/Rollout lỗi indexing, nhưng vẫn cần
kiểm tra gradient, độ hữu hạn, bản đồ có khác 0 và sự phù hợp khi masking
vùng nổi bật. Không coi việc xuất được PNG là đủ kiểm chứng saliency.

Lộ trình áp dụng: cài môi trường đúng -> chạy A có score TSV hợp lệ ->
sửa/kiểm chứng B trên một câu có vị trí lỗi biết trước -> mở rộng dataset ->
ghi bảng so sánh và giới hạn kết luận. Với tiếng Việt, riêng phần âm vị
và alignment cần thiết kế lại cấu hình/ngôn ngữ và đánh giá độc lập.
