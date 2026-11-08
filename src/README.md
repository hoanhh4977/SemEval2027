# Task 8 (StereoQueerEval): source code + hướng dẫn chạy trên Kaggle

## 1. Phương pháp đề xuất ("P")

Một encoder đa ngôn ngữ (mặc định XLM-RoBERTa-base) đọc cả **comment và ngữ cảnh video**, rồi giải
đồng thời 3 subtask bằng các đầu (head) dùng chung encoder:



```
 <cls> comment <sep><sep> title | description <sep>
                 │
          encoder (dùng chung)  ->  vector h (token <cls>)
                 │
   ┌─────────────┼──────────────────────────────┐
 head B       head A  (nhìn [h, softmax(B)])   head C (nhìn [h, softmax(B)])
 no/implicit/  stereotype yes/no               target group/individual  +  9 identity (sigmoid)
 explicit
```

Các thành phần (mỗi thành phần có một thí nghiệm ablation để chứng minh nó có ích):

| Thành phần | Ý nghĩa | Ablation |
|---|---|---|
| Ngữ cảnh title+description | comment ngắn thường không đủ nghĩa nếu thiếu video | `A_P_no_context`, `A_P_title_only` |
| Multi-task A/B/C | 3 nhãn liên quan nhau, học chung được lợi | `A_P_single_task`, `B_xlmrb_single_*` |
| Điều kiện phân cấp (hier) | A và C nhìn xác suất của B (hate có hay không) | `A_P_no_hier` |
| Loss cân bằng lớp (sqrt) | nhãn hiếm (`q`, `i`, `a`, `nb`) nặng ngang nhãn phổ biến, vì metric là macro-F1 | `A_P_no_cw`, `A_P_balanced_cw` |
| Giảm LR theo tầng (LLRD 0.9) | tầng thấp thay đổi chậm, ổn định hơn với dữ liệu nhỏ | `A_P_no_llrd` |
| Supervised contrastive loss | kéo các comment cùng mức hate lại gần nhau (ý tưởng từ SemEval-2023 Task 3) | `A_P_no_supcon` |
| C chỉ huấn luyện/đánh giá trên comment hateful | đúng như định nghĩa của task | (luôn bật) |

Thêm vào để so sánh: FGM (huấn luyện đối kháng), pre-train tiếp trên dữ liệu miền (DAPT), nhiều backbone,
chuyển ngôn ngữ (train EN → test IT/NL/FA), đường cong theo lượng dữ liệu, LLM + LoRA.

P là sự kết hợp các kỹ thuật đã biết cho đúng bài toán này, không phải một kiến trúc hoàn toàn mới.
Đóng góp của nó được **đo bằng ablation** (5 seed, kiểm định cặp). Thành phần nào không có ích thì sẽ bị bỏ.

## 2. Cấu trúc thư mục

```
src/
  plan.json            kế hoạch: round -> session -> 2 lane (2 GPU) -> danh sách thí nghiệm
  run_session.py       *** lệnh chính: chạy 1 session của plan ***
  run_experiments.py   worker chạy vài thí nghiệm trên 1 GPU (run_session gọi nó)
  plan_summary.py      in/ghi thời gian ước tính của từng session
  analyze.py           (chạy trên máy bạn) bảng kết quả + kiểm định thống kê từ file zip
  requirements.txt
  task8/               code lõi, mỗi file một việc:
     data.py           đọc dữ liệu, mã hoá nhãn, chia train/val/test theo video
     encoding.py       text -> token ids -> batch
     model.py          encoder + các head A/B/C
     losses.py         các hàm loss
     trainer.py        vòng huấn luyện, chọn epoch tốt nhất
     metrics.py        metric chính thức + chỉnh ngưỡng
     classical.py      baseline không neural (majority, random, TF-IDF)
     dapt.py           pre-train tiếp (MLM) trên văn bản của task
     pipeline.py       chạy 1 (thí nghiệm, seed): chia dữ liệu -> train -> chấm điểm -> lưu
     runner.py         vòng lặp thí nghiệm x seed, zip, bỏ qua lỗi, giới hạn thời gian
     experiments.py    danh sách tên thí nghiệm = cấu hình mặc định + phần thay đổi
```

## 3. Kế hoạch (xem `plan.json`)

Mỗi session = 1 phiên Kaggle (GPU T4 x2) = 2 lane chạy song song. Thời gian là **ước tính** (10 phút / 1 run
chuẩn); round R0 sẽ đo tốc độ thật và tôi sẽ chỉnh lại.

| Round | Session | Tên notebook | Nội dung | Giờ ước tính (gpu0 / gpu1) |
|---|---|---|---|---|
| R0 | `r0s1` | `t8-r0s1-calibration` | chạy thử + đo tốc độ, 1 seed | ~0.3 / ~1.1 (ngắn, ngoại lệ duy nhất) |
| R1 | `r1s1` | `t8-r1s1-core-comparison` | baseline, so sánh chính, ablation | ~9.1 / ~9.9 |
| R2 | `r2s1` | `t8-r2s1-backbones-extensions` | FGM, DAPT, backbone khác, chuyển ngôn ngữ, đường cong dữ liệu | ~9.7 / ~9.6 |
| R3 | `r3s1`, `r3s2` | `t8-r3s1-llm`, `t8-r3s2-llm-7b-rest` | (tuỳ chọn) LLM Qwen + LoRA | ~5 / ~9.6 và ~3 / ~3 |
| R4 | (để dành) | | hệ thống cuối: ensemble, train lại trên toàn bộ dữ liệu | sau R2 |

Giữa các round, bạn gửi kết quả cho tôi. Tôi phân tích rồi có thể sửa các round sau.
Chạy theo thứ tự R0 → R1 → R2. Đừng gộp session.

## 4. Chạy trên Kaggle (làm một lần)

**Bước 1. Tạo 2 dataset riêng tư** (Kaggle: *Create → New Dataset*, để **Private**):

| Tên dataset | Upload cái gì |
|---|---|
| `semeval27-t8-data` | thư mục `data/task8` (có `train/` và `dev/`). Dữ liệu thuộc DUA, **không để public** |
| `semeval27-src` | thư mục `src/` này (chứa `run_session.py`, `task8/`, `plan.json`...) |

Khi sửa code thì vào dataset `semeval27-src` → *New Version* → upload lại.

**Bước 2. Tạo notebook** (*Create → New Notebook*), đặt tên **đúng theo cột "Tên notebook"** ở bảng trên
(ví dụ `t8-r1s1-core-comparison`). Quy ước tên: `t8-<id session>-<mô tả ngắn>`.
Trong notebook:
* *Add Input* → thêm cả hai dataset ở bước 1.
* Settings: **Accelerator = GPU T4 x2**, **Internet = On**.

**Bước 3. Dán 3 cell** (đổi `r1s1` thành session muốn chạy):

```python
# Cell 1: chép code ra /kaggle/working và cài thư viện
import glob, os, shutil
src = os.path.dirname(glob.glob('/kaggle/input/**/run_session.py', recursive=True)[0])
shutil.copytree(src, '/kaggle/working/src', dirs_exist_ok=True)
!pip install -q -r /kaggle/working/src/requirements.txt
!nvidia-smi -L
```
```python
# Cell 2: xem kế hoạch (tuỳ chọn) và kiểm tra lệnh sẽ chạy
!python /kaggle/working/src/plan_summary.py
!python /kaggle/working/src/run_session.py --session r1s1 --dry_run
```
```python
# Cell 3: chạy thật (mất khoảng 10 giờ)
!python /kaggle/working/src/run_session.py --session r1s1
```
Nếu `nvidia-smi -L` chỉ thấy 1 GPU, thêm `--one_gpu` vào lệnh ở Cell 3 (hai lane chạy lần lượt, lâu gấp đôi.
Runner sẽ tự không bắt đầu run nào vượt giới hạn giờ; phần chưa chạy xong sẽ báo cho tôi để xếp lại).

**Bước 4. Chạy.**
* **R0 (ngắn):** bấm *Run All* ngay trong notebook, theo dõi log trực tiếp.
* **R1, R2, R3 (dài):** bấm **Save Version → "Save & Run All (Commit)"**. Kaggle chạy nền, bạn có thể đóng trình
  duyệt. Một session tối đa 12 giờ, nên plan dừng bắt đầu run mới sau **11 giờ** (`hard_stop_hours`).
  Với session ~10 giờ, thường vẫn còn dư.

**Bước 5. Lấy kết quả.** Khi xong, vào trang notebook → tab *Output* (hoặc mục Versions → Output) và tải:
* `r1s1.zip`: file gộp, **gửi cho tôi file này**.
* `r1s1_gpu0.zip`, `r1s1_gpu1.zip`: bản từng lane, được cập nhật sau **mỗi thí nghiệm**.
  Nếu session chết giữa chừng thì gửi hai file này (vẫn đầy đủ phần đã xong).

Trong zip có `result.json` + `preds.npz` của từng (thí nghiệm, seed), `log_*.txt`, và `errors_*.jsonl` nếu có lỗi.
Trọng số model không được lưu nên file nhỏ (vài chục MB).

## 5. Sau khi chạy R0, bạn kiểm tra giúp tôi

1. Có file `errors_*.jsonl` không? Nếu có thì gửi tôi.
2. Trong log, dòng `seed 0: done in X min` cho từng thí nghiệm (đây là tốc độ thật). Gửi cả `r0s1.zip`.
3. Tôi sẽ sửa `minutes_per_cost_unit` trong `plan.json` và chạy `plan_summary.py --write` để cập nhật giờ ước tính.

## 6. Lưu ý Kaggle

* GPU có hạn mức theo tuần (thường **30 giờ/tuần**, xem ở *Settings → Quota* trong tài khoản). Một session 10 giờ
  tốn khoảng 10 giờ. Nên chạy R0 + R1 trong một tuần, R2 tuần sau.
* Chạy lại cùng session thì các (thí nghiệm, seed) đã xong trong **cùng** phiên đó sẽ bỏ qua. Phiên mới của
  Kaggle bắt đầu từ thư mục trống, vì vậy phần bị bỏ lại do hết giờ sẽ được tôi xếp vào session sau.
* Lỗi thường gặp: *CUDA out of memory* → run đó ghi vào `errors_*.jsonl`, các run khác vẫn chạy; báo tôi để giảm
  batch size. *Không tải được model* → kiểm tra Internet = On.

## 7. Thử nhanh trên máy cá nhân (không cần GPU)

```
pip install -r src/requirements.txt
python src/run_experiments.py --exps B_majority B_tfidf_comment --seeds 0 1 --out out_test
python src/analyze.py --inputs out_test.zip --ref B_majority --out report_test
```
