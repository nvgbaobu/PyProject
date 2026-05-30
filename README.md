# Nhóm 17 – Đề tài 12: Job Trend IT
## Phân loại vai trò IT từ tin tuyển dụng

| | |
|---|---|
| **Học phần** | Ngôn ngữ lập trình Python |
| **GVHD** | ThS.NCS. Hà Thanh Dũng |
| **Nhóm** | 17 |
| **Thành viên** | Nguyễn Võ Gia Bảo – Huỳnh Xuân Mai – Hoàng Công Phú |
| **Tuần** | 4 (25/5 – 30/5/2026) |

---

## Kết quả chính thức Tuần 4

| Mô hình | Accuracy | **Macro F1** | ∆ vs Baseline |
|---|---|---|---|
| LR gốc (baseline T2) | 0.6221 | **0.4720** | — |
| LR + balanced (T3) | 0.6270 | **0.6509** | +0.1789 |
| Linear SVM + balanced (T3) | 0.7052 | **0.7065** | +0.2345 |
| LR + title+desc (T3) | 0.7117 | **0.7349** | +0.2629 |
| SVM + title-only (T4) | 0.7720 | **0.7674** | +0.2954 |
| **SVM + title+desc C=1.0 (T4 ★)** | **0.7769** | **0.7741** | **+0.3021** |

> **Metric chính: Macro F1** | Test: 614 mẫu | Threshold ≥35 (15 lớp) | Seed=42 | Stratify=True

---

## Cấu trúc thư mục

```
Nhom17_Tuan4/
├── notebooks/
│   ├── Nhom17_12_Tuan4.ipynb    ← Notebook chính (Hoàng Công Phú)
│   ├── 02_model_tuning.ipynb    ← GridSearch + Ablation Study (Huỳnh Xuân Mai)
│   └── 03_error_analysis.ipynb  ← Phân tích lỗi (Nguyễn Võ Gia Bảo)
├── demo/
│   └── predict.py               ← Demo dự đoán
├── results/
│   ├── final_metrics.csv        ← Bảng kết quả chính thức ✓
│   ├── confusion_matrix.png     ← Tự sinh khi chạy notebook
│   ├── error_analysis.csv       ← Mẫu sai chi tiết ✓
│   ├── per_class_f1_tuan4.csv   ← Tự sinh khi chạy notebook
│   └── figures/                 ← Biểu đồ
├── data_sample/
│   └── sample_input.csv         ← Mẫu demo batch
├── README.md
└── requirements.txt
```

---

## Hướng dẫn chạy pipeline

### Bước 1: Cài thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Đặt dữ liệu
Đặt `job_descriptions.csv` vào `data/raw/`.

### Bước 3: Chạy notebook chính
```
jupyter notebook notebooks/Nhom17_12_Tuan4.ipynb
```
Chạy từ đầu đến cuối. Tất cả file trong `results/` sẽ tự sinh.

### Bước 4: Tối ưu mô hình
```
jupyter notebook notebooks/02_model_tuning.ipynb
```

### Bước 5: Phân tích lỗi
```
jupyter notebook notebooks/03_error_analysis.ipynb
```

### Bước 6: Demo
```bash
# Interactive (chọn mẫu sẵn hoặc nhập tay)
python demo/predict.py

# Batch predict
python demo/predict.py --input data_sample/sample_input.csv
```

---

## Phân công tuần 4

| Thành viên | Công việc | File minh chứng | Trạng thái |
|---|---|---|---|
| Hoàng Công Phú | Rà soát pipeline, ổn định số liệu, bảng so sánh | `Nhom17_12_Tuan4.ipynb` | Hoàn thành |
| Huỳnh Xuân Mai | GridSearch tham số C, Ablation Study, bảng model_tuning | `02_model_tuning.ipynb`, `results/model_tuning_results.csv` | Hoàn thành |
| Nguyễn Võ Gia Bảo | Phân tích lỗi 4 câu hỏi, demo script, README | `03_error_analysis.ipynb`, `demo/predict.py` | Hoàn thành |

---

## Cải tiến tuần 4 so với tuần 3

| Cải tiến | Chi tiết |
|---|---|
| Kết hợp tốt nhất | SVM (mô hình tốt nhất) + title+desc (đặc trưng tốt nhất) → Macro F1: 0.7349 → **0.7741** |
| Tối ưu C | GridSearch C ∈ {0.1, 0.3, 0.5, 1.0, 2.0} → chọn C=1.0 |
| Title-only baseline | Phân tích shortcut: title-only F1 = 0.7674 vs title+desc F1 = 0.7741 (Δ < 1%) |
| Per-class F1 | Đủ tất cả 15 lớp (threshold ≥35) |
| Phân tích Other | Top 15 nghề bị gom, biểu đồ, nhận xét |
| Ablation study | Xác nhận title+desc kết hợp là tối ưu |
| Inference time | Đo và so sánh 6 mô hình |
| Cross-validation | 5-fold để kiểm tra ổn định |

---

## Khai báo sử dụng AI
Nhóm sử dụng Claude (Anthropic) hỗ trợ cấu trúc notebook, viết hàm helper và README. Toàn bộ số liệu, kết quả thực nghiệm và phân tích do nhóm tự thực hiện.
