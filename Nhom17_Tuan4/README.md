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

| Mô hình | Accuracy | **Macro F1** | Weighted F1 | ∆ vs Baseline |
|---|---|---|---|---|
| LR gốc (baseline T2) | 0.5912 | **0.3400** | 0.5404 | — |
| LR + balanced (T3) | 0.6205 | **0.6284** | 0.6175 | +0.2884 |
| Linear SVM + balanced (T3) | 0.7296 | **0.7219** | 0.7309 | +0.3819 |
| LR + title+desc (T3 best) | 0.7492 | **0.7553** | 0.7452 | +0.4153 |
| **SVM + title+desc C=1.0 (T4 ★)** | **0.8208** | **0.8147** | **0.8196** | **+0.4747** |

> **Metric chính: Macro F1** | Test: 614 mẫu | Threshold ≥35 (15 lớp) | Seed=42 | Stratify=True

---

## Ablation Study – Phân tích vai trò của Title

| Đặc trưng | Accuracy | Macro F1 |
|---|---|---|
| Description only | 0.7296 | 0.7219 |
| Title only | 0.8290 | 0.8216 |
| **Title + Description (T4 ★)** | **0.8208** | **0.8147** |

**Kết luận:** Title là feature chính (F1 cao nhất khi dùng đơn lẻ). Kết hợp description giảm nhẹ < 1% Macro F1 nhưng giúp mô hình ổn định hơn ở các lớp ít mẫu, không bị phụ thuộc hoàn toàn vào title. → **Title + Description là lựa chọn tối ưu.**

---

## Cấu trúc thư mục

```
Nhom17_Tuan4/
├── notebooks/
│   ├── Nhom17_12_Tuan4.ipynb    ← Notebook chính (Hoàng Công Phú)
│   ├── 02_model_tuning.ipynb    ← GridSearch + Ablation Study (Huỳnh Xuân Mai)
│   └── 03_error_analysis.ipynb  ← Phân tích lỗi (Nguyễn Võ Gia Bảo)
├── demo/
│   └── app.py                   ← Streamlit dashboard
├── results/
│   ├── final_metrics.csv        ← Bảng kết quả chính thức
│   ├── confusion_matrix.png     ← Tự sinh khi chạy notebook
│   ├── error_analysis.csv       ← Mẫu sai chi tiết
│   ├── per_class_f1_tuan4.csv   ← Tự sinh khi chạy notebook
│   └── figures/                 ← Biểu đồ
├── data/
│   ├── raw/job_descriptions.csv ← Dữ liệu gốc (tự đặt vào)
│   └── processed/job_cleaned.csv← Tự sinh khi chạy notebook
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
```bash
jupyter notebook notebooks/Nhom17_12_Tuan4.ipynb
```
Chạy từ đầu đến cuối. Tất cả file trong `results/` sẽ tự sinh.

### Bước 4: Tối ưu mô hình
```bash
jupyter notebook notebooks/02_model_tuning.ipynb
```

### Bước 5: Phân tích lỗi
```bash
jupyter notebook notebooks/03_error_analysis.ipynb
```

### Bước 6: Chạy demo Streamlit
```bash
cd demo
python -m streamlit run app.py
```

---

## Phân công tuần 4

| Thành viên | Công việc | File minh chứng | Trạng thái |
|---|---|---|---|
| Hoàng Công Phú | Rà soát pipeline, ổn định số liệu, bảng so sánh | `Nhom17_12_Tuan4.ipynb` | Hoàn thành |
| Huỳnh Xuân Mai | GridSearch tham số C, Ablation Study, bảng model_tuning | `02_model_tuning.ipynb`, `results/model_tuning_results.csv` | Hoàn thành |
| Nguyễn Võ Gia Bảo | Phân tích lỗi 4 câu hỏi, demo Streamlit, README | `03_error_analysis.ipynb`, `demo/app.py` | Hoàn thành |

---

## Cải tiến tuần 4 so với tuần 3

| Cải tiến | Chi tiết |
|---|---|
| Kết hợp tốt nhất | LinearSVC + title+desc C=1.0 → Macro F1: 0.7553 → **0.8147** (+7.9%) |
| GridSearch C | C ∈ {0.1, 0.3, 0.5, 1.0, 2.0} – 5-fold CV → C=1.0 tối ưu |
| Ablation Study | Title-only F1=0.8216 vs Title+Desc F1=0.8147 (Δ<1%) → kết hợp ổn định hơn |
| Per-class F1 | Đủ 15 lớp (threshold ≥35 mẫu) – yêu cầu thầy mục 10 |
| Phân tích Other | Top 15 nghề bị gom, biểu đồ phân bố, nhận xét |
| Cross-validation | 5-fold StratifiedKFold kiểm tra độ ổn định |
| Inference time | Đo và so sánh 5 mô hình |
| Demo Streamlit | 3 tab: xu hướng / kết quả / dự đoán AI |

---

## Khai báo sử dụng AI
Nhóm sử dụng Claude (Anthropic) hỗ trợ cấu trúc notebook, viết hàm helper và README. Toàn bộ số liệu, kết quả thực nghiệm và phân tích do nhóm tự thực hiện trên dữ liệu thực.
