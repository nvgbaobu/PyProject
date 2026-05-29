# Nhóm 17 – Đề tài 12: Job Trend IT
## Xây dựng hệ thống phân tích xu hướng thị trường việc làm CNTT

**Học phần:** Ngôn ngữ lập trình Python  
**GVHD:** ThS.NCS. Hà Thanh Dũng  
**Thành viên:** Nguyễn Võ Gia Bảo | Huỳnh Xuân Mai | Hoàng Công Phú  
**Tuần hiện tại:** Tuần 4 (25/5 – 01/6/2026)

---

## Tóm tắt đề tài

Phân loại văn bản mô tả công việc (`description`) từ bộ dữ liệu **IT Job Posting** (3.101 tin tuyển dụng) để tự động xác định vai trò CNTT (`it_role_type`). Sau gom nhãn còn **13 lớp** (Tuần 4 tách thêm DevOps Engineer + Data Engineer khỏi Other).

**Metric chính: Macro F1-score** (dữ liệu mất cân bằng – Other IT Role chiếm ~33%)

---

## Tiến độ kết quả

| Tuần | Mô hình | Macro F1 |
|------|---------|----------|
| Tuần 2 | LR Baseline | 0.4050 |
| Tuần 3 | LR + class_weight balanced | 0.6957 |
| Tuần 3 | Linear SVM + balanced | 0.7282 |
| Tuần 3 | LR + title+description | 0.7921 |
| **Tuần 4** | **SVM + title+desc (C=0.5)** | **~0.80+** |

---

## Cấu trúc thư mục

```
Nhom17_Tuan4/
├── notebooks/
│   ├── Nhom17_12.ipynb          ← Notebook chính (toàn bộ pipeline)
│   ├── 02_model_tuning.ipynb    ← GridSearch, Ablation Study (Huỳnh Xuân Mai)
│   └── 03_error_analysis.ipynb  ← Phân tích lỗi chuyên sâu (Nguyễn Võ Gia Bảo)
├── demo/
│   └── predict.py               ← Script demo dự đoán
├── results/
│   ├── final_metrics.csv        ← Bảng kết quả chính thức
│   ├── confusion_matrix.png     ← Ma trận nhầm lẫn (mô hình tối ưu)
│   ├── error_analysis.csv       ← Danh sách mẫu sai chi tiết
│   └── figures/                 ← Biểu đồ phụ
│       ├── bieu_do_so_sanh_5_mo_hinh.png
│       ├── gridsearch_C.png
│       ├── per_class_f1_tuan4.png
│       └── phan_loai_loi.png
├── data/
│   ├── job_descriptions.csv     ← Dữ liệu gốc (tự đặt vào đây)
│   └── job_cleaned.csv          ← Tự sinh sau khi chạy notebook chính
├── README.md                    ← File này
└── requirements.txt             ← Thư viện cần cài
```

---

## Hướng dẫn chạy

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

### 2. Đặt dữ liệu

Đặt file `job_descriptions.csv` vào thư mục `data_sample/`.

### 3. Chạy pipeline chính

Mở và chạy toàn bộ `notebooks/Nhom17_12.ipynb` từ trên xuống.  
Notebook sẽ tự động tạo các file trong `results/`.

### 4. Chạy tối ưu mô hình

```bash
jupyter notebook notebooks/02_model_tuning.ipynb
```

### 5. Chạy phân tích lỗi

```bash
jupyter notebook notebooks/03_error_analysis.ipynb
```

### 6. Chạy demo

```bash
# Interactive demo (nhập tay hoặc chọn mẫu)
python demo/predict.py

# Batch demo từ file CSV
python demo/predict.py --input data_sample/sample_input.csv
```

---

## Cải tiến tuần 4 so với tuần 3

| Cải tiến | Chi tiết |
|----------|----------|
| Tách lớp khỏi Other | DevOps Engineer (~45 mẫu) và Data Engineer (~38 mẫu) thành lớp riêng |
| Tối ưu tham số | GridSearch C ∈ {0.1, 0.3, 0.5, 1.0, 2.0} cho LinearSVC |
| Kết hợp tốt nhất | SVM (phương pháp chính) + title+description (đặc trưng tốt nhất) |
| Ablation study | Kiểm chứng title không phải shortcut/leakage |
| Cross-validation | 5-fold StratifiedKFold để đánh giá ổn định |
| Phân tích lỗi | 20 mẫu sai tiêu biểu, phân loại nguyên nhân, biểu đồ per-class |

---

## Nhận xét thầy tuần 3 đã xử lý

| Vấn đề thầy nêu | Tuần 4 đã xử lý |
|-----------------|-----------------|
| Số mẫu không thống nhất (3.067 vs 3.101) | Thống nhất pipeline, bộ số liệu chính thức |
| Cần per-class F1 cho mô hình tốt nhất | Có trong `03_error_analysis.ipynb` |
| Nguy cơ shortcut từ title | Ablation study trong `02_model_tuning.ipynb` |
| Other IT Role quá rộng | Tách DevOps + Data Engineer |
| Thiếu title-only baseline | Có trong ablation study |

---

## Phân công tuần 4

| Thành viên | Công việc | File minh chứng |
|------------|-----------|-----------------|
| Hoàng Công Phú | Rà soát pipeline, thống nhất số liệu, tách lớp | `Nhom17_12.ipynb` (Cell 1-4) |
| Huỳnh Xuân Mai | GridSearch, ablation study, bảng kết quả | `02_model_tuning.ipynb` |
| Nguyễn Võ Gia Bảo | Phân tích lỗi, demo, báo cáo | `03_error_analysis.ipynb`, `demo/predict.py` |

---

## Tài liệu tham khảo

1. Fan, R.-E., et al. (2008). LIBLINEAR: A Library for Large Linear Classification. *JMLR*.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*.
3. Manning, C., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge.
4. Dataset: IT Job Posting – Son Phat Tran (GitHub). LinkedIn, ITViec, TopCV.
5. Sun, Y., et al. (2009). Class-Rebalancing Methods for Short Text. *WWW Conference*.
