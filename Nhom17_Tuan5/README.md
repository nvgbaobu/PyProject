# Nhóm 17 – Đề tài 12: Job Trend IT
## Phân loại vai trò IT từ tin tuyển dụng

| Thông tin | Chi tiết |
|---|---|
| **Học phần** | Ngôn ngữ lập trình Python |
| **GVHD** | ThS.NCS. Hà Thanh Dũng |
| **Nhóm** | 17 |
| **Thành viên** | Nguyễn Võ Gia Bảo – Huỳnh Xuân Mai – Hoàng Công Phú |

---

## 1. Kết quả chính thức

Sau khi tối ưu hóa tham số và thực hiện các thử nghiệm kết hợp đặc trưng, mô hình tốt nhất đạt được là **LinearSVC (C=1.0, class_weight='balanced') kết hợp văn bản Title + Description**. 

Kết quả đánh giá trên tập Test (614 mẫu, random seed = 42, gom nhãn các lớp < 35 mẫu thành `Other IT Role`):

| Mô hình | Accuracy | **Macro F1** | Weighted F1 | ∆ vs Baseline |
|---|---|---|---|---|
| LR gốc (baseline T2) | 0.5912 | **0.3400** | 0.5404 | — |
| LR + balanced (T3) | 0.6205 | **0.6284** | 0.6175 | +0.2884 |
| Linear SVM + balanced (T3) | 0.7296 | **0.7219** | 0.7309 | +0.3819 |
| LR + title+desc (T3 best) | 0.7492 | **0.7553** | 0.7452 | +0.4153 |
| **SVM + title+desc C=1.0 (T4 ★)** | **0.8208** | **0.8147** | **0.8196** | **+0.4747** |

> **⚠️ Lý do dùng Macro F1:** Dữ liệu phân bổ rất mất cân bằng (lớp `Other IT Role` chiếm hơn 31.1%, trong khi một số vị trí như `Embedded Engineer` chỉ chiếm 1.7%). Sử dụng Accuracy sẽ bị thiên lệch lớn về nhóm đa số, do đó nhóm chọn **Macro F1** làm thước đo đánh giá hiệu suất thực tế của mô hình.

---

## 2. Cấu trúc mã nguồn & Cải tiến kỹ thuật

Dự án bao gồm 3 tệp Jupyter Notebook tương ứng với các phân đoạn công việc đã thực hiện trong Tuần 4:

### 2.1. `01_final_data_pipeline.ipynb` (Xây dựng Pipeline chính thức)
- Đọc, làm sạch (loại bỏ trùng lặp, chuẩn hóa nhãn) và gom nhãn các vị trí có ít mẫu. Tổng số dữ liệu chốt: 3067 mẫu (chia Train/Test với tỷ lệ 80/20, `stratify=True`).
- Thiết lập quy trình chuẩn hóa và huấn luyện 5 mô hình trên cùng một tập dữ liệu đầu vào để so sánh, đảm bảo sự nhất quán cho bảng kết quả đầu ra.

### 2.2. `02_model_tuning.ipynb` (Tinh chỉnh siêu tham số)
- **GridSearch CV (5-fold):** Tiến hành tìm kiếm siêu tham số `C` tối ưu cho mô hình `LinearSVC` với dải giá trị `{0.1, 0.3, 0.5, 1.0, 2.0}`. Kết quả ghi nhận `C=1.0` mang lại CV Macro F1 trung bình cao nhất (0.7645).
- **Ablation Study:** So sánh hiệu suất của các nhóm đặc trưng. Dù `Title only` mang lại Macro F1 cao (0.8216) nhưng việc kết hợp `Title + Description` giúp mô hình ổn định hơn ở các lớp ít mẫu mà không bị quá phụ thuộc vào tiêu đề tin tuyển dụng.

### 2.3. `03_error_analysis.ipynb` (Phân tích lỗi chuyên sâu)
- **Đánh giá mức độ sai lệch:** Tỷ lệ phân loại sai trên tập test là 17.9% (110/614 mẫu). Top các cặp phân loại nhầm lẫn nhiều nhất xoay quanh `Other IT Role` và `Backend Developer`.
- **Chẩn đoán nguyên nhân:**
  - Lớp *Other IT Role* chứa tới hơn 314 chức danh bị gom lại nên quá rộng và phân mảnh bộ từ vựng.
  - Sự chồng lấp ngữ nghĩa giữa những vị trí có mô tả công việc (JD) tương đồng cao (VD: *Backend Developer* vs *Software Engineer*).
  - Thiếu tín hiệu nhận diện đặc trưng trong các tin tuyển dụng mô tả quá ngắn.

---

## 4. Định hướng hoàn thiện cho Tuần 5

Dựa trên phân tích lỗi của Tuần 4, nhóm đã vạch ra các hướng tiếp cận sau:
- Khảo sát tách các lớp có số lượng mẫu tiệm cận (như Web Developer) ra khỏi lớp `Other IT Role` để làm giảm nhiễu.
- Thử nghiệm kết hợp thêm đặc trưng Character n-gram nhằm nắm bắt các từ khóa kỹ năng chứa ký tự đặc biệt (VD: `C++`, `C#`, `.NET`) vốn dễ bị mất khi dùng Word-level Tokenizer.
- Đưa ra các biện pháp xử lý riêng đối với các tin tuyển dụng có mô tả công việc (description) quá ngắn.

## Phân công tuần 5

| Thành viên | Công việc | File minh chứng | Trạng thái |
|---|---|---|---|
| Hoàng Công Phú | Rà soát pipeline, ổn định số liệu, bảng so sánh | `01_final_data_pipeline.ipynb` | Hoàn thành |
| Huỳnh Xuân Mai | GridSearch tham số C, Ablation Study, bảng model_tuning | `02_final_model.ipynb`, `results/model_tuning_results.csv` | Hoàn thành |
| Nguyễn Võ Gia Bảo | Phân tích lỗi 4 câu hỏi, demo Streamlit, README | `03_error_analysis.ipynb`, `demo/app.py` | Hoàn thành |

## Cấu trúc thư mục

Nhom17_Tuan4/
├── notebooks/
│   ├── 01_final_data_pipeline.ynb    ← Notebook chính (Hoàng Công Phú)
│   ├── 02_final_data_pipeline.ipynb    ← GridSearch + Ablation Study (Huỳnh Xuân Mai)
│   └── 03_error_analysis_demo.ipynb  ← Phân tích lỗi (Nguyễn Võ Gia Bảo)
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

## Khai báo sử dụng AI
Nhóm sử dụng Claude (Anthropic) hỗ trợ cấu trúc notebook, viết hàm helper và README. Toàn bộ số liệu, kết quả thực nghiệm và phân tích do nhóm tự thực hiện trên dữ liệu thực.