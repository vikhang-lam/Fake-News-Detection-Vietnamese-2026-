# Phát Hiện Tin Giả Tiếng Việt - Fake News Detection Vietnamese 2026

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: 55K+](https://img.shields.io/badge/dataset-55K%2B%20records-green.svg)](https://github.com/vikhang-lam/Fake-News-Detection-Vietnamese-2026-)

Một hệ thống **phát hiện tin giả tiếng Việt** sử dụng **PhoBERT** và các kỹ thuật hiệu chỉnh hàm Loss hiện đại, đạt độ chính xác **99.42%** trên tập kiểm tra.

## 🎯 Tổng Quan

Dự án này xây dựng một pipeline hoàn chỉnh để **phân loại tin giả vs. tin thật** trong văn bản tiếng Việt, bao gồm:

- ✅ **Thu thập & tổng hợp dữ liệu** từ 5 nguồn độc lập (Kaggle + crawl)
- ✅ **Chuẩn bị dữ liệu chuyên sâu**: làm sạch, tiền xử lý tiếng Việt (tách từ, loại stopwords)
- ✅ **Huấn luyện mô hình** với PhoBERT-base và Focal Loss
- ✅ **Phân tích lỗi** chi tiết để cải thiện mô hình
- ✅ **Bộ dữ liệu công khai** 54,244 bản ghi với nhãn bán tự động

---

## 📊 Dataset

### Thành Phần Dữ Liệu

| Nguồn | Nhãn | Số lượng | Nền tảng |
|-------|------|---------|----------|
| Vietnamese Medical Fake News | Tin giả (y tế) | 4,925 | Kaggle |
| FakeNewsVN | Tin giả (đa chủ đề) | 820 | Kaggle |
| Vietnamese Fake News PBL7 | Tin giả (đa chủ đề) | 375 | Kaggle |
| Việt Tân & Thời Báo | Tin giả (chính trị) | 9,583 | Excel tùy chỉnh |
| VnExpress & Báo Nhân Dân | Tin thật | 39,426 | Crawl |
| **Tổng cộng** | - | **55,129** | - |

### Thống Kê Dữ Liệu (Sau Làm Sạch)

```
Tổng bản ghi:        54,244
├─ Tin thật (0):     38,978 (71.9%)
└─ Tin giả (1):      15,266 (28.1%)

Tỷ lệ mất cân bằng:   2.55x

Phân chia:
├─ Train (70%):      43,395 (31,182 thật, 12,213 giả)
├─ Validation (15%): 9,328
└─ Test (15%):       10,849 (7,796 thật, 3,053 giả)
```

### Thống Kê Độ Dài Content (Số Token Sau Tiền Xử Lý)

```
Toàn bộ dữ liệu:
├─ Mean:    331.9 token
├─ Median:  274 token
├─ Min:     30 token
└─ Max:     19,942 token

Tin thật:
├─ Mean:    311.8 token
└─ Median:  263 token

Tin giả:
├─ Mean:    383.2 token
└─ Median:  301 token

Mann-Whitney U test: p = 2.24e-77 (***)
```

---

## 🔧 Quy Trình Chuẩn Bị Dữ Liệu

### Phần 1: Thu Thập & Làm Sạch Thô

1. **Tải dữ liệu** từ Kaggle API và các file Excel cục bộ
2. **Xử lý cơ bản**:
   - Xóa 0 giá trị null
   - Xóa 0 chuỗi rỗng
   - Xóa 2 bản ghi trùng lặp
   - Lọc bản ghi < 10 từ
3. **Kết quả**: 54,946 → 54,946 bản ghi

### Phần 2: Tiền Xử Lý Văn Bản Tiếng Việt

Pipeline 10 bước xử lý văn bản (song song hóa với `pandarallel`):

```
1. Chuẩn hóa Unicode (NFC)
2. Chuyển chữ thường
3. Xóa URL, thẻ HTML
4. Xóa timestamp (dd/mm/yyyy, yyyy-mm-dd)
5. Xóa ký tự đặc biệt (giữ chữ Việt, số, khoảng trắng)
6. Tách từ tiếng Việt (ViTokenizer)
7. Xóa URL fragment, timestamp dạng "dd mm yyyy"
8. Xóa token số 1-2 chữ số & token Latin 1 ký tự
9. Xóa từ dừng (3,513 stopwords)
10. Lọc bản ghi < 30 token
```

**Kết quả tiền xử lý**:
- Giảm token trung bình: **581.1 → 327.9** (-43.6%)
- Giảm bản ghi: **54,946 → 54,244** (-702 outlier)

### Phần 3: Phân Tích Outlier

Sử dụng **Tukey Fence mở rộng** (ngưỡng: Q3 + 3×IQR):

```
Ngưỡng outlier:  1,173 token
Bản ghi outlier: 666 (1.2%)
Đề xuất:         Cắt ngắn tại 1,173 token khi huấn luyện
```

---

## 🧠 Mô Hình & Huấn Luyện

### Kiến Trúc Mô Hình

```
┌─────────────────────────────────┐
│     Input: Văn bản Tiếng Việt   │
└──────────────┬──────────────────┘
               │
        ┌──────▼──────┐
        │ Tokenizer   │ (PhoBERT)
        │ Max: 256    │ (Head+Tail Truncation)
        └──────┬──────┘
               │
        ┌──────▼──────────────┐
        │ PhoBERT-base        │
        │ (vinai/phobert)     │
        │ 12 layers, 768 dims │
        └──────┬──────────────┘
               │
        ┌──────▼──────┐
        │ Classification      │
        │ Head        │
        └──────┬──────┘
               │
        ┌──────▼──────────────┐
        │ Output (Tin giả/    │
        │ Tin thật)           │
        └─────────────────────┘
```

### Kịch Bản Huấn Luyện

#### 1️⃣ Baseline: Weighted CrossEntropy

```python
Loss = WeightedCrossEntropy(
    weight_real=1.0,
    weight_fake=2.25  # Bù cho mất cân bằng lớp
)
```

**Hyperparameter**:
- Learning Rate: 2e-5
- Batch Size: 32
- Epochs: 5
- Optimizer: AdamW
- Warmup Steps: 5% tổng steps

#### 2️⃣ Đề Xuất: Focal Loss

```python
Loss = FocalLoss(
    alpha=0.75,    # Trọng số lớp
    gamma=2.0      # Trọng số khó
)
```

**Lợi thế**: 
- Tập trung vào mẫu khó (hard examples)
- Giảm ảnh hưởng của mẫu dễ
- Tốt hơn cho dữ liệu mất cân bằng

---

## 📈 Kết Quả

### So Sánh Hiệu Suất (Tập Test: 10,849 bản ghi)

| Chỉ số | Baseline (WCE) | Focal Loss | Chênh lệch |
|--------|----------------|------------|-----------|
| **Accuracy** | **99.42%** ✅ | 99.18% | -0.24% |
| **Precision (macro)** | 99.38% | 99.27% | -0.11% |
| **Recall (macro)** | 99.19% | 98.70% | -0.49% |
| **F1-macro** | 99.28% | 98.98% | -0.30% |
| **F1 Tin giả** | **98.97%** ✅ | 98.52% | -0.45% |
| **Recall Tin giả** | **98.65%** ✅ | 97.60% | -1.05% |

### Confusion Matrix

#### Baseline (Weighted CE)
```
                 Predicted
                Thật    Giả
Actual  Thật    7,780   16  (FP)
        Giả       31  3,022 (FN)
        
Total errors: 47 / 10,849 (0.43%)
├─ False Positive (FP): 16 (tin thật → giả)
└─ False Negative (FN): 31 (tin giả → thật) ⚠️
```

#### Focal Loss
```
                 Predicted
                Thật    Giả
Actual  Thật    7,784   12  (FP)
        Giả       55  2,998 (FN)
        
Total errors: 67 / 10,849 (0.62%)
├─ False Positive (FP): 12 (tin thật → giả)
└─ False Negative (FN): 55 (tin giả → thật) ⚠️
```

### Phân Tích Chi Tiết Mẫu Sai

#### Baseline (Weighted CE) - 47 mẫu sai

**Đặc điểm mẫu sai**:
- Độ dài trung bình: **2,089 ký tự** (vs. 2,412 ký tự của mẫu đúng)
- Confidence trung bình: **0.963** (vs. 0.999 của mẫu đúng)
- Phân loại: 31 FN, 16 FP

**Nhận xét**: Mô hình có tự tin cao ngay cả khi dự đoán sai → cần khám phá thêm các đặc trưng khó phân biệt.

#### Focal Loss - 67 mẫu sai

**Đặc điểm mẫu sai**:
- Độ dài trung bình: **1,883 ký tự** (ngắn hơn)
- Confidence trung bình: **0.808** (thấp hơn, tốt hơn)
- Phân loại: 55 FN, 12 FP

**Nhận xét**: Focal Loss giúp mô hình bớt tự tin vào các dự đoán sai, nhưng tăng False Negative → Baseline vẫn tốt hơn cho bài toán này.

---

## 📁 Cấu Trúc Repository

```
Fake-News-Detection-Vietnamese-2026-/
├── README.md                                    # 📄 Tài liệu chính
│
├── FakeNews_DataPrep_TiengViet_2026.ipynb      # 🔧 Chuẩn bị dữ liệu
│   └── Output:
│       ├── fakenews_cleaned_full.csv            # Dữ liệu đầy đủ (54,244 bản ghi)
│       ├── fakenews_train.csv                   # Train (70%)
│       └── fakenews_test.csv                    # Test (20%)
│
├── FakeNews_ModelTraining_TiengViet_2026.ipynb # 🧠 Huấn luyện & đánh giá
│   └── Output:
│       ├── phan_tich_loi_fakenews.xlsx          # 📊 Chi tiết mẫu sai
│       ├── model_weighted_ce.pth                # Checkpoint baseline
│       └── model_focal_loss.pth                 # Checkpoint focal loss
│
└── data/                                         # 📊 Dữ liệu (từ Google Drive)
    ├── viettan_fake.xlsx
    ├── thoibao_fake.xlsx
    ├── vn_express_true.xlsx
    └── nhandan_true.csv
```

---

## 🚀 Cài Đặt & Sử Dụng

### 1. Yêu Cầu Môi Trường

```bash
Python >= 3.8
CUDA 11.8+ (tùy chọn, nếu dùng GPU)
```

### 2. Cài Đặt Thư Viện

```bash
# Nếu chạy notebook
pip install -q kagglehub pyvi requests pandas matplotlib seaborn openpyxl regex scikit-learn scipy wordcloud datasets pandarallel transformers torch

# Hoặc sử dụng requirements.txt (nếu có)
pip install -r requirements.txt
```

### 3. Chuẩn Bị Dữ Liệu

```bash
# Clone repository
git clone https://github.com/vikhang-lam/Fake-News-Detection-Vietnamese-2026-.git
cd Fake-News-Detection-Vietnamese-2026-

# Tải dữ liệu từ Google Drive
# (Hoặc chạy notebook FakeNews_DataPrep_TiengViet_2026.ipynb)
# Link Drive: https://drive.google.com/drive/folders/1n27Z90tOLnT5SZK96ntJ2WLl97EMDP_B

# Hoặc sử dụng dữ liệu đã xử lý từ CSV
```

### 4. Chạy Notebook

#### Phần 1: Chuẩn Bị Dữ Liệu
```bash
jupyter notebook FakeNews_DataPrep_TiengViet_2026.ipynb
```

**Kết quả**:
- `fakenews_cleaned_full.csv` (54,244 bản ghi)
- `fakenews_train.csv` (70%)
- `fakenews_test.csv` (20%)

#### Phần 2: Huấn Luyện Mô Hình
```bash
jupyter notebook FakeNews_ModelTraining_TiengViet_2026.ipynb
```

**Kết quả**:
- Mô hình checkpoint (`.pth`)
- File phân tích lỗi (`phan_tich_loi_fakenews.xlsx`)
- Biểu đồ đánh giá

---

## 📊 Trực Quan Hóa Dữ Liệu

### Phân Phối Lớp (Class Distribution)

```
Tin Thật (False): 38,978 (71.9%) ████████████████████████░░░
Tin Giả (True):   15,266 (28.1%) ██████████
```

### Phân Phối Độ Dài Nội Dung

```
Tin Thật: Median = 263 token, Mean = 311.8
Tin Giả:  Median = 301 token, Mean = 383.2

   │  ┌─────────────────────────────────┐
   │  │ Tin Giả dài hơn trung bình      │
   │  │ Có thể là đặc trưng phân loại   │
   │  └─────────────────────────────────┘
```

### Top 20 Từ Phổ Biến

**Tin Thật**: báo, đất, nước, tháng, năm, triệu, tỷ, công, người, đây, ...

**Tin Giả**: chữa, trị, khỏi, bệnh, giá, ăn, uống, sức, khỏe, thần, ...

→ **Nhận xét**: Tin giả tập trung vào y tế, làm đẹp, kiếm tiền; Tin thật về tin tức chính trị, kinh tế.

---

## 🔍 Phân Tích Lỗi

### Những Trường Hợp Mô Hình Sai

#### ❌ False Negative (Tin giả → Dự đoán là Tin thật)

**Nguyên nhân tiềm ẩn**:
- Tin giả nhân vật hóa: "Ông A nói rằng..." (giống tin thật)
- Nội dung ngắn, ít đặc trưng
- Tính cách bình thường, ít từ khóa nghi ngờ
- Tin giả "tắng kèm", pha trộn sự thật

**Ví dụ**:
```
"Theo báo chí, bệnh viện X vừa khám phá chữa khỏi bệnh Y"
→ Mô hình: 95% tin thật ❌ (sai)
→ Thực tế: tin giả (khẳng định sai)
```

#### ❌ False Positive (Tin thật → Dự đoán là Tin giả)

**Nguyên nhân tiềm ẩn**:
- Tin thật có nhiều từ khóa "nghi ngờ"
- Chứa số liệu không xác minh
- Tiêu đề kích động

**Ví dụ**:
```
"Cảnh báo: Loại cây này có thể gây [bệnh]..."
→ Mô hình: 60% tin giả ❌ (sai)
→ Thực tế: tin thật (cảnh báo hợp lệ)
```

### File Chi Tiết

📊 **`phan_tich_loi_fakenews.xlsx`** chứa:

| Cột | Nội dung |
|-----|---------|
| text | Nội dung gốc |
| label | Nhãn thực tế (0=Tin thật, 1=Tin giả) |
| pred_baseline | Dự đoán Baseline (WCE) |
| conf_baseline | Confidence Baseline |
| pred_focal | Dự đoán Focal Loss |
| conf_focal | Confidence Focal Loss |
| text_len | Độ dài nội dung |
| error_type | Loại lỗi (FN/FP) |

---

## 💡 Hướng Cải Thiện

### Ngắn Hạn

1. **Tăng dữ liệu**: Thu thập thêm tin giả từ các chủ đề khác
2. **Tinh chỉnh tokenizer**: Cân nhắc cắt ngắn tại 1,173 token (cutoff outlier)
3. **Thử mô hình khác**: PhoBERT-large, vDistilBERT
4. **Điều chỉnh ngưỡng**: Sử dụng threshold tùy chỉnh thay vì 0.5

### Dài Hạn

1. **Ensemble methods**: Kết hợp nhiều mô hình (Voting, Stacking)
2. **Multi-task learning**: Phân loại đồng thời lớp + lĩnh vực
3. **Explainability**: Sử dụng LIME/SHAP để giải thích dự đoán
4. **Xây dựng hệ thống**: API, Web UI, dashboard thời gian thực
5. **Cập nhật liên tục**: Retrain định kỳ với dữ liệu mới

---

## 📚 Công Nghệ Sử Dụng

| Lĩnh vực | Công Nghệ |
|----------|-----------|
| **NLP** | PyVi, Transformers, PhoBERT |
| **Học Sâu** | PyTorch, HuggingFace |
| **Xử Lý Dữ Liệu** | Pandas, NumPy, scikit-learn |
| **Đánh Giá** | scikit-learn metrics, Confusion Matrix |
| **Trực Quan** | Matplotlib, Seaborn, Plotly |
| **Tối Ưu** | Song song hóa (pandarallel), GPU |
| **Dataset** | Kaggle API, Google Drive |

---

## 📖 Tài Liệu Tham Khảo

### Bài Báo & Tài Liệu

- [PhoBERT: Pre-trained Language Models for Vietnamese (Pho18)](https://aclanthology.org/2020.acl-main.150/)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

### Dataset Gốc

- [Vietnamese Medical Fake News Dataset - Kaggle](https://www.kaggle.com/datasets/leviettrieu369/vietnamese-medical-fake-news-dataset)
- [FakeNewsVN - Kaggle](https://www.kaggle.com/datasets/chuynvinquc/fakenewvn)
- [Vietnamese Fake News Dataset PBL7 - Kaggle](https://www.kaggle.com/datasets/goumanguyen/vietnamese-fake-news-dataset-pbl7)

### Công Cụ

- [PyVi - Vietnamese Text Tokenizer](https://github.com/undertheseanlp/pyvi)
- [Transformers Library - HuggingFace](https://huggingface.co/transformers/)
- [Google Drive API](https://drive.google.com/)

---

## 👤 Tác Giả

**Vikhang Lam**
- GitHub: [@vikhang-lam](https://github.com/vikhang-lam)
- Repository: [Fake-News-Detection-Vietnamese-2026-](https://github.com/vikhang-lam/Fake-News-Detection-Vietnamese-2026-)

---

## 📄 Giấy Phép

Dự án này được cấp phép dưới **MIT License** - xem file [LICENSE](LICENSE) để chi tiết.

---

## 🤝 Đóng Góp

Chúng tôi chào đón mọi đóng góp! Hãy:

1. **Fork** repository
2. **Tạo branch** cho feature mới: `git checkout -b feature/YourFeature`
3. **Commit** thay đổi: `git commit -m 'Add YourFeature'`
4. **Push** đến branch: `git push origin feature/YourFeature`
5. **Mở Pull Request**

### Hướng Dẫn Đóng Góp

- Cải thiện hướng dẫn / tài liệu
- Báo cáo lỗi hoặc vấn đề
- Đề xuất tính năng mới
- Tối ưu hiệu suất mô hình

---

## ❓ FAQ

### Q: Dữ liệu ở đâu?
A: Dữ liệu được tổng hợp từ 5 nguồn:
- 3 dataset Kaggle công khai
- 2 file Excel tùy chỉnh + crawl từ VnExpress, Báo Nhân Dân
- Lưu trữ tại Google Drive: [Link](https://drive.google.com/drive/folders/1n27Z90tOLnT5SZK96ntJ2WLl97EMDP_B)

### Q: Mô hình có thể dùng cho ngôn ngữ khác không?
A: Không trực tiếp. Bạn cần:
- Dữ liệu đào tạo cho ngôn ngữ đó
- Tokenizer phù hợp
- PhoBERT chỉ được huấn luyện trên tiếng Việt

### Q: Độ chính xác 99.42% có quá tốt không?
A: Có thể do:
- Dữ liệu test không đa dạng
- Hệ thống dễ phân biệt
- Cần kiểm tra thêm trên dữ liệu thực tế

### Q: Làm thế nào để sử dụng mô hình trong production?
A: Tham khảo hướng cải thiện "Xây dựng hệ thống" để tạo API.

---

## 📞 Liên Hệ & Hỗ Trợ

Nếu bạn có câu hỏi, vấn đề, hoặc đề xuất:

- **Issues**: [GitHub Issues](https://github.com/vikhang-lam/Fake-News-Detection-Vietnamese-2026-/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vikhang-lam/Fake-News-Detection-Vietnamese-2026-/discussions)

---

## 🎓 Trích Dẫn

Nếu bạn sử dụng dự án này trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@github{vikhang_fake_news_2026,
  author = {Vikhang Lam},
  title = {Fake News Detection Vietnamese 2026},
  year = {2026},
  url = {https://github.com/vikhang-lam/Fake-News-Detection-Vietnamese-2026-}
}
```

---

## 📊 Thống Kê Repository

- **Tổng bản ghi dữ liệu**: 55,129 → 54,244
- **Độ chính xác mô hình**: 99.42%
- **Dữ liệu test**: 10,849 bản ghi
- **Thời gian huấn luyện**: ~15 phút (GPU)
- **Kích thước mô hình**: ~355 MB (PhoBERT-base)

---

**⭐ Nếu bạn thấy dự án hữu ích, vui lòng star repo này! ⭐**

---

*Cập nhật lần cuối: Tháng 6, 2026*
