# Fake News Detection - Vietnamese 2026 🔍

Một dự án machine learning để phát hiện tin giả tiếng Việt using advanced NLP techniques.

## 📋 Nội dung

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cách sử dụng](#cách-sử-dụng)
- [Dataset](#dataset)
- [Mô hình](#mô-hình)
- [Kết quả](#kết-quả)
- [Đóng góp](#đóng-góp)
- [Liên hệ](#liên-hệ)

## 🎯 Giới thiệu

Dự án này phát triển các mô hình deep learning để phát hiện và phân loại tin giả trong các bài viết tiếng Việt. Mục tiêu là giúp người dùng xác định thông tin sai lệch và tin không đáng tin cậy.

## ✨ Tính năng

- ✅ Xử lý dữ liệu tiếng Việt
- ✅ Tiền xử lý văn bản tự động (tokenization, normalization)
- ✅ Huấn luyện nhiều mô hình ML/DL
- ✅ Đánh giá chi tiết với các metrics (Accuracy, Precision, Recall, F1-Score)
- ✅ Hỗ trợ transfer learning
- ✅ API dễ sử dụng để dự đoán

## 🔧 Yêu cầu hệ thống

- Python 3.8+
- Jupyter Notebook
- GPU (tùy chọn, để huấn luyện nhanh hơn)

## 📦 Cài đặt

1. Clone repository:
```bash
git clone https://github.com/vikhang-lam/Fake-News-Detection-Vietnamese-2026-.git
cd Fake-News-Detection-Vietnamese-2026-
```

2. Tạo virtual environment (khuyến khích):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Cách sử dụng

### 1. Khám phá dữ liệu
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Tiền xử lý dữ liệu
```bash
jupyter notebook notebooks/02_data_preprocessing.ipynb
```

### 3. Huấn luyện mô hình
```bash
jupyter notebook notebooks/03_model_training.ipynb
```

### 4. Đánh giá mô hình
```bash
jupyter notebook notebooks/04_model_evaluation.ipynb
```

### 5. Dự đoán tin giả
```bash
jupyter notebook notebooks/05_prediction.ipynb
```

## 📊 Dataset

- **Nguồn dữ liệu**: [Thêm thông tin về nguồn dữ liệu]
- **Kích thước**: [Số lượng samples]
- **Các lớp**: 
  - 0: Tin thật
  - 1: Tin giả
- **Tỷ lệ**: [Tỷ lệ chia train/test]

## 🧠 Mô hình

### Các mô hình được thử nghiệm:

1. **Logistic Regression** - Baseline
2. **Naive Bayes** - Mô hình xác suất
3. **SVM (Support Vector Machine)** - Phân loại tuyến tính
4. **Random Forest** - Ensemble learning
5. **LSTM/GRU** - Deep learning
6. **Transformer (BERT tiếng Việt)** - Transfer learning

## 📈 Kết quả

| Mô hình | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| Logistic Regression | - | - | - | - |
| Random Forest | - | - | - | - |
| LSTM | - | - | - | - |
| BERT | - | - | - | - |

*Cập nhật kết quả sau khi huấn luyện*

## 🤝 Đóng góp

Chúng tôi rất hoan nghênh mọi đóng góp! Nếu bạn muốn cải thiện dự án:

1. Fork repository
2. Tạo branch feature (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 License

Dự án này được cấp phép dưới [Thêm loại license]

## 📧 Liên hệ

- **Tác giả**: vikhang-lam
- **Email**: [Thêm email]
- **GitHub**: [@vikhang-lam](https://github.com/vikhang-lam)

---

**Cảm ơn bạn đã quan tâm đến dự án này!** ⭐
