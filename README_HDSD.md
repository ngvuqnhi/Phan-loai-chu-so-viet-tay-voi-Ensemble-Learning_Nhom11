# HƯỚNG DẪN SỬ DỤNG ĐỒ ÁN HỌC MÁY

## Lời mở đầu

Kính gửi Cô,

Em xin phép trình bày hướng dẫn chi tiết cách chạy và sử dụng các model trong đồ án này. Em đã cố gắng tổ chức code và viết hướng dẫn sao cho Cô có thể dễ dàng reproduce lại kết quả. Tất cả các notebook đều được test kỹ và chạy thành công trên cả môi trường local và Google Colab.

---

## Tổng quan về đồ án

Đồ án này thực hiện bài toán phân loại chữ số viết tay MNIST sử dụng 6 thuật toán Machine Learning khác nhau:

**Ba thuật toán cơ bản:**
- Softmax Regression (Hồi quy Softmax)
- Naive Bayes (Bayes ngây thơ)
- Decision Tree (Cây quyết định)

**Ba phương pháp ensemble:**
- Hard Voting (Kết hợp bằng bỏ phiếu)
- Random Forest (Rừng ngẫu nhiên)
- AdaBoost (Boosting thích ứng)

Ngoài ra, em còn xây dựng 2 ứng dụng demo sử dụng Gradio để Cô có thể test trực tiếp bằng cách vẽ chữ số.

---

## Yêu cầu hệ thống

**Môi trường Python:**
- Python 3.8 hoặc cao hơn (em dùng Python 3.10)
- Jupyter Notebook hoặc Google Colab

**Các thư viện cần thiết:**
```bash
pip install numpy pandas matplotlib scikit-learn gradio pillow
```

Thư viện `pickle` đã có sẵn trong Python nên không cần cài thêm.

**Lưu ý về Google Colab:** Em khuyến nghị Cô dùng Google Colab để chạy vì:
1. Không cần cài đặt môi trường
2. Có đủ RAM để train các model lớn (đặc biệt là AdaBoost)
3. Có thể tạo public link cho các demo app

---

## Cấu trúc files trong đồ án

Em xin giải thích về các files trong project:

### Files dữ liệu

**processed_data.npz** (khoảng 18MB)
- Đây là file quan trọng nhất, được dùng cho hầu hết các notebook
- Chứa dữ liệu MNIST đã được tiền xử lý:
  - Chuẩn hóa pixel values về khoảng [0, 1]
  - Reshape từ (28, 28) thành vector (784,)
  - Chia sẵn thành 3 tập: training (48,000), validation (12,000), và test (10,000)
- Format: NumPy compressed array (.npz)

**Các file MNIST raw** (t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte, v.v.)
- Đây là dữ liệu MNIST gốc ở định dạng IDX
- Em giữ lại để tham khảo, nhưng không dùng trực tiếp trong code
- Đã được xử lý thành processed_data.npz rồi

### Files model đã train

Sau khi train xong, mỗi model được lưu thành file .pkl:
- **decision_tree_model.pkl** (222KB): Cây quyết định đã train
- **naive_bayes_model.pkl** (62KB): Model Naive Bayes
- **my_softmax.pkl** (62KB): Softmax regression
- **hard_voting_model.pkl** (345KB): Ensemble của 3 models trên
- **adaboost_bundle.pkl** (30MB): Model AdaBoost với tất cả weak learners
- **random_forest.pkl** (kích thước tùy thuộc số trees): Model Random Forest

### Files code Python

**decision_tree.py**
- File này rất quan trọng cho Random Forest
- Chứa định nghĩa các class: Node, DecisionTreeClassifier, RandomForest
- Cả notebook training Random Forest và demo app đều cần file này
- Lý do: Khi load model từ .pkl, Python cần biết định nghĩa các class này

### Các Jupyter Notebooks

Em đã tổ chức thành 3 nhóm:

**Nhóm 1: Base Models**
- Softmax.ipynb
- Bernoulli_Naive_Bayes.ipynb
- Decision_Tree.ipynb

**Nhóm 2: Ensemble Models**
- Hard_Voting.ipynb
- RandomForest.ipynb
- AdaBoost_final.ipynb

**Nhóm 3: Demo Applications**
- adaboost_demo.ipynb (demo AdaBoost)
- RF_DEMO.ipynb (demo Random Forest)

---

## Hướng dẫn chi tiết cách chạy từng model

### Phần 1: Ba thuật toán cơ bản
#### 1.1. Softmax Regression

**File cần có:**
- processed_data.npz

**Notebook:**
- Softmax.ipynb

**Cách thực hiện:**

Bước 1: Mở notebook Softmax.ipynb

Bước 2: Upload file processed_data.npz

Bước 3: Run All

**Quá trình thực thi:**
- Load và chuẩn hóa dữ liệu (standardization)
- Thêm bias term (thêm cột 1 vào cuối)
- Train bằng gradient descent
- Monitor loss trên cả training và validation sets
- Vẽ learning curves
- Đánh giá trên test set
- Lưu weights và bias thành my_softmax.pkl

**Thời gian chạy:** Khoảng 1-2 phút

**Kết quả mong đợi:**
- Accuracy trên test set: khoảng 89-91%
- File output: my_softmax.pkl (chứa dictionary với keys 'W' và 'b')

**Giải thích thuật toán:** Softmax regression là extension của logistic regression cho multi-class classification. Em sử dụng cross-entropy loss và gradient descent để tối ưu. Learning rate và số epochs được tune để đảm bảo hội tụ nhưng không overfit.

---
#### 1.2. Naive Bayes

**File cần có:**
- processed_data.npz

**Notebook:**
- Bernoulli_Naive_Bayes.ipynb

**Cách thực hiện:**

Bước 1: Mở notebook Bernoulli_Naive_Bayes.ipynb

Bước 2: Upload file processed_data.npz

Bước 3: Run All

**Quá trình thực thi:**
- Load dữ liệu
- Chuyển đổi dữ liệu sang binary (threshold = 0.5) vì dùng Bernoulli Naive Bayes
- Train model với Laplace smoothing (alpha = 1.0)
- Đánh giá performance
- Lưu model thành naive_bayes_model.pkl

**Thời gian chạy:** Rất nhanh, chỉ khoảng 30 giây

**Kết quả mong đợi:**
- Accuracy trên test set: khoảng 83-84%
- File output: naive_bayes_model.pkl

**Giải thích thuật toán:** Em sử dụng Bernoulli Naive Bayes vì phù hợp với binary features. Dữ liệu được binarize với threshold 0.5 (pixel > 0.5 → 1, ngược lại → 0). Model giả định các pixel độc lập với nhau, tuy không hoàn toàn đúng nhưng vẫn cho kết quả khá tốt.

---

#### 1.3. Decision Tree (Cây quyết định)

**File cần có:**
- processed_data.npz

**Notebook:**
- Decision_Tree.ipynb

**Cách thực hiện:**

Bước 1: Mở notebook Decision_Tree_Complete.ipynb

Bước 2: Upload file processed_data.npz vào cùng thư mục với notebook

Bước 3: Chạy toàn bộ notebook từ đầu đến cuối (Run All)

**Quá trình thực thi:**
- Load và kiểm tra dữ liệu
- Train Decision Tree với các hyperparameters: max_depth=10, min_samples_split=2
- Đánh giá trên tập validation
- Test trên tập test cuối cùng
- Hiển thị confusion matrix và classification report
- Lưu model thành decision_tree_model.pkl

**Thời gian chạy:** Khoảng 2-5 phút tùy cấu hình máy

**Kết quả mong đợi:**
- Accuracy trên test set: khoảng 87-88%
- File output: decision_tree_model.pkl

**Giải thích thuật toán:** Em sử dụng CART (Classification and Regression Trees) với Gini impurity làm tiêu chí split. Cây được xây dựng theo chiều sâu tối đa 10 để tránh overfitting.

---

### Phần 2: Phương pháp Ensemble

#### 2.1. Hard Voting

**Files cần có:**
- processed_data.npz
- decision_tree_model.pkl
- naive_bayes_model.pkl
- my_softmax.pkl

**Lưu ý quan trọng:** Cô cần train xong 3 base models ở phần 1 trước khi chạy notebook này.

**Notebook:**
- Hard_Voting.ipynb

**Cách thực hiện:**

Bước 1: Đảm bảo đã có đủ 4 files (1 file data + 3 files models)

Bước 2: Upload tất cả vào cùng thư mục

Bước 3: Mở notebook Hard_Voting.ipynb

Bước 4: Run All

**Quá trình thực thi:**
- Load dữ liệu và 3 models đã train
- Đánh giá performance của từng model riêng lẻ trên validation set
- Chọn "leader" (model có accuracy cao nhất trên validation)
- Implement hard voting strategy:
  - Nếu 2/3 hoặc 3/3 models đồng ý: chọn kết quả đa số
  - Nếu cả 3 models cho kết quả khác nhau: nghe theo leader
- So sánh accuracy của ensemble với từng model đơn lẻ
- Lưu ensemble model thành hard_voting_model.pkl

**Thời gian chạy:** Gần như instant vì chỉ inference, không có training

**Kết quả mong đợi:**
- Accuracy trên test set: khoảng 90-91% (cao hơn hầu hết các base models)
- File output: hard_voting_model.pkl

**Giải thích phương pháp:** Hard voting là ensemble method đơn giản nhưng hiệu quả. Ý tưởng là "nhiều đầu tốt hơn một đầu" - các models khác nhau có thể bù lỗi cho nhau. Em implement thêm leader selection để xử lý trường hợp tie-breaking.

---

#### 2.2. Random Forest

**Files cần có:**
- processed_data.npz
- decision_tree.py (RẤT QUAN TRỌNG!)

**Lưu ý đặc biệt:** File decision_tree.py là bắt buộc vì chứa định nghĩa các class mà model cần. Khi load model từ .pkl, Python cần import các class này.

**Notebook:**
- RandomForest.ipynb

**Cách thực hiện:**

Bước 1: Upload 2 files: processed_data.npz và decision_tree.py vào cùng thư mục

Bước 2: Mở notebook 05_RandomForest_FN.ipynb

Bước 3: Run All

**Quá trình thực thi:**
- Import các class từ decision_tree.py
- Load dữ liệu
- Train Random Forest với các tham số:
  - n_estimators: 100-200 trees
  - max_depth: None (trees được phép phát triển tối đa)
  - max_features: 'sqrt' (random feature selection)
- Mỗi tree được train trên bootstrap sample
- Mỗi split chỉ xem xét sqrt(784) ≈ 28 features ngẫu nhiên
- Tính feature importances
- Tính out-of-bag (OOB) score
- Đánh giá trên test set
- Lưu model thành random_forest.pkl

**Thời gian chạy:** 5-15 phút tùy số trees

**Kết quả mong đợi:**
- Accuracy trên test set: khoảng 94-96%
- OOB score: tương đương test accuracy
- File output: random_forest.pkl

**Giải thích thuật toán:** Random Forest là bagging ensemble - train nhiều Decision Trees độc lập trên các bootstrap samples và random feature subsets, rồi average predictions. Method này giảm variance và rất khó overfit. OOB score cho phép đánh giá model mà không cần validation set riêng.

---

#### 2.3. AdaBoost

**Files cần có:**
- processed_data.npz
- (Optional) adaboost_bundle.pkl nếu chỉ muốn test model có sẵn

**Notebook:**
- AdaBoost.ipynb

**Cách thực hiện:**

**Trường hợp 1: Train model từ đầu**

Bước 1: Upload processed_data.npz

Bước 2: Mở notebook AdaBoost_final.ipynb

Bước 3: Run All

Bước 4: Đợi training hoàn thành (10-30 phút)

**Trường hợp 2: Load model đã train sẵn (nhanh hơn)**

Bước 1: Upload cả processed_data.npz và adaboost_bundle.pkl

Bước 2: Mở notebook

Bước 3: Skip các cell training, chỉ chạy các cell load model và evaluation

**Quá trình thực thi khi train:**
- Load dữ liệu
- Initialize sample weights (đều nhau ban đầu)
- Vòng lặp qua từng weak learner:
  - Train một Decision Tree nhỏ (max_depth=1, decision stump)
  - Tính error và weight của weak learner
  - Update sample weights (tăng weight cho samples bị sai)
- Combine tất cả weak learners
- Đánh giá trên test set
- Lưu tất cả weak learners và weights thành adaboost_bundle.pkl

**Thời gian chạy:** 10-30 phút tùy số estimators và máy tính

**Kết quả mong đợi:**
- Accuracy trên test set: khoảng 90-92%
- File output: adaboost_bundle.pkl (khoảng 30MB)

**Giải thích thuật toán:** AdaBoost là adaptive boosting - mỗi weak learner tập trung vào những samples mà learner trước đó sai. Em sử dụng decision stumps (trees với depth=1) làm weak learners. File .pkl lớn vì phải lưu tất cả weak learners (thường 50-100 cái).

**Lưu ý về memory:** Nếu Cô gặp lỗi MemoryError khi train trên local, em khuyến nghị dùng Google Colab vì cần khá nhiều RAM.

---

### Phần 3: Demo Applications

Phần này em xây dựng 2 ứng dụng web đơn giản để Cô có thể test models bằng cách vẽ trực tiếp.

#### 3.1. Demo AdaBoost

**Files cần có:**
- adaboost_bundle.pkl

**Notebook:**
- adaboost_demo.ipynb

**Cách thực hiện:**

Bước 1: Upload adaboost_bundle.pkl

Bước 2: Mở notebook app.ipynb

Bước 3: Run All

Bước 4: Đợi Gradio interface khởi động

Bước 5: Click vào link được generate (local hoặc public nếu dùng Colab)

**Sử dụng interface:**

1. **Canvas:** Vẽ chữ số 0-9 bằng chuột hoặc touchpad
   - Vẽ to và rõ ràng
   - Cố gắng vẽ ở giữa canvas

2. **Checkbox "Invert":**
   - Tích vào nếu Cô vẽ màu đen trên nền trắng
   - Không tích nếu vẽ màu trắng trên nền đen
   - Đây là để phù hợp với format MNIST (nền đen, chữ trắng)

3. **Threshold slider:**
   - Điều chỉnh threshold cho việc crop ảnh
   - Default là 200, Cô có thể thử điều chỉnh nếu kết quả không tốt

4. **Button "Predict":** Click để model dự đoán

**Output:**
- Dự đoán của model (số từ 0-9)
- Ảnh 28x28 sau khi preprocess
- Debug information: shape, min/max/mean/std của input

**Về preprocessing:** Em implement đầy đủ pipeline:
1. Invert nếu cần (để match với MNIST format)
2. Crop để loại bỏ whitespace
3. Center và pad thành vuông
4. Resize về 28x28
5. Normalize pixel values
6. Standardize theo mean và std của training set

**Lưu ý:** Model train trên chữ số kiểu MNIST nên có thể không hoàn toàn chính xác với mọi kiểu viết tay. Cô nên vẽ đơn giản, rõ ràng để model dễ nhận dạng hơn.

---

#### 3.2. Demo Random Forest

**Files cần có:**
- random_forest.pkl
- decision_tree.py (QUAN TRỌNG - demo cũng cần!)

**Notebook:**
- RF_DEMO.ipynb

**Cách thực hiện:**

Bước 1: Upload 2 files: random_forest.pkl và decision_tree.py

Bước 2: Mở notebook RF_DEMOipynb.ipynb

Bước 3: Run All (em khuyến nghị dùng Colab để có public link)

**Đặc điểm của demo này:**

Demo Random Forest có feature đặc biệt - nó test 2 phương pháp preprocessing khác nhau và cho Cô thấy kết quả của cả 2:

**Method 1: Raw pixels (0-255) WITH invert**
- Giữ nguyên pixel values ở scale 0-255
- Có invert ảnh để match MNIST format

**Method 2: Normalized (0-1) WITHOUT invert**
- Normalize pixel values về [0, 1]
- Không invert

**Mục đích:** Để minh họa rằng preprocessing ảnh hưởng rất nhiều đến kết quả. Hai methods này có thể cho ra predictions khác nhau cho cùng một chữ số.

**Output:**
Interface sẽ hiển thị kết quả của cả 2 methods và hỏi: "Which one is CORRECT? Tell me!"

Cô có thể tự đánh giá xem method nào cho kết quả tốt hơn. Điều này giúp understand tầm quan trọng của data preprocessing trong ML.

**So sánh 2 demos:**

| Tiêu chí | AdaBoost Demo | Random Forest Demo |
|----------|--------------|-------------------|
| Canvas type | Vẽ tự do | Brush với color picker |
| Preprocessing methods | 1 method | 2 methods để so sánh |
| Checkbox Invert | Có | Không (built-in trong 2 methods) |
| Output | 1 prediction | 2 predictions |
| Mục đích | Test model | Illustrate preprocessing importance |

---

## Xử lý các lỗi thường gặp

Em xin liệt kê một số lỗi Cô có thể gặp và cách khắc phục:

### Lỗi 1: "FileNotFoundError: processed_data.npz"

**Nguyên nhân:** File không có trong thư mục làm việc

**Cách khắc phục:**
1. Kiểm tra đã upload file processed_data.npz chưa
2. Đảm bảo file ở cùng thư mục với notebook
3. Check tên file có đúng không (case-sensitive, chú ý .npz không phải .zip)

### Lỗi 2: "ModuleNotFoundError: No module named 'decision_tree'"

**Nguyên nhân:** Thiếu file decision_tree.py (với Random Forest)

**Cách khắc phục:**
1. Upload file decision_tree.py vào cùng thư mục
2. Restart kernel trong Jupyter
3. Chạy lại từ đầu

### Lỗi 3: "KeyError: 'x_train'" hoặc "KeyError: 'W'"

**Nguyên nhân:** File .pkl hoặc .npz bị corrupt hoặc sai version

**Cách khắc phục:**
1. Download lại file từ source gốc
2. Check file size có đúng không
3. Nếu là .pkl file, thử train lại model

### Lỗi 4: "MemoryError"

**Nguyên nhân:** Không đủ RAM (thường xảy ra với AdaBoost)

**Cách khắc phục:**
1. Dùng Google Colab thay vì local (Colab có 12GB RAM free)
2. Giảm số estimators trong code
3. Giảm kích thước training data (sample một phần)

### Lỗi 5: Demo app không hiển thị kết quả

**Nguyên nhân:** Vẽ quá mờ hoặc preprocessing không phù hợp

**Cách khắc phục:**
1. Vẽ đậm và to hơn
2. Vẽ ở giữa canvas
3. Với AdaBoost demo: thử toggle checkbox "Invert"
4. Điều chỉnh threshold slider

### Lỗi 6: Gradio "Connection Error"

**Nguyên nhân:** Port bị block hoặc kernel crashed

**Cách khắc phục:**
1. Restart kernel
2. Chạy lại notebook
3. Check firewall/antivirus settings
4. Dùng Colab để có public link (recommended)

---

## Thứ tự chạy khuyến nghị

Em xin đề xuất thứ tự chạy để Cô có thể hiểu rõ flow của project:

### Cách 1: Đầy đủ từ đầu (nếu muốn reproduce toàn bộ)

1. **Train 3 Base Models** (có thể parallel)
   - Softmax.ipynb 
   - Bernoulli_Naive_Bayes.ipynb
   - Decision_Tree_Complete.ipynb
   
   Output: 3 files .pkl

2. **Train Hard Voting** (cần 3 models trên)
   - Hard_Voting.ipynb
   
   Output: hard_voting_model.pkl

3. **Train Random Forest** (độc lập, nhớ upload decision_tree.py)
   - 05_RandomForest_FN.ipynb
   
   Output: random_forest.pkl

4. **Train AdaBoost** (độc lập, train lâu nhất)
   - AdaBoost_final.ipynb
   
   Output: adaboost_bundle.pkl

5. **Test Demo Apps**
   - adaboost_demo.ipynb (với adaboost_bundle.pkl)
   - RF_DEMO.ipynb (với random_forest.pkl và decision_tree.py)

### Cách 2: Nhanh (nếu chỉ muốn xem demo và kết quả)

1. Download tất cả files .pkl đã train sẵn
2. Chạy trực tiếp các demo apps
3. Review kết quả trong notebooks (skip phần training)

---

## Kết quả đạt được

Em xin tóm tắt kết quả accuracy trên test set của các models:

| Model | Test Accuracy| Nhận xét |
|-------|--------------|-----------|
| Naive Bayes | ~83-84%  | Nhanh nhất, baseline tốt |
| Decision Tree | ~87-88% | Balance giữa speed và accuracy |
| Softmax | ~89-91% | Tốt cho linear separable classes |
| Hard Voting | ~90-91% | Cải thiện nhẹ so với base models |
| AdaBoost | ~88-90% | Tốt nhưng train lâu |
| Random Forest | ~94-96% | Accuracy cao nhất |

**Nhận xét:**
- Random Forest cho kết quả tốt nhất (~95%)
- Hard Voting là ensemble đơn giản nhưng hiệu quả
- AdaBoost train lâu nhưng kết quả tương đương Hard Voting
- Softmax tốt hơn expected cho linear model
- Naive Bayes nhanh nhất, phù hợp cho baseline

---

## Giải thích về các hyperparameters đã chọn

Em xin giải thích lý do chọn các hyperparameters:

**Decision Tree:**
- max_depth=10: Đủ sâu để học patterns phức tạp nhưng không quá sâu để overfit
- min_samples_split=2: Default value, works well với MNIST
- Gini impurity: Faster computation và tương đương accuracy với entropy

**Naive Bayes:**
- alpha=1.0 (Laplace smoothing): Tránh zero probabilities
- threshold=0.5: Binarization threshold cho Bernoulli NB

**Softmax:**
- learning_rate: Được tune trên validation set
- số epochs: Dừng khi validation loss không giảm nữa (early stopping)
- standardization: Giúp gradient descent hội tụ nhanh hơn

**AdaBoost:**
- n_estimators=50-100: Balance giữa accuracy và training time
- max_depth=1 (decision stumps): Weak learners đơn giản là đủ
- learning_rate=1.0: Default cho AdaBoost

**Random Forest:**
- n_estimators=100-200: Nhiều trees → stable predictions
- max_features='sqrt': Random feature selection giảm correlation giữa trees
- max_depth=None: Cho trees phát triển đầy đủ (bagging tự handle overfitting)

---

## Lưu ý khi chạy trên Google Colab

Nếu Cô chạy trên Colab, có một số lưu ý:

1. **Upload files:** Dùng file browser bên trái hoặc code:
```python
from google.colab import files
uploaded = files.upload()
```

2. **Download models sau khi train:**
```python
from google.colab import files
files.download('decision_tree_model.pkl')
```

3. **Public link cho demos:** Colab tự động tạo public link cho Gradio apps, rất tiện để share

4. **Session timeout:** Colab session có thể timeout sau 90 phút idle. Em khuyến nghị train xong thì download models về ngay.

5. **Restart runtime nếu gặp lỗi:** Runtime → Restart runtime

---

## Lời kết

Em xin tóm tắt lại:

1. **Files quan trọng nhất:** processed_data.npz và decision_tree.py
2. **Thứ tự train:** Base models → Hard Voting → Random Forest/AdaBoost
3. **Model tốt nhất:** Random Forest (~95% accuracy)
4. **Demo apps:** Để test interactive và hiểu về preprocessing
5. **Khuyến nghị:** Dùng Google Colab cho tiện

Tất cả notebooks đã được test kỹ và chạy thành công. Nếu Cô gặp bất kỳ vấn đề gì khi reproduce, em rất sẵn lòng hỗ trợ thêm.

Em cảm ơn Cô đã dành thời gian đọc hướng dẫn này. Em hy vọng mọi thứ được giải thích rõ ràng và dễ hiểu.

Trân trọng,
[NHÓM 11]

---

**Phụ lục: Quick Reference Table**

| Notebook | Files input | Files output |
|----------|------------|--------------|
| Decision_Tree_Complete | processed_data.npz | decision_tree_model.pkl | 
| Bernoulli_Naive_Bayes | processed_data.npz | naive_bayes_model.pkl | 
| Softmax | processed_data.npz | my_softmax.pkl | 
| Hard_Voting | processed_data.npz + 3 pkl | hard_voting_model.pkl |
| AdaBoost | processed_data.npz | adaboost_bundle.pkl | 
| RandomForest | processed_data.npz + decision_tree.py | random_forest.pkl |
| adaboost_demo | adaboost_bundle.pkl | None (demo) | 
| RF_DEMO | random_forest.pkl + decision_tree.py | None (demo) | 
---

*Hướng dẫn này được viết với tâm huyết và kiểm tra kỹ lưỡng*
*Mọi thắc mắc xin liên hệ qua email: 23133050@student.hcmute.edu.vn*
*Chúc Cô có trải nghiệm tốt khi chạy code!*
