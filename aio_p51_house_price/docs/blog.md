

<figure style="text-align: center;">
  <img src="/static/uploads/20251101_233849_ae57103e.png" alt="Project Overview Image" width="600">
  <figcaption><em>Hình 1. Best Model</em></figcaption>
</figure>

> [**Xem thêm Dashboard và các biểu đồ liên quan ở đây**](https://huggingface.co/spaces/hieutudo/projet51)
# 1/ Tổng quan dự án

Dự án này tập trung vào việc xây dựng một mô hình học máy để dự đoán giá nhà ở thành phố Ames, Iowa, dựa trên bộ dữ liệu chi tiết về các bất động sản được bán trong giai đoạn từ năm 2006 đến 2010. Đây là một bài toán hồi quy kinh điển, yêu cầu xử lý một lượng lớn các đặc trưng đa dạng, từ đó xây dựng một quy trình (pipeline) hoàn chỉnh bao gồm các bước: tiền xử lý dữ liệu, kỹ thuật đặc trưng (feature engineering), huấn luyện mô hình, và đánh giá hiệu suất. Mục tiêu cuối cùng là đưa ra dự đoán giá bán ($SalePrice$) cho các căn nhà chưa có thông tin về giá.

# 2/ Công nghệ & Công cụ

Để xây dựng quy trình này, dự án đã sử dụng một bộ công cụ (tech stack) hiện đại và phổ biến trong khoa học dữ liệu:

* **Xử lý Dữ liệu (Data Processing):**
    * **Pandas:** Thao tác và phân tích dữ liệu (data manipulation).
    * **NumPy:** Tính toán số học.
    * **Scikit-learn:** Tiền xử lý (preprocessing).

* **Kỹ thuật Đặc trưng (Feature Engineering):**
    * **Scikit-learn:** Lựa chọn đặc trưng (Mutual Information), biến đổi (Categorical encoding, MinMaxScaler).

* **Mô hình Học máy (ML Models):**
    * **Scikit-learn:** Các mô hình tuyến tính (Ridge, Lasso), GridSearchCV, Polynomial features.
    * **XGBoost:** Mô hình tree-based (Gradient Boosting).

* **Tự động hóa & Triển khai (Automation & Deployment):**
    * **Makefile:** Điều phối quy trình (pipeline orchestration).
    * **Python Modules:** Cấu trúc code mô-đun (src/, clean code).
    * **Streamlit:** Caching và lưu trữ (persistence).
    * **config.py:** Quản lý các tham số có thể thay đổi.

* **Trực quan hóa & Dashboard (Visualization & Dashboard):**
    * **Streamlit:** Xây dựng web dashboard tương tác.
    * **Plotly:** Biểu đồ tương tác.
    * **Matplotlib:** Trực quan hóa cơ bản.
    * **SHAP:** Giải thích và diễn giải mô hình (model interpretability).

# 3/ Các Phương pháp Tốt nhất (Best Practices)

Dự án được triển khai theo các tiêu chuẩn tốt nhất trong ngành để đảm bảo tính chính xác, hiệu quả và khả năng tái sản xuất:

* **Phòng chống Rò rỉ Dữ liệu (Data Leakage Prevention):**
    * Bộ tiền xử lý (Preprocessor) chỉ `fit` (học) trên dữ liệu huấn luyện (train data).
    * Bộ biến đổi (Transformer) chỉ `fit` trên dữ liệu huấn luyện.
    * Áp dụng *cùng một* phép biến đổi cho cả tập validation và tập test.
    * Mã hóa (Encoding) được thực hiện trước khi co giãn (scaling) để tránh rò rỉ.

* **Xác thực chéo (Cross-Validation) đúng đắn:**
    * Sử dụng 5-fold Cross-Validation cho tất cả các mô hình `GridSearchCV`.
    * Phân chia tập dữ liệu theo tỷ lệ Train/Validation là 75%/25%.
    * Sử dụng cơ chế `Early Stopping` cho mô hình XGBoost để tránh overfitting.

* **Tinh chỉnh Siêu tham số (Hyperparameter Tuning):**
    * `GridSearchCV`: Tìm kiếm có hệ thống trên lưới tham số (systematic grid search).
    * `RandomizedSearchCV`: Tìm kiếm ngẫu nhiên (random sampling).
    * Sử dụng các chỉ số trên tập validation để lựa chọn mô hình tốt nhất.

* **Đảm bảo Khả năng Tái sản xuất (Reproducibility):**
    * Cố định `random_state` trong tất cả các bước (chia dữ liệu, mô hình) để đảm bảo kết quả nhất quán.
    * Sử dụng seed cho cả quá trình chia (split) và xác thực chéo (CV).
    * Lưu trữ tất cả các siêu tham số trong file `results.csv`.

* **Thiết kế Mô-đun (Modular Design):**
    * Tách biệt các thành phần: dữ liệu (data), đặc trưng (features), mô hình (models).
    * Sử dụng các thành phần có thể tái sử dụng (classes, functions).
    * Giúp dễ dàng mở rộng và gỡ lỗi (debug).

* **Quy trình Tự động hóa (Automated Pipeline):**
    * Sử dụng `Makefile` để điều phối các dependencies (ví dụ: `make all`, `make data`).
    * Không cần thực thi thủ công từng bước.
    * Sử dụng `Caching` (bộ nhớ đệm) để tránh tính toán lại các bước không cần thiết.

# 4/ Tổng quan dữ liệu

<figure style="text-align: center;">
  <img src="/static/uploads/20251101_234651_c8ed6796.png" alt="Data Overview" width="600">
  <figcaption><em>Hình 2. Data Overview</em></figcaption>
</figure>

Bộ dữ liệu gốc bao gồm 1460 mẫu (tương ứng với 1460 căn nhà đã được bán) và 81 cột (đặc trưng) mô tả các thuộc tính khác nhau của mỗi căn nhà. Biến mục tiêu cần dự đoán là $$SalePrice$$.

Các đặc trưng có thể được phân loại vào các nhóm chính sau:

* **Vị trí:** Các đặc trưng như `Neighborhood` (khu vực lân cận), `MSZoning` (phân loại quy hoạch chung), và `Street` (loại đường vào) cung cấp thông tin về vị trí địa lý của bất động sản.
* **Lô đất:** Các đặc trưng như `LotArea` (diện tích lô đất), `LotFrontage` (chiều dài mặt tiền tiếp xúc với đường), và `LotShape` (hình dạng lô đất) mô tả về khu đất.
* **Kết cấu:** Các đặc trưng như `HouseStyle` (kiểu nhà), `BldgType` (loại nhà), `YearBuilt` (năm xây dựng), và `YearRemodAdd` (năm tu sửa) cho biết thông tin về cấu trúc và tuổi đời của căn nhà.
* **Chất lượng và Tình trạng:** Các đặc trưng như `OverallQual` (chất lượng vật liệu và hoàn thiện tổng thể), `OverallCond` (tình trạng tổng thể), `ExterQual` (chất lượng ngoại thất), và `ExterCond` (tình trạng ngoại thất) là các thang đo định tính về chất lượng của căn nhà.
* **Tầng hầm:** Các đặc trưng như `BsmtQual` (chất lượng tầng hầm), `BsmtCond` (tình trạng tầng hầm), và `TotalBsmtSF` (tổng diện tích tầng hầm) mô tả chi tiết về tầng hầm.
* **Phòng:** Các đặc trưng như `BedroomAbvGr` (số phòng ngủ trên mặt đất), `KitchenAbvGr` (số phòng bếp trên mặt đất), và `TotRmsAbvGrd` (tổng số phòng trên mặt đất) cho biết về số lượng và loại phòng.
* **Garage:** Các đặc trưng như `GarageType` (loại garage), `GarageCars` (sức chứa của garage tính bằng số lượng xe), và `GarageArea` (diện tích garage) cung cấp thông tin về khu vực để xe.
* **Diện tích:** Các đặc trưng quan trọng như `GrLivArea` (diện tích sinh hoạt trên mặt đất), `TotalBsmtSF` (tổng diện tích tầng hầm), `1stFlrSF` (diện tích tầng một), và `2ndFlrSF` (diện tích tầng hai) là các biến số liên tục mô tả quy mô của căn nhà.
* **Thông tin bán hàng:** Các đặc trưng như `YrSold` (năm bán), `MoSold` (tháng bán), `SaleType` (loại hình mua bán), và `SaleCondition` (điều kiện bán) cung cấp ngữ cảnh về giao dịch.

# 5/ Kỹ thuật đặc trưng (Feature Engineering)

Đây là bước quan trọng nhằm biến đổi dữ liệu thô thành các đặc trưng có ý nghĩa hơn, giúp mô hình học máy có thể "hiểu" và khai thác thông tin hiệu quả hơn. Quá trình này kết hợp cả kiến thức chuyên môn (domain knowledge) và các kỹ thuật tự động.

* **Lựa chọn đặc trưng dựa trên Mutual Information (MI):** Mutual Information đo lường mức độ phụ thuộc giữa mỗi đặc trưng và biến mục tiêu $$SalePrice$$. Những đặc trưng có điểm MI gần bằng 0 được coi là không cung cấp thông tin hữu ích cho việc dự đoán và sẽ bị loại bỏ. Điều này giúp giảm độ phức tạp của mô hình và tránh nhiễu.
* **Biến đổi toán học (Tạo tỷ lệ & tương tác):** Tạo ra các đặc trưng mới bằng cách kết hợp các đặc trưng hiện có, dựa trên kiến thức miền (domain knowledge). Việc này giúp mô hình nắm bắt được các mối quan hệ phức tạp hơn.
    * **LivLotRatio:** Tỷ lệ giữa diện tích sinh hoạt và diện tích lô đất. Đặc trưng này có thể phản ánh mức độ "rộng rãi" của căn nhà so với khu đất.
        $$LivLotRatio = \frac{\text{GrLivArea}}{\text{LotArea}}$$
    * **Spaciousness:** Diện tích sinh hoạt trung bình trên mỗi phòng. Đặc trưng này cho biết mức độ "thoáng đãng" của không gian sống.
        $$Spaciousness = \frac{\text{1stFlrSF} + \text{2ndFlrSF}}{\text{TotRmsAbvGrd}}$$
* **Đặc trưng tương tác (Interactions):** Tạo ra các đặc trưng tương tác bằng cách kết hợp các đặc trưng phân loại và đặc trưng số. Ví dụ, đặc trưng `BldgType` (loại nhà) được mã hóa one-hot, sau đó nhân với `GrLivArea` (diện tích sinh hoạt). Điều này giúp mô hình hiểu được rằng ảnh hưởng của diện tích lên giá nhà có thể khác nhau tùy thuộc vào loại nhà.
* **Đặc trưng đếm (Counts):** Tạo đặc trưng `PorchTypes` bằng cách đếm số lượng các loại hiên nhà khác nhau có diện tích lớn hơn 0 (bao gồm `WoodDeckSF`, `OpenPorchSF`, `EnclosedPorch`, `ScreenPorch`). Một căn nhà có nhiều loại hiên có thể được đánh giá cao hơn.
* **Tổng hợp theo nhóm (Group Aggregations):** Tính toán các thống kê theo nhóm. Ví dụ, đặc trưng `MedNhbdArea` được tạo ra bằng cách tính giá trị trung vị (median) của `GrLivArea` cho mỗi khu vực (`Neighborhood`). Đặc trưng này giúp mô hình so sánh một căn nhà với các căn nhà khác trong cùng một khu vực.
* **Mã hóa mục tiêu (Target Encoding):** Đặc trưng `MSSubClass` (phân loại loại nhà) được mã hóa bằng phương pháp M-Estimate Encoder. Đây là một kỹ thuật mã hóa mục tiêu (target encoding) được thực hiện với xác thực chéo 5 lần (5-fold cross-validation) để **tránh rò rỉ dữ liệu (data leakage)**.
* **Co giãn (Scaling):** Tất cả các đặc trưng số cuối cùng được co giãn về khoảng [0, 1] bằng `MinMaxScaler`. Việc này đảm bảo rằng các đặc trưng có thang đo khác nhau sẽ có đóng góp tương đương trong quá trình huấn luyện các mô hình nhạy cảm với thang đo (như Ridge, Lasso).

# 6/ Mô hình (Models)

Dự án huấn luyện và đánh giá một loạt các mô hình hồi quy để tìm ra mô hình dự đoán $SalePrice$ tốt nhất.

### Các mô hình Tiêu chuẩn (Standard Models)

Tổng cộng 5 mô hình tiêu chuẩn đã được huấn luyện, bao gồm các mô hình tuyến tính cơ sở, các mô hình được điều chuẩn (regularized) và các mô hình tree-based.

| Mô hình | Loại | Tinh chỉnh | CV | Tham số |
| :--- | :--- | :--- | :--- | :--- |
| **linear** | Linear Regression | None | No | None |
| **ridge** | Ridge L2 Regularization | GridSearchCV | 5-fold | 50 alphas |
| **lasso** | Lasso L1 Selection | GridSearchCV | 5-fold | 50 alphas |
| **xgboost_fast** | Gradient Boosting | Pre-tuned | Early stop | n_est=1000, lr=0.03 |
| **xgboost_random** | Gradient Boosting | RandomSearch | 2-fold | 20 iterations |

<p align="center"><em>*Bảng 1. Các mô hình tiêu chuẩn*</em></p>

* **Chi tiết tinh chỉnh:**
    * Các mô hình `ridge` và `lasso` sử dụng `GridSearchCV` với 5-fold CV để tìm ra tham số `alpha` (hệ số điều chuẩn) tối ưu từ 50 giá trị khác nhau (tổng cộng $50 \times 5 = 250$ lượt huấn luyện cho mỗi mô hình).
    * Mô hình `xgboost_random` sử dụng `RandomizedSearchCV` với 20 lượt thử (iterations) để tìm kiếm không gian siêu tham số hiệu quả.

### Các mô hình Đa thức (Polynomial Models)

Để nắm bắt các mối quan hệ phi tuyến, 3 mô hình tuyến tính đã được huấn luyện lại trên bộ đặc trưng được mở rộng với các tương tác đa thức bậc hai.

<div align="center">

| Mô hình | Đặc trưng | Tinh chỉnh | CV | Bậc |
| :--- | :--- | :--- | :--- | :--- |
| **linear_poly** | Degree-2 interactions | None | No | 2 |
| **ridge_poly** | Polynomial + Ridge | GridSearchCV | 5-fold | 2 |
| **lasso_poly** | Polynomial + Lasso | GridSearchCV | 5-fold | 2 |

</div>
<p align="center"><em>*Bảng 2. Các mô hình đa thức.*</em></p>

### Phương pháp Ensemble

Để tăng cường độ chính xác và sự ổn định (robustness) của dự đoán, các kết quả từ nhiều mô hình riêng lẻ được kết hợp lại.

* **Kết hợp các mô hình đa dạng:** Phương pháp này kết hợp các dự đoán từ cả mô hình tuyến tính (linear) và mô hình tree-based (XGBoost) để tận dụng điểm mạnh của từng loại.
* **Cải thiện khả năng tổng quát hóa:** Bằng cách lấy trung bình dự đoán, mô hình ensemble giúp giảm phương sai (variance) và cải thiện khả năng tổng quát hóa (generalization) trên dữ liệu mới.
* **3 phương pháp được sử dụng:**
    1.  **Trung bình (Mean):** Lấy trung bình cộng của các dự đoán.
    2.  **Trung vị (Median):** Lấy giá trị trung vị của các dự đoán (ít nhạy cảm với các giá trị ngoại lệ).
    3.  **Trung bình có trọng số (Weighted Mean):** Lấy trung bình cộng có trọng số, ưu tiên các mô hình có hiệu suất tốt hơn.

Các mô hình được đánh giá dựa trên các chỉ số **R-squared** (hệ số xác định), **Root Mean Squared Error (RMSE)** (sai số toàn phương trung bình), và **Mean Absolute Error (MAE)** (sai số tuyệt đối trung bình) trên tập dữ liệu kiểm định (validation set). Mô hình hoạt động tốt nhất sẽ được chọn để đưa ra dự đoán cuối cùng trên tập dữ liệu thử nghiệm (test set).

# 7/ Kết quả đánh giá

## 7.1/ So sánh giữa các mô hình

| Model           | Train_R2 | Val_R2 | Train_RMSE | Val_RMSE | Train_MAE | Val_MAE | Time  |
|-----------------|-----------|--------|-------------|-----------|------------|----------|-------|
| xgboost_fast    | 0.9731    | 0.9165 | 12,778      | 24,184    | 8,366      | 15,165   | 0.9s  |
| xgboost_random  | 0.9643    | 0.9155 | 14,712      | 24,332    | 9,505      | 15,278   | 15.0s |
| ridge_poly      | 0.9301    | 0.9077 | 20,597      | 25,425    | 12,443     | 15,592   | 17.2s |
| ridge           | 0.8677    | 0.8858 | 28,345      | 28,290    | 15,871     | 16,709   | 2.2s  |
| linear          | 0.8714    | 0.8771 | 27,939      | 29,339    | 15,552     | 17,164   | 0.0s  |
| lasso           | 0.8675    | 0.8731 | 28,367      | 29,816    | 15,833     | 17,040   | 1.8s  |
| lasso_poly      | 0.8632    | 0.8604 | 28,820      | 31,268    | 16,812     | 17,873   | 27.6s |
| linear_poly     | 1.0000    | 0.7763 | 0           | 39,586    | 0          | 26,709   | 0.7s  |

<p align="center"><em>Bảng 3. So sánh kết quả các mô hình hồi quy (Regression Models) và thời gian huấn luyện.</em></p>

## 7.2/ Metrics Comparison

<figure style="text-align: center;">
  <img src="/static/uploads/20251101_234029_1fe6ba44.png" alt="Metrics Comparison" width="600">
  <figcaption><em>Hình 3. Metrics Comparison</em></figcaption>
</figure>

## 7.3/ Overfitting Analysis

<figure style="text-align: center;">
  <img src="/static/uploads/20251101_234833_fe103e2f.png" alt="Overfitting Analysis" width="600">
  <figcaption><em>Hình 4. Overfitting Analysis</em></figcaption>
</figure>

## 7.4 MAE Comparison

<figure style="text-align: center;">
  <img src="/static/uploads/20251101_235023_2a487b58.png" alt="MAE Comparison" width="600">
  <figcaption><em>Hình 5. MAE Comparison</em></figcaption>
</figure>

# 8/ Kết luận

Dự án xây dựng mô hình dự đoán giá nhà tại Ames, Iowa, đã thành công trong việc áp dụng các kỹ thuật học máy hiện đại để xử lý bộ dữ liệu phức tạp với 81 đặc trưng, đạt được hiệu suất cao nhất từ mô hình XGBoost_fast với R² validation đạt 0.9165 và RMSE 24,184, chứng tỏ khả năng tổng quát hóa tốt trên dữ liệu mới. Việc áp dụng kỹ thuật đặc trưng như Mutual Information, tạo tương tác và target encoding, kết hợp với các best practices như phòng chống data leakage và cross-validation, không chỉ nâng cao độ chính xác mà còn đảm bảo tính tái sản xuất và mô-đun hóa của pipeline. Trong tương lai, có thể mở rộng bằng cách tích hợp dữ liệu thời gian thực hoặc thử nghiệm các mô hình deep learning để cải thiện thêm dự đoán, mang lại giá trị thực tiễn cho lĩnh vực bất động sản.

> [Source code tham khảo](https://colab.research.google.com/drive/1cvGy77tMEMcjlFU1aDMMJfhZsfPHPX28)

