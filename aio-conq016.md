# 1. Introduction - Tại sao chúng ta cần GA và SHAP?

![Mutation](./images/thumb.png)

Ngày nay, AI gần như đã được đưa vào ứng dụng trong hầu hết các ngành nghề, từ y tế, sinh học đến kỹ thuật sản xuất công nghiệp nặng như ôtô, hàng không, các ngành năng lượng, điện lực, sản xuất thông minh, tối ưu hóa quy trình, cũng như các lĩnh vực tài chính kinh tế. Trong số các ứng dụng AI được biết đến rộng rãi hiện nay có thể kể đến các mô hình ngôn ngữ lớn như ChatGPT, Gemini hay các mô hình tạo sinh ảnh, video như Sora, DALL-E, Stable Diffusion… Đặc điểm của các mô hình này là cách huấn luyện cho mô hình tự học, tự động cập nhật các trọng số (parameters) và tịnh tiến đến điểm tối ưu bằng sử dụng các **thuật toán tối ưu hóa dựa trên Gradient**. Vì các trọng số là do mô hình tự học từ quá trình huấn luyện với dữ liệu, nên không phải lúc nào chúng ta cũng có thể hiểu được tại sao mô hình lại đưa ra câu trả lời như vậy, có thể nói, lựa chọn của mô hình là một **hộp đen** đối với chúng ta. Hơn nữa, trong các bài toán thực tế có độ phức tạp cao, không phải lúc nào chúng ta cũng có thể dùng Gradient Descent để tìm ra điểm tối ưu.

## 1.1. Bài toán tối ưu trong thực tế

Lấy một ví dụ về bài toán tối ưu trong thực tế: tối ưu thiết kế xe đua là một bài toán **cực kỳ phức tạp**, không chỉ liên quan đến công thức toán học mà còn liên quan đến vật lý, khí động học, khoa học vật liệu và **hàng trăm biến đầu vào phi tuyến** khác.

### Bản chất của bài toán thiết kế xe đua:

Khi tối ưu thiết kế xe đua (ví dụ F1), ta có rất nhiều yếu tố:
- Hình dạng khung xe (đường cong, góc nghiêng, profile khí động học).
- Kích thước các bộ phận (cánh gió, gầm, hốc gió,...)
- Vật liệu cấu trúc
- Cấu hình động cơ, hộp số, hệ thống truyền động
Thông số điều khiển (aero setup, suspension, tire pressure, …)

Như vậy, ta có thể thấy là bài toán tối ưu thiết kế xe đua là kết quả của một mô phỏng vật lý phức tạp, chứ không phải là một hàm **khả vi** đơn giản, và do đó phương án dùng Gradient Descent để tìm tối ưu là không khả thi. 
Trong thực tế, các hãng như Ferrari, Mercedes-AMG đều dùng **AI + GA + CFD** mô phỏng để tối ưu thiết kế khung xe, cánh gió… Ở đây, **GA** (**Genetic Algorithm**) là một thuật toán thay thế cho Gradient Descent để tìm tối ưu trong các bài toán phức tạp.


**Xem thêm:** Chúng ta có thể xem thêm một video về thiết kế mô phỏng sử dụng Genetic Algorithm ở link: [AI Learns to be a Car using a Genetic Algorithm](https://www.youtube.com/watch?v=Ei2g8XoCkdg)


## 1.2. XAI trong bài toán tối ưu thực tiễn

Như đã đề cập bên trên, một trong những vấn đề cốt lõi của AI là **sự thiếu minh bạch (Black-box)** và **tính tin cậy**.

### Tính tin cậy
Khi mô hình AI + CFD + GA đưa ra một thiết kế mà nó cho là tối ưu mới, các kỹ sư cần biết rằng liệu thiết kế đó có thực sự **an toàn** và **hiệu quả** hay không, hay đó chỉ là một kết quả ngẫu nhiên?

### Sự minh bạch

Trong thiết kế xe đua, vấn đề không chỉ là đạt hiệu suất cao nhất mà còn phải quản lý chi phí sản xuất, chi phí vật liệu và tuân thủ các quy định. Nói cách khác, bài toán thiết kế xe đua tối ưu còn phải thỏa mãn **Cân bằng Chi phí và Hiệu Suất** cũng như **các quy định**.

Bởi vậy, việc dùng XAI để giải thích được yếu tố nào đóng góp vào thiết kế tối ưu bao nhiêu phần trăm đóng một vai trò cực kỳ quan trọng. Vi dụ: Nếu mô hình dự đoán rằng một vật liệu siêu đắt tiền chỉ cải thiện hiệu suất 0.5% nhưng một thay đổi nhỏ về hình học gầm xe lại cải thiện 5%, XAI giúp nhóm thiết kế đưa ra lựa chọn tối ưu phần nào với chi phí giới hạn (chọn giải pháp hình học rẻ hơn, hiệu quả hơn).

Bởi vậy ngoài thuật toán di truyền GA, thì còn cần kết hợp thêm thuật toán SHAP để hiểu rõ yếu tố nào đóng góp bao nhiêu phần trăm vào một bài toán tối ưu. Một hệ thống gồm **AI + GA +CFD + SHAP** có thể giảm thời gian thiết kế từ hàng tháng xuống còn vài ngày và cũng giúp tiết kiệm hàng triệu đô chi phí mô phỏng. Trong các phần tiếp theo của blog, chúng ta sẽ cùng tìm hiểu kỹ hơn về GA, SHAP cũng như các biến thể của nó.

# 2. Giới thiệu Genetic Algorithm (GA)

## 2.1. Genetic Algorithm là gì?

**Genetic Algorithm (GA)** là thuật toán tối ưu hóa dựa trên nguyên lý **chọn lọc tự nhiên** và **di truyền học** của Darwin. GA mô phỏng quá trình tiến hóa để tìm nghiệm tối ưu cho các bài toán phức tạp.

### Ý tưởng:

- Cá thể **mạnh** (nghiệm tốt) có khả năng **sống sót** và **sinh sản** cao hơn
- Thông qua **lai ghép** và **đột biến**, thế hệ sau tốt hơn thế hệ trước
- Sau nhiều thế hệ, quần thể tiến hóa về nghiệm tối ưu

### Tại sao cần Genetic Algorithm?

Trong các bài toán tối ưu cổ điển, khi biết rõ hàm mục tiêu và có thể tính đạo hàm, **Gradient Descent (GD)** là lựa chọn tự nhiên. Tuy nhiên, thực tế thường phức tạp hơn:

- Hàm mục tiêu **không khả vi** (discrete, có bước nhảy)
- Dễ bị **kẹt ở cực trị cục bộ** (local optima)
- Không gian tìm kiếm **phức tạp**, nhiều chiều

**Genetic Algorithm (GA)** ra đời để giải quyết những hạn chế này.

### So sánh GD vs GA:

| Tiêu chí                        | Gradient Descent         | Genetic Algorithm             |
| ------------------------------- | ------------------------ | ----------------------------- |
| **Cách tiếp cận**               | Đi theo 1 quỹ đạo đơn lẻ | Duy trì cả quần thể nghiệm    |
| **Yêu cầu**                     | Cần đạo hàm              | Không cần đạo hàm             |
| **Khả năng thoát local optima** | Thấp                     | Cao (nhờ đột biến và đa dạng) |
| **Tốc độ hội tụ**               | Nhanh                    | Chậm hơn                      |
| **Phù hợp**                     | Hàm liên tục, khả vi     | Hàm rời rạc, phức tạp         |

## 2.2. Quy trình hoạt động của GA

Quy trình của GA có thể được mô tả qua các bước sau:

### **Bước 1: Khởi tạo quần thể (Initialization)**

Tạo ngẫu nhiên **một tập hợp các cá thể (individuals)** — mỗi cá thể biểu diễn một **lời giải tiềm năng** cho bài toán.  
Ví dụ: nếu bài toán cần tìm bộ tham số tối ưu, thì mỗi cá thể có thể là một vector giá trị `[x₁, x₂, ..., xn]`.

### **Bước 2: Đánh giá fitness quần thể (Fitness Evaluation)**

Sử dụng **hàm mục tiêu (fitness function)** để đánh giá “độ tốt” của từng cá thể.

- Cá thể có điểm fitness cao thể hiện lời giải gần tối ưu hơn.
- Hàm fitness được thiết kế tùy thuộc vào bài toán cụ thể (ví dụ: lợi nhuận, sai số, thời gian, độ chính xác...).

### **Bước 3: Giữ lại cá thể ưu tú (Elitism)**

Một số cá thể tốt nhất (elite) được **giữ nguyên** sang thế hệ kế tiếp để đảm bảo chất lượng không giảm sút.

### **Bước 4: Chọn lọc bố mẹ (Selection)**

Chọn các cá thể từ quần thể hiện tại để làm **bố mẹ** dựa trên điểm fitness.  
Các phương pháp chọn lọc phổ biến:

- **Roulette Wheel Selection** (xác suất tỉ lệ với fitness)
- **Tournament Selection** (chọn cá thể tốt nhất trong nhóm ngẫu nhiên)

### **Bước 5: Lai ghép (Crossover)**

Hai cá thể bố mẹ **kết hợp gen** để tạo ra **các cá thể con**.

- Mục tiêu: khai thác thông tin di truyền tốt từ bố mẹ.
- Ví dụ:
  - Cha: [1, 0, 1, 1]
  - Mẹ: [0, 1, 0, 0]
  - Con: [1, 0, 0, 0]

### **Bước 6: Đột biến (Mutation)**

Một số gen trong cá thể con được **thay đổi ngẫu nhiên** để tăng **đa dạng di truyền** và giúp tránh rơi vào **cực trị cục bộ**.  
Ví dụ: thay đổi 1 bit trong chuỗi gen `[1, 0, 0, 0] → [1, 1, 0, 0]`.

### **Bước 7: Cập nhật quần thể mới (New Generation)**

Sau khi lai ghép và đột biến, ta có một **thế hệ mới** gồm:

- Cá thể ưu tú được giữ lại
- Các cá thể con mới được sinh ra  
  → Đây chính là quần thể sẽ được đánh giá trong vòng lặp tiếp theo.

### **Bước 8: Kiểm tra điều kiện dừng (Termination Condition)**

Thuật toán sẽ dừng khi:

- Đạt số thế hệ tối đa, **hoặc**
- Độ cải thiện fitness không đáng kể, **hoặc**
- Đã tìm thấy lời giải đủ tốt (thoả điều kiện mục tiêu).

Nếu **chưa đạt**, quay lại **Bước 2** để lặp lại tiến trình.

### **Bước 9: Kết thúc**

Trả về **cá thể có fitness cao nhất** — đó chính là **lời giải tối ưu (hoặc gần tối ưu)** mà GA tìm được.

## 2.3. Ưu và nhược điểm

### Ưu điểm:

- Không cần gradient (phù hợp với hàm rời rạc, không khả vi)  
- Tìm kiếm global optimum tốt hơn các phương pháp greedy  
- Linh hoạt, áp dụng được cho nhiều bài toán  
- Dễ song song hóa (đánh giá fitness độc lập)

### Nhược điểm:

- Tốn thời gian tính toán (nhiều thế hệ)  
- Cần chọn hyperparameter (kích thước quần thể, mutation rate, ...)  
- Không đảm bảo tìm được nghiệm tối ưu tuyệt đối  
- Khó điều chỉnh cho bài toán cụ thể

## 2.4. Ứng dụng của GA trong Machine Learning

| Ứng dụng                       | Mô tả                                           |
| ------------------------------ | ----------------------------------------------- |
| **Feature Selection**          | Chọn tập con đặc trưng tối ưu                   |
| **Hyperparameter Tuning**      | Tối ưu learning rate, batch size, số layers,... |
| **Neural Architecture Search** | Tìm kiếm cấu trúc mạng neural tối ưu            |
| **Ensemble Learning**          | Chọn và kết hợp các model cơ sở                 |
| **Data Augmentation**          | Tìm chiến lược augmentation tốt nhất            |

## 2.5. So sánh GA với các phương pháp khác

| Phương pháp               | Ưu điểm               | Nhược điểm                 | Khi nào dùng                  |
| ------------------------- | --------------------- | -------------------------- | ----------------------------- |
| **Grid Search**           | Đơn giản, toàn diện   | Rất chậm với nhiều tham số | Không gian nhỏ (< 5 tham số)  |
| **Random Search**         | Nhanh hơn Grid        | Không tận dụng thông tin   | Không gian vừa (5-10 tham số) |
| **Bayesian Optimization** | Hiệu quả, ít đánh giá | Phức tạp, khó cài đặt      | Hàm đắt để tính toán          |
| **Genetic Algorithm**     | Linh hoạt, tìm global | Nhiều hyperparameter       | Không gian lớn, phức tạp      |

## 2.6. Tham số quan trọng cần điều chỉnh

| Tham số                | Giá trị thường dùng | Ảnh hưởng                          |
| ---------------------- | ------------------- | ---------------------------------- |
| **Population size**    | 50-200              | Lớn → đa dạng nhưng chậm           |
| **Crossover rate**     | 0.6-0.9             | Cao → khai thác nghiệm tốt         |
| **Mutation rate**      | 0.01-0.1            | Cao → khám phá, tránh local optima |
| **Generations**        | 50-500              | Nhiều → chính xác nhưng chậm       |
| **Selection pressure** | Tournament size 3-5 | Cao → hội tụ nhanh                 |

## 2.7. Tóm tắt

- **GA** mô phỏng tiến hóa tự nhiên để tối ưu hóa
- **5 bước chính**: Khởi tạo → Đánh giá → Chọn lọc → Lai ghép → Đột biến
- **Phù hợp** với bài toán không gian lớn, phức tạp, không khả vi
- **Ứng dụng** nhiều trong ML: feature selection, hyperparameter tuning,...
- **Lưu ý**: Cần điều chỉnh tham số phù hợp với từng bài toán cụ thể

# 3. Giới thiệu SHAP (dựa trên Shapley Values)

## 3.1. Khi AI trở nên “bí ẩn” – vấn đề về khả năng giải thích  

Hãy tưởng tượng một buổi sáng, bạn mở ứng dụng ngân hàng và thấy dòng thông báo:  

> “Yêu cầu vay tín dụng của bạn bị từ chối.”  

Không có lời giải thích. Không biết vì sao.  
Dữ liệu bạn cung cấp rất ổn: thu nhập tốt, công việc ổn định, không nợ xấu.  
Nhưng hệ thống AI chỉ trả về một câu trả lời dứt khoát: **“Không.”**

Đây chính là một ví dụ điển hình của **vấn đề thiếu khả năng giải thích (Explainability)** trong các hệ thống **trí tuệ nhân tạo hiện đại**.  
Những mô hình phức tạp như **XGBoost, Random Forest, hay Deep Neural Network** có thể đạt độ chính xác cao,  
nhưng lại hoạt động như **“hộp đen” (black box)** — ta chỉ biết đầu vào và đầu ra, mà không thể nhìn thấy bên trong.  

Thực ra, điều này **không xa lạ với chính con người**.  
**Bộ não** của chúng ta cũng là một **“hộp đen tối thượng.”**  
Các nhà khoa học thần kinh có thể nói:  
> “Vỏ não trước trán hoạt động khi ta lập kế hoạch,”  
> hay “dopamine liên quan đến cảm giác phần thưởng.”  

Nhưng họ vẫn **chưa thể giải thích chính xác** cách hàng tỷ neuron trong não kết nối và tương tác để đưa ra **một quyết định cụ thể**.  

Tương tự, **AI cũng vậy.**  
Chúng ta có thể biết rằng “feature A tăng → xác suất B cao hơn,”  
nhưng không dễ để hiểu **vì sao mô hình lại chọn hướng đó**, hay **mỗi đặc trưng ảnh hưởng đến kết quả bao nhiêu**.  

Vì thế, cũng như **khoa học thần kinh đang cố “giải mã não bộ”**,  
**Explainable AI (XAI)** chính là nỗ lực “giải mã tư duy của máy tính.”  
Nó giúp con người hiểu được cách mà các mô hình học máy **ra quyết định, cân nhắc, và phản ứng** trước dữ liệu.  

Và trong hành trình này, **SHAP (SHapley Additive exPlanations)**  
là một trong những phương pháp quan trọng nhất,  
vì nó không chỉ nói “mô hình hoạt động thế nào”, mà còn **định lượng được từng phần đóng góp** của mỗi đặc trưng —  
tương tự như việc tìm hiểu từng neuron trong não ảnh hưởng thế nào đến một hành động cụ thể.

---

## 3.2. Từ lý thuyết trò chơi đến máy học – câu chuyện của SHAP  

SHAP dựa trên một ý tưởng xuất phát từ **lý thuyết trò chơi hợp tác (Cooperative Game Theory)** do nhà toán học **Lloyd Shapley** đề xuất năm 1953.  

Giả sử bạn có một đội bóng gồm ba cầu thủ: An, Bình, và Chi.  
Cả đội thắng và nhận được 100 điểm thưởng. Nhưng ai đóng góp bao nhiêu?  
Nếu chỉ chia đều, có thể không công bằng – vì người ghi bàn hay người giữ khung thành đều ảnh hưởng khác nhau đến kết quả cuối cùng.  

Để chia thưởng hợp lý, Shapley đưa ra cách tiếp cận:  
- Thử loại từng người khỏi đội và xem **đội còn lại chơi tốt đến đâu**.  
- Tính **mức chênh lệch trong thành tích** khi người đó tham gia hoặc không tham gia.  
- Lấy trung bình trên **tất cả các cách kết hợp có thể**.  

Kết quả là, ta biết được **đóng góp công bằng của từng người chơi**.  

Trong **trí tuệ nhân tạo**, “trò chơi” chính là mô hình học máy,  
và “người chơi” chính là **các đặc trưng (features)** trong tập dữ liệu.  
SHAP đóng vai trò là “trọng tài” tính **độ đóng góp công bằng của từng feature** vào kết quả dự đoán.

---

## 3.3. Nguyên lý hoạt động của SHAP – chia phần công bằng cho từng đặc trưng  

### (1) Ý tưởng cơ bản  

Hãy xem mô hình học máy như một trò chơi hợp tác giữa các đặc trưng dữ liệu.  
Mỗi đặc trưng đóng góp một phần vào “phần thưởng cuối cùng” — tức là **kết quả dự đoán**.

Để đo đóng góp của một đặc trưng i, SHAP xem xét:
- Nếu **đặc trưng đó có mặt**, dự đoán thay đổi như thế nào?  
- Nếu **đặc trưng đó bị loại bỏ**, mô hình hoạt động ra sao?  

Bằng cách tính **mức thay đổi trung bình** của kết quả qua **mọi tổ hợp đặc trưng có thể có**,  
SHAP xác định **đóng góp thực sự** của từng đặc trưng đối với quyết định cuối cùng.

---

### (2) Công thức tổng quát  

$$
ϕ_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(M - |S| - 1)!}{M!} [ f(S ∪ \{i\}) - f(S) ]
$$

Trong đó:  
- **N**: tập hợp toàn bộ đặc trưng  
- **S**: một tập con của các đặc trưng không chứa i  
- **f(S)**: giá trị dự đoán khi mô hình chỉ xem xét các đặc trưng trong S  
- **ϕᵢ**: giá trị SHAP của đặc trưng i  

Công thức này thể hiện rõ triết lý “công bằng” của Shapley:  
> Mỗi đặc trưng được chia phần tương ứng với **mức ảnh hưởng trung bình** của nó,  
> được tính trên **tất cả các cách tham gia vào mô hình**.

---

### (3) Ví dụ trực quan  

Giả sử ta có một mô hình dự đoán **giá nhà** dựa trên ba đặc trưng:
- Diện tích (x₁)  
- Số phòng ngủ (x₂)  
- Vị trí (x₃)

Khi chỉ dùng (x₁, x₂), mô hình dự đoán **2.5 tỷ**.  
Khi thêm “vị trí” (x₃), dự đoán tăng lên **3.0 tỷ**.  
→ Đóng góp của x₃ là **+0.5 tỷ**.  

Nếu ta làm tương tự cho mọi tổ hợp khác (chỉ x₁, chỉ x₃, x₁+x₃, v.v.),  
rồi lấy trung bình có trọng số trên tất cả, ta thu được giá trị SHAP cuối cùng cho “vị trí”.  

Khi thực hiện điều này cho tất cả các đặc trưng, ta thu được mô hình giải thích tuyến tính:

$$
f(x) = ϕ₀ + ϕ₁ + ϕ₂ + ϕ₃
$$

Trong đó:  
- ϕ₀: giá trị dự đoán trung bình (baseline)  
- ϕ₁, ϕ₂, ϕ₃: đóng góp cụ thể của từng đặc trưng  

---

### (4) Minh họa thực tế – ví dụ dự đoán phê duyệt vay  

Giả sử một ngân hàng dùng mô hình AI để dự đoán khả năng khách hàng được duyệt vay.

| Đặc trưng | Giá trị SHAP (đóng góp) | Ảnh hưởng |
|------------|--------------------------|------------|
| Giá trị trung bình (baseline) | 0.30 | Mặc định 30% cơ hội |
| Tuổi | +0.10 | Tuổi phù hợp tăng khả năng duyệt |
| Thu nhập | +0.20 | Thu nhập cao giúp tăng điểm tín dụng |
| Nợ xấu | −0.05 | Hồ sơ nợ xấu kéo điểm xuống |

Cộng tất cả lại:

$$
f(x) = 0.30 + 0.10 + 0.20 − 0.05 = 0.55
$$

→ Mô hình dự đoán rằng khách hàng có **55% khả năng được duyệt vay**.  
Không chỉ có kết quả, mà giờ ta **hiểu rõ “vì sao”**:  
thu nhập và tuổi đóng góp tích cực, trong khi lịch sử nợ xấu làm giảm điểm.

---

### (5) Tính chất đặc biệt của SHAP  

SHAP nổi bật nhờ ba tính chất lý thuyết được chứng minh:  

| Nguyên tắc | Ý nghĩa |
|-------------|----------|
| **Local Accuracy** | Tổng các giá trị SHAP đúng bằng sai lệch giữa dự đoán và giá trị trung bình. |
| **Missingness** | Nếu một đặc trưng không xuất hiện, đóng góp của nó bằng 0. |
| **Consistency** | Nếu một đặc trưng trở nên quan trọng hơn trong mô hình mới, giá trị SHAP của nó không được giảm đi. |

Nhờ tuân theo ba nguyên tắc này, SHAP được xem là **phương pháp duy nhất trong nhóm XAI đảm bảo tính công bằng và minh bạch.**

---

### (6) Mã minh họa trực quan trên Google Colab  

```python
# Cài đặt thư viện
!pip install shap xgboost

import shap
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

# 1. Tải dữ liệu và huấn luyện mô hình
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgboost.XGBRegressor().fit(X_train, y_train)

# 2. Tạo SHAP explainer và tính giá trị
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# 3. Hiển thị biểu đồ tổng hợp
shap.summary_plot(shap_values, X_test, show=True)
```

Kết quả là **SHAP Summary Plot** – biểu đồ thể hiện mức ảnh hưởng của từng đặc trưng đến kết quả dự đoán:

- **Trục ngang:** độ ảnh hưởng (impact)  
- **Màu đỏ:** giá trị feature cao  
- **Màu xanh:** giá trị feature thấp  
- **Càng xa trục 0 → ảnh hưởng càng mạnh đến kết quả**

---

## 3.4. Sự ra đời của các biến thể SHAP  

Khi mô hình và dữ liệu ngày càng lớn, việc tính toán **Shapley Values** đầy đủ trở nên tốn kém.  
Do đó, nhiều **biến thể mở rộng của SHAP** như **TreeSHAP**, **KernelSHAP**, **DeepSHAP** đã được phát triển để cân bằng giữa **tốc độ** và **độ chính xác**.  
Các phương pháp này sẽ được trình bày chi tiết hơn ở **phần 6** của blog.


# 4. Vì sao SHAP không thể trực tiếp giải thích Genetic Algorithm – và những hướng khắc phục khả thi

## 4.1. Vấn đề “Hộp đen trong hộp đen” của trí tuệ tiến hoá
Trong học máy hiện đại, khả năng giải thích mô hình (model interpretability) là yếu tố thiết yếu để đảm bảo tính tin cậy, minh bạch và khả năng kiểm chứng của hệ thống AI.
Các mô hình có độ phức tạp cao như mạng nơ-ron sâu, mô hình boosting, hoặc các thuật toán tiến hoá thường đạt hiệu năng vượt trội, song lại hoạt động như những “hộp đen” – con người khó hiểu được tại sao và bằng cách nào chúng đưa ra kết quả.

Để đối phó với vấn đề này, các kỹ thuật Giải thích mô hình (XAI – eXplainable Artificial Intelligence) như LIME, Permutation Importance, và đặc biệt SHAP (SHapley Additive exPlanations) đã được phát triển. SHAP nổi bật bởi nền tảng toán học vững chắc từ lý thuyết giá trị Shapley trong trò chơi hợp tác, cho phép định lượng mức đóng góp trung bình của từng đặc trưng tới đầu ra mô hình.

Tuy nhiên, khi chuyển sang lĩnh vực tính toán tiến hoá (Evolutionary Computation) – cụ thể là Genetic Algorithm (GA) – câu hỏi nảy sinh:

Liệu SHAP có thể giúp giải thích “cách thức tiến hoá” của GA giống như cách nó giải thích các mô hình học máy khác không?

Câu trả lời, ít nhất ở thời điểm hiện tại, là không thể trực tiếp.

## 4.2. GA và SHAP: Hai thế giới khác nhau

### 4.2.1. Genetic Algorithm (GA)

GA là một thuật toán tối ưu hoá không dựa trên gradient, mô phỏng quá trình tiến hoá tự nhiên. Điểm cốt lõi là Genetic Algorithm (GA) không phải một mô hình dự đoán (predictive model), mà là một quy trình **tối ưu hoá động** (dynamic optimization process) — tức là thay vì học một hàm ánh xạ cố định giữa đầu vào và đầu ra, GA liên tục tiến hoá quần thể nghiệm thông qua các bước **chọn lọc**, **lai ghép** và **đột biến** để dần tìm ra nghiệm tốt nhất.

### 4.2.2.  SHAP

Mục tiêu của SHAP là định lượng mức độ đóng góp của từng đặc trưng đầu vào đối với kết quả đầu ra của một **mô hình dự đoán tĩnh** — tức là một hàm ánh xạ xác định từ không gian đầu vào sang đầu ra, ký hiệu chung là:

$$
f: X \rightarrow Y
$$
SHAP hoạt động dựa trên **nguyên tắc additivity** (tính cộng gộp), tức là đầu ra của mô hình có thể được biểu diễn như tổng các đóng góp của từng đặc trưng, cùng với một giá trị nền (baseline prediction):

$$
f(x) = \phi_0 + \sum_i \phi_i
$$

Điểm cốt lõi của SHAP nằm ở việc mô hình phải **có thể được gọi lặp lại nhiều lần** với các tổ hợp đặc trưng khác nhau, để ước lượng **tác động biên (marginal contribution)** của từng đặc trưng lên kết quả dự đoán. Khi đó, SHAP coi mỗi đặc trưng như một “người chơi” trong trò chơi hợp tác, và phân phối **công bằng giá trị dự đoán** giữa các người chơi theo nguyên tắc Shapley.

Về mặt bản chất, SHAP là phương pháp **giải thích hàm (function-based explanation)**, chứ không phải **quy trình (process-based explanation)**. Nó yêu cầu mô hình có:

- **Đầu vào – đầu ra xác định**,  
- **Tính lặp lại (determinism)** trong dự đoán,  
- **Khả năng tách biệt rõ ràng giữa các đặc trưng**.

Do đó, SHAP phù hợp để giải thích các mô hình như hồi quy, mạng nơ-ron, hay các mô hình tree-based (như Random Forest, XGBoost) — nơi ta có thể xác định rõ ràng $f(x)$ cho mỗi điểm dữ liệu.

Ngược lại, với các thuật toán như GA, nơi không tồn tại một **hàm dự đoán tĩnh duy nhất**, mà chỉ có một **quy trình tiến hoá phụ thuộc vào ngẫu nhiên và thời gian**, SHAP không thể áp dụng trực tiếp.

## 4.3. Tại sao SHAP không thể giải thích GA trực tiếp

SHAP yêu cầu mô hình có **hàm dự đoán cố định** $f(x)$ có thể gọi lại nhiều lần để đo tác động của từng đặc trưng. GA không thỏa điều kiện này vì:  

### 4.3.1. GA không có hàm mô hình cố định
GA chỉ biết **hàm fitness**, còn quá trình tiến hoá là tổ hợp nhiều phép biến đổi ngẫu nhiên:

$$
\text{Population}_{t+1} = \text{Mutate}(\text{Crossover}(\text{Select}(\text{Population}_t)))
$$

Kết quả phụ thuộc vào:  
- Cấu hình khởi tạo  
- Ngẫu nhiên trong lai ghép/đột biến  
- Thứ tự đánh giá fitness  

→ Chạy lại GA với cùng tham số vẫn có thể cho kết quả khác nhau. **Không tồn tại hàm $f(x)$ cố định** để SHAP áp dụng.

### 4.3.2. GA không tạo ra đầu ra tức thời
Để đánh giá ảnh hưởng của một gene, phải tiến hoá lại toàn bộ quá trình, vì loại bỏ gene sẽ thay đổi toàn bộ quỹ đạo tiến hoá.

### 4.3.3. GA là quy trình tuần tự – ngẫu nhiên – phi tuyến
GA là phi tuyến, phi xác định, với tương tác mạnh giữa các gene. Việc tính SHAP value trực tiếp sẽ chỉ tạo ra **nhiễu thống kê**, không phản ánh bản chất tiến hoá.

> Như vậy, nguyên nhân khiến SHAP không thể áp dụng trực tiếp cho GA là **tính ngẫu nhiên, phi tuyến và thiếu hàm dự đoán cố định**. Điều này dẫn đến sự khác biệt rõ ràng trong bản chất “hộp đen” của GA so với các mô hình tree-based, như sẽ trình bày ở phần tiếp theo.

## 4.4. Sự khác biệt về bản chất “hộp đen” giữa GA và các mô hình tree-based

Mặc dù cả GA và các mô hình tree-based (như Random Forest, Gradient Boosting, XGBoost) thường được xem là “hộp đen” trong học máy, song bản chất “đen” của hai nhóm này đến từ những nguyên nhân hoàn toàn khác nhau.

Thứ nhất, đối với mô hình tree-based, “hộp đen” xuất phát từ độ phức tạp cấu trúc và số lượng mô hình con.  
Một mô hình như Random Forest có thể bao gồm hàng trăm đến hàng nghìn cây quyết định, mỗi cây học một tập con dữ liệu và đặc trưng khác nhau. Khi dự đoán, đầu ra của từng cây được kết hợp (thông qua trung bình hoặc voting), tạo thành một hệ thống phi tuyến, khó suy luận trực tiếp mối quan hệ giữa từng đặc trưng và đầu ra cuối cùng.  
Tuy nhiên, về bản chất, các mô hình tree-based vẫn là **hàm xác định** $f(x)$: với cùng một đầu vào, luôn cho ra cùng một đầu ra. Điều này giúp chúng có thể được giải thích **hậu định (post-hoc)** bằng SHAP hoặc các kỹ thuật tương tự.

Thứ hai, đối với GA, tính “hộp đen” lại đến từ bản chất **quá trình tiến hoá ngẫu nhiên và động theo thời gian**.  
GA không học một hàm cố định, mà thực hiện quá trình lặp gồm chọn lọc, lai ghép và đột biến — liên tục thay đổi không gian nghiệm qua từng thế hệ. Kết quả cuối cùng (nghiệm tối ưu) không chỉ phụ thuộc vào dữ liệu đầu vào mà còn vào trạng thái khởi tạo, các tham số tiến hoá (như mutation rate, crossover rate) và yếu tố ngẫu nhiên trong từng lần chạy.  
Do đó, GA không có **hàm ánh xạ tĩnh** từ đầu vào sang đầu ra, mà chỉ là một **chuỗi biến đổi phi xác định**. Điều này khiến việc áp dụng SHAP hay các phương pháp dựa trên khái niệm “đóng góp đặc trưng” trở nên không khả thi theo nghĩa truyền thống.

Tóm lại, “tính hộp đen” của tree-based models đến từ sự phức tạp cấu trúc nhưng vẫn có thể giải thích hậu định, trong khi “tính hộp đen” của GA đếnn từ sự **ngẫu nhiên, phi tuyến và không có hàm mô hình cố định** — khiến việc giải thích cần hướng sang **mức meta** hoặc **surrogate model** thay vì trực tiếp.

## 4.5. Các hướng giải thích gián tiếp cho GA (Indirect explainability for GA)

### Ý tưởng chung

Vì GA là một quy trình tiến hóa động (không phải một hàm dự đoán tĩnh), nên SHAP không thể áp dụng trực tiếp để “giải thích” GA. Tuy nhiên, ta có thể giải thích kết quả hoặc hành vi của GA bằng các phương pháp gián tiếp sau — mỗi hướng đều cho phép đưa vào SHAP hoặc quan niệm Shapley để đạt interpretability.

---

### 4.5.1. Surrogate-based explainability (Mô hình thay thế + SHAP)

**Cơ chế:**

Trong quá trình chạy GA, lưu lại nhiều cặp (genome, fitness) (ví dụ: các cá thể trong nhiều thế hệ).  

Huấn luyện một mô hình surrogate $\hat{f}$ (ví dụ XGBoost, RandomForest) để xấp xỉ hàm fitness:  
$$
\text{fitness} \approx \hat{f}(\text{genome})
$$

Áp dụng SHAP lên $\hat{f}$ để tính SHAP values cho từng gene — từ đó biết gene nào ảnh hưởng mạnh đến fitness.

**Code minh họa:**

```python
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor

# Giả sử ta đã có dữ liệu GA log
genomes = np.random.rand(200, 10)
fitness = np.sin(genomes[:, 0]) + np.cos(genomes[:, 1]) + np.random.randn(200) * 0.1

# Huấn luyện mô hình surrogate
model = RandomForestRegressor()
model.fit(genomes, fitness)

# Áp dụng SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(genomes)

# Hiển thị kết quả
shap.summary_plot(shap_values, genomes, feature_names=[f"gene_{i}" for i in range(genomes.shape[1])])
```

![SHAP summary plot](images/shap_summary.png)


### 4.5.2. Logging + Post-hoc Attribution (Ghi log tiến hoá → phân tích Shapley)

#### **Cơ chế**

- Ghi chép chi tiết **logs** trong quá trình tiến hoá:  
  - Cá thể được chọn hoặc loại bỏ.  
  - **Marginal contribution** của cá thể hoặc biến trong từng thế hệ.  
  - **Tần suất xuất hiện gene**, hoặc các thống kê về sự thay đổi fitness.  

- Dùng các **chỉ số thống kê** (ví dụ: tần suất, *average marginal gain*, *rate of improvement* khi gene xuất hiện) để **định lượng tầm quan trọng của gene**.  

- Áp các **biến thống kê** này làm **input cho SHAP** (hoặc chỉ báo *Shapley-like*) nhằm **giải thích tại sao GA tiến hoá theo hướng đó**.

---

#### **Dẫn chứng học thuật**

Các công trình **game-theoretic feature selection** (*Zaeri-Amirani et al., 2018*) đã minh hoạ cách **GA chọn các coalition** và **tính marginal contribution** để **ước lượng giá trị Shapley**.  
→ Ý tưởng **logging coalition + lấy marginal** chính là tiền đề cho hướng tiếp cận này.  

📄 *Tham khảo:* Zaeri-Amirani, et al. (2018). *GA-based Monte Carlo Shapley approximation for feature selection.*

---

#### **Ý nghĩa / Ưu nhược**

- **Ưu điểm:**  
  - Giữ lại phần **động học tiến hoá** — cho phép phân tích chiến lược thích nghi qua các giai đoạn.  
  - Có thể chỉ ra **gene nào quan trọng ở từng giai đoạn** (đầu, giữa, cuối).  

- **Nhược điểm:**  
  - Cần **lưu lượng dữ liệu lớn** (toàn bộ lịch sử tiến hoá).  
  - **Phân tích phức tạp** vì mang tính chuỗi thời gian (*time-series of importance*).  

---

#### **Ví dụ minh hoạ (Python pseudo-code)**

```python
# Logging GA evolution process
import pandas as pd

logs = []

for gen in range(num_generations):
    population = evolve(population)
    for ind in population:
        logs.append({
            'generation': gen,
            'individual': ind.genes,
            'fitness': ind.fitness,
            'marginal_gain': ind.fitness - avg_prev_fitness(ind)
        })

# Convert logs to DataFrame
df_logs = pd.DataFrame(logs)

# Aggregate statistics per gene
gene_stats = df_logs.groupby('gene').agg({
    'marginal_gain': 'mean',
    'fitness': 'mean'
}).reset_index()

# Apply SHAP (post-hoc explainability)
import shap
model = shap.Explainer(surrogate_model)
shap_values = model(gene_stats)

shap.summary_plot(shap_values, gene_stats)
```
![SHAP summary plot](images/post_hoc.png)
```
   generation                                        gene_values   fitness  \
0           0  [0.9255534584552242, 0.21746386134556728, 0.13...  2.791049   
1           0  [0.6129517726409283, 0.708922426261847, 0.0337...  1.135790   
2           0  [0.9304056014269897, 0.7445169972848723, 0.524...  3.474486   
3           0  [0.3835035901648618, 0.5292369439072145, 0.792...  1.888319   
4           0  [0.06554067151333975, 0.9469961975270887, 0.55...  0.908424   

   marginal_gain  
0       0.558826  
1      -1.096434  
2       1.242262  
3      -0.343905  
4      -1.323799  
```
### 4.5.3. SHAP-guided diagnostics — Dùng SHAP để chẩn đoán GA

#### Cơ chế

Ý tưởng của hướng này là **áp dụng SHAP không để giải thích bản thân GA**, mà để **giám sát hoặc chẩn đoán hành vi tiến hoá của GA** thông qua các mô hình phụ (surrogate) hoặc tập nghiệm tốt nhất (best solutions) mà GA sinh ra.

Cụ thể:

- Áp SHAP lên **mô hình surrogate** được huấn luyện để xấp xỉ mối quan hệ giữa bộ gene và fitness,  
  **hoặc** trực tiếp lên **tập nghiệm tốt nhất** thu được sau quá trình tiến hoá.  
- Nếu kết quả SHAP cho thấy **fitness bị chi phối bởi một số ít gene**, điều này gợi ý hiện tượng:
  - *loss of diversity* (mất đa dạng quần thể), hoặc  
  - *bias* do GA hội tụ sớm vào một **local optimum**.  
- Khi đó, người nghiên cứu có thể điều chỉnh **siêu tham số của GA** để khắc phục:
  - tăng mutation rate  
  - thay đổi selection pressure  
  - bổ sung cơ chế duy trì đa dạng (niching, crowding, speciation)

---
#### Cơ sở học thuật và minh hoạ

Fryer et al. (2021) trong bài viết về **giới hạn của giá trị Shapley cho bài toán chọn đặc trưng (feature selection)** đã chỉ ra rằng các chỉ số attribution như SHAP có thể phát hiện những đặc trưng “áp đảo”, từ đó cảnh báo về **sự mất cân bằng đóng góp** trong mô hình.  
Ý tưởng này có thể mở rộng cho GA: nếu một vài gene có attribution vượt trội, ta có thể coi đó là tín hiệu cảnh báo về **thiên lệch tiến hoá (evolutionary bias)**.

Thực tiễn, việc áp dụng SHAP-guided diagnostics giúp theo dõi **động lực học thích nghi** (adaptive dynamics) của GA và điều chỉnh cấu hình để tránh hội tụ sớm.

---

#### Ý nghĩa và đánh giá

**Ưu điểm:**
- SHAP trở thành công cụ **giám sát** (monitoring) tiến trình tiến hoá của GA.  
- Dễ triển khai, có thể kết hợp với các công cụ trực quan hoá (ví dụ summary plot hoặc dependence plot).

**Hạn chế:**
- Phương pháp vẫn **gián tiếp** — không phân tích GA ở cấp độ cơ chế mà thông qua mô hình trung gian.  
- Việc điều chỉnh GA dựa quá nhiều vào attribution có thể làm **giảm tính khám phá** nếu can thiệp quá mức vào mutation hoặc selection.

---

**Nguồn tham khảo:**  
Fryer et al., *On the Limitations of Shapley Values for Feature Selection*, arXiv:2106.XXXX, 2021.

### 4.5.4. Hướng ngược lại — GA → Shapley (ước lượng giá trị Shapley)

Đây là hướng nổi bật trong tài liệu hiện có và là mấu chốt nên nhấn mạnh trong blog — có cơ sở học thuật rõ ràng (Zaeri-Amirani et al., 2018) và một số công trình gần đây mở rộng ý tưởng này.

#### a. Tại sao cần ước lượng Shapley?

**Vấn đề:**  
Shapley value theo lý thuyết yêu cầu xét tất cả các tập con → chi phí tính toán O(2^N) (không khả thi khi N lớn).

**Giải pháp tổng quát:**  
- Sampling (Monte-Carlo)  
- Kernel-approximation (KernelSHAP)  
- Heuristic chọn mẫu thông minh  

GA là một lựa chọn sampling hướng mục tiêu (targeted sampling) để tìm những coalition có đóng góp lớn, từ đó ước lượng giá trị trung bình chính xác hơn với ít mẫu hơn.

**Nguồn gốc & dẫn chứng:**  
Zaeri-Amirani et al. (2018) trình bày GA-based Monte-Carlo method để sinh các coalition có marginal contribution lớn và ước lượng Shapley value từ chúng; phương pháp này giảm phức tạp tính toán và đạt kết quả tốt trong bài toán giảm báo động giả trên dữ liệu PhysioNet (AUC ≈ 0.81 cho Shapley µ=3.5).

---

#### b. Nguyên lý hoạt động (chi tiết thuật toán)

**Mã hóa (Encoding):**  
Mỗi chromosome = vector nhị phân biểu diễn coalition (ví dụ các feature có trong coalition).

**Fitness function:**  
Định nghĩa marginal contribution của một feature i đối với coalition T:

$$
f_i(\text{chr}(T)) = \nu(T \cup \{i\}) - \nu(T)
$$

Trong đó ν(T) là giá trị (score) của coalition T (Zaeri-Amirani dùng kết hợp sensitivity & specificity làm ν).

**Quá trình tiến hoá:**  
1. Khởi tạo population cho mỗi kích thước coalition t  
2. Tiến hoá bằng selection / crossover / mutation để tìm các coalition có marginal contribution lớn  
3. Lặp cho nhiều t và bình quân các marginal contributions để ước lượng φ̂_i

---

#### c. Lý luận thống kê & phức tạp

- Nếu giới hạn kích thước coalition tới n_max và sinh n_G mẫu, complexity giảm từ exponential xuống:

$$
O(n_f \times n_{\text{max}} \times n_G)
$$

- Sử dụng mô hình phân phối cực trị (Gumbel / EX1) để ước lượng expected max của marginal sample và điều chỉnh sai số sampling.

---

#### d. Kết quả thực nghiệm & so sánh

- Trên bộ dữ liệu PhysioNet, phương pháp GA-Shapley chọn 20 features và đạt AUC ≈ 0.80–0.81  
- Vượt hầu hết phương pháp truyền thống (χ², tree-based, Relief) trong mục tiêu tăng sensitivity cho bài toán false alarm reduction.

**Lưu ý:**  
Phương pháp này ước lượng Shapley (approximate) — không cho kết quả chính xác tuyệt đối nhưng đánh đổi bằng tiết kiệm tính toán và khả năng mở rộng tới tập feature lớn (hàng trăm).

---

**Nguồn tham khảo:**  
Zaeri-Amirani et al., 2018.
Các hướng giải thích gián tiếp cho GA (Indirect Explainability for Genetic Algorithms)## 7. Hướng phát triển: Explainable Evolutionary Computation (EEC)

Tương lai có thể hình thành một nhánh nghiên cứu mới mang tên **Explainable Evolutionary Computation (EEC)** – hướng tới việc **tích hợp khả năng tự giải thích (self-explainability)** vào chính các thuật toán tiến hoá.

Ba hướng phát triển tiềm năng gồm:

1. **Surrogate explainability:**  
   GA được mô phỏng (surrogate) bởi một mô hình học máy, sau đó mô hình này được giải thích bằng SHAP.  
   → Cách tiếp cận gián tiếp, cho phép hiểu được mối quan hệ giữa cấu trúc gene và fitness.

2. **In-process explainability:**  
   GA được mở rộng để tự ghi nhận các thống kê trong quá trình tiến hoá — chẳng hạn như **gene importance**, **mutation impact**, hoặc **selection pressure** theo thời gian.  
   → Cách này biến GA thành hệ thống có “nhật ký tiến hoá” có thể phân tích trực tiếp.

3. **Dual explainability:**  
   GA tích hợp một **Shapley-like metric nội tại** để đo lường đóng góp của từng gene ngay trong quá trình tiến hoá.  
   → Khi đó, “SHAP cho GA” xuất hiện ở tầng **meta**, tức là GA **tự đánh giá tầm quan trọng gene của mình** mà không cần mô hình ngoài.

Những hướng này mở ra khả năng phát triển các hệ tiến hoá **tự quan sát, tự đánh giá, và tự điều chỉnh**, góp phần định hình một thế hệ **thuật toán tiến hoá có thể giải thích được** (explainable evolutionary algorithms).

---

## 4.6. Kết luận

SHAP là công cụ mạnh mẽ cho **model explainability**, nhưng chỉ khi **đối tượng là mô hình có hàm đầu ra xác định**.  
Ngược lại, **Genetic Algorithm** là một **quá trình tìm kiếm tiến hoá ngẫu nhiên**, không có cấu trúc hàm cố định.

Vì vậy:

- SHAP **không thể trực tiếp giải thích** GA,  
- nhưng **có thể giải thích gián tiếp** hành vi hoặc kết quả mà GA tạo ra,  
- và **GA lại có thể hỗ trợ SHAP** trong việc giảm chi phí tính toán giá trị Shapley.

> SHAP giúp ta hiểu mô hình;  
> GA giúp ta tìm lời giải;  
> Kết hợp cả hai – ta tiến gần hơn tới một **trí tuệ nhân tạo vừa mạnh, vừa minh bạch.**



# 5. Case Study: Tối ưu đường đi của robot giao hàng (Delivery Route Optimization)

Sau khi đã hiểu rõ **Genetic Algorithm (GA)** là gì và **SHAP** có thể giúp ta nhìn thấu “hộp đen” của mô hình ra sao, hãy cùng chuyển sang một ví dụ thực tiễn cụ thể — nơi **GA thực sự phát huy sức mạnh tối ưu hóa**.


## 5.1. Từ bối cảnh đến mô hình bài toán

Trong các thành phố hiện đại, dịch vụ giao hàng tự động bằng **robot di chuyển mặt đất (AGV – Autonomous Ground Vehicle)** đang ngày càng phổ biến.  
Một công ty logistic cần điều phối nhiều robot giao hàng trong khu công nghiệp, sao cho **mỗi robot hoàn thành tất cả điểm giao với quãng đường ngắn nhất**, **tiết kiệm năng lượng**, và **không trễ thời gian giao hàng**.

Nếu mô hình hóa bài toán này, ta thấy nó là biến thể của **Traveling Salesman Problem (TSP)** — một trong những bài toán **NP-hard** kinh điển trong tối ưu tổ hợp.

Mục tiêu của ta là tìm **chu trình tối ưu** đi qua tất cả các điểm giao hàng và quay về kho, với chi phí tổng thể tối thiểu.

---

### 5.1.1. Mô hình hóa toán học

Giả sử có:
- $N$ điểm giao hàng (không tính kho)
- Kho hàng ký hiệu là điểm $0$
- Ma trận khoảng cách $D = [d_{ij}]$, trong đó $d_{ij}$ là quãng đường giữa điểm $i$ và $j$

Khi đó, đường đi (hay “chromosome”) được biểu diễn bằng một hoán vị:
$$
R = [0, r_1, r_2, \dots, r_N, 0]
$$

Tổng quãng đường phải di chuyển là:
$$
L(R) = \sum_{i=0}^{N} d_{r_i, r_{i+1}}
$$

---

### 5.1.2. Hàm mục tiêu (Fitness Function)

Để mô hình phản ánh thực tế, ngoài khoảng cách, ta cần tính đến:
- Năng lượng tiêu hao $E(R)$ (tăng theo tải trọng và quãng đường)
- Phạt trễ thời gian $P(R)$ nếu giao hàng muộn hơn deadline

Hàm tối ưu tổng quát là:

$$
\text{Minimize } F(R) = \alpha \times L(R) + \beta \times E(R) + \gamma \times P(R)
$$

với $\alpha, \beta, \gamma$ là các hệ số trọng số (trade-off giữa độ dài, năng lượng và độ trễ).  
Trong GA, ta thường định nghĩa **hàm fitness** tỷ lệ nghịch với chi phí này:

$$
\text{Fitness}(R) = \frac{1}{F(R) + \varepsilon}
$$

với $\varepsilon$ là hằng số rất nhỏ để tránh chia cho 0.


## 5.2. Cách GA hoạt động trong bài toán này

Hãy xem cách GA “tiến hóa” để tìm lời giải tối ưu cho bài toán trên.

### **Bước 1 – Khởi tạo quần thể**

Tạo ngẫu nhiên $M$ tuyến đường khác nhau, mỗi tuyến là một hoán vị hợp lệ của các điểm giao hàng.

Ví dụ:
```
Cá thể 1: [0, 2, 4, 1, 3, 0]
Cá thể 2: [0, 3, 1, 4, 2, 0]
...
```

Mỗi cá thể biểu diễn **một phương án lộ trình** mà robot có thể đi.


### **Bước 2 – Đánh giá fitness**

Tính **fitness** cho từng cá thể dựa trên hàm $F(R)$ đã nêu:

$$
\text{Fitness}(R) = \frac{1}{\alpha L(R) + \beta E(R) + \gamma P(R)}
$$

→ Cá thể có **fitness cao** nghĩa là tuyến đường **ngắn**, **tiết kiệm năng lượng**, và **ít trễ hẹn**.


### **Bước 3 – Chọn lọc (Selection)**

Chọn các cá thể tốt để làm “bố mẹ”.  
Hai cách chọn phổ biến:

- **Roulette Wheel Selection:**  
  Xác suất chọn tỷ lệ với fitness. Cá thể tốt có nhiều “phần bánh” hơn.
- **Tournament Selection:**  
  Chọn ngẫu nhiên $k$ cá thể, lấy cá thể có fitness cao nhất.


### **Bước 4 – Lai ghép (Crossover)**

Hai cá thể bố mẹ kết hợp gen để sinh ra cá thể con.  
Với bài toán TSP, ta dùng **Order Crossover (OX)**:

| Cha | 0  | 2 | 4 | 1 | 3 | 0 |
|-----|----|---|---|---|---|---|
| Mẹ  | 0  | 3 | 1 | 4 | 2 | 0 |

- Giữ nguyên đoạn từ cha, điền phần còn lại theo thứ tự mẹ → đảm bảo không trùng lặp điểm.

---

### **Bước 5 – Đột biến (Mutation)**

Hoán đổi ngẫu nhiên 2 điểm trong lộ trình để duy trì đa dạng quần thể:

$$
R' = [0, 3, 1, 2, 4, 0] \rightarrow [0, 3, \underline{4}, 2, \underline{1}, 0]
$$

→ Đột biến giúp **tránh kẹt ở cực trị cục bộ (local optima)**.


### **Bước 6 – Tạo thế hệ mới**

- Giữ lại một số cá thể tốt nhất (elitism).  
- Thêm các cá thể con mới được lai/đột biến.

Lặp lại quá trình này qua **nhiều thế hệ (generations)** cho đến khi fitness hội tụ.


## 5.3. Minh họa Genetic Algorithm giúp cho Robot giao hàng
- Minh hoạ thể hiện cách GA tiến hoá để tối ưu hoá lộ trình giao hàng cho 1 robot với 15 điểm giao:

### Tuyến đường tối ưu:
![Tuyến đường tối ưu của robot giao hàng](images/5.GA-run.png)

### Tiến trình tối ưu (Fitness qua các thế hệ)
![Tiến trình tối ưu](images/5.GA-vis.png)

## Phân tích kết quả

Sau vài trăm thế hệ, GA dần tìm ra tuyến đường có:
- Tổng quãng đường ngắn hơn ~10–15% so với random baseline  
- Năng lượng tiêu hao thấp hơn (vì đường đi hợp lý hơn)  
- Không trễ deadline (nếu có ràng buộc thời gian)

Quá trình tiến hóa thể hiện **tinh thần Darwin** rõ rệt:
- Thế hệ đầu → hỗn loạn, lộ trình dài và tốn năng lượng  
- Thế hệ giữa → các gen “tốt” (cụm điểm gần nhau) dần được giữ lại  
- Thế hệ cuối → lộ trình hội tụ, fitness gần như bão hòa

---

## 5.4. Ý nghĩa và mở rộng

### (1) Về mặt kỹ thuật
GA chứng minh khả năng giải bài toán **tối ưu phi tuyến, không khả vi** — nơi Gradient Descent bất lực.  
Bài toán Delivery Route Optimization là ví dụ kinh điển thể hiện:
- **Tính toàn cục (global search)**  
- **Khả năng thoát local optima**  
- **Khả năng xử lý ràng buộc đa mục tiêu**

### (2) Về mặt nghiên cứu
Khi kết hợp với các mô hình **Explainability (ví dụ SHAP)** như phần 6 sắp tới, ta có thể:
- Phân tích **gene nào** trong lộ trình (ví dụ: nhóm điểm gần nhau, hoặc góc rẽ cụ thể) ảnh hưởng mạnh nhất đến fitness  
- Giải thích tại sao GA hội tụ theo hướng đó  
- Thiết kế lại tham số mutation/crossover có cơ sở hơn

---

> Từ một bài toán giao hàng tưởng chừng đơn giản, ta đã thấy **Genetic Algorithm** mô phỏng trọn vẹn quy luật tiến hóa tự nhiên:  
> chọn lọc, lai ghép, đột biến, và tiến hóa dần đến tối ưu.  
> Đây cũng chính là sức mạnh khiến GA trở thành công cụ nền tảng cho nhiều hệ thống **AI mô phỏng tự nhiên (Nature-inspired AI)** hiện nay.

# 6. Đề xuất cải tiến
Ở phần này, ta sẽ đi sâu vào các phương pháp cải tiến nổi bật dựa trên lý thuyết SHAP để giải thích mô hình học máy, bao gồm  **Tree SHAP**,  **Kernel SHAP**  và  **Deep SHAP**. Mỗi phương pháp có những điểm mạnh riêng để giải quyết các hạn chế trong việc tính toán giá trị Shapley truyền thống.

## 6.1. Tree SHAP - tăng tốc cho mô hình cây
Tree SHAP là thuật toán tối ưu để tính giá trị SHAP (Shapley Additive Explanations) cho các mô hình dựa trên cây quyết định như XGBoost, LightGBM, CatBoost... Nó dựa trên lý thuyết giá trị Shapley trong trò chơi hợp tác, chia đều kết quả dự đoán của mô hình cho từng đặc trưng một cách công bằng.
### Nguyên lý tính toán Tree SHAP

-   Giá trị SHAP truyền thống yêu cầu tính trung bình đánh giá sự đóng góp của đặc trưng  $i$  trên  mọi tập con có hoặc không có đặc trưng đó, dẫn đến độ phức tạp tính toán theo hàm mũ  $O(2^{|N|})$, với  $∣N∣$  là số đặc trưng.
    
-   Tree SHAP sử dụng đặc thù cấu trúc cây để tính toán giá trị này  một cách hiệu quả bằng thuật toán đệ quy duyệt các nhánh cây, tính trọng số xác suất mẫu đi qua từng nhánh dựa trên tập đặc trưng tham gia, mà không cần xét hết tất cả tập con.
    
-   Thuật toán tận dụng tính chất  tính cộng của Shapley  (SHAP cho mô hình rừng cây là tổng SHAP của từng cây) và cấu trúc phân nhánh của cây, giảm độ phức tạp xuống đa thức:  $O(TLD^2)$, với  $T$  số cây,  $L$  số lá tối đa mỗi cây,  $D$  độ sâu tối đa cây.

Dưới đây là bảng so sánh sự khác biệt giữa SHAP và Tree SHAP:
| Tiêu chí| SHAP Thường| Tree SHAP  
|-|-|-|
| Phạm vi áp dụng | Mọi mô hình (model-agnostic) | Mô hình cây và ensemble cây (XGBoost, LightGBM, Random Forest)|
| Công thức cơ bản | Giá trị Shapley trung bình trên mọi tập con| Thuật toán đệ quy trên cây tính trọng số xác suất từng nhánh|
| Độ phức tạp tính toán | $O(2^{\|N\|})$, với  $∣N∣$  là số đặc trưng|$O(TLD^2)$, với  $T$  số cây,  $L$  số lá tối đa mỗi cây,  $D$  độ sâu tối đa cây |
| Tốc độ| Rất chậm với đặc trưng nhiều| Nhanh, sử dụng cấu trúc cây tối ưu hóa|



## 6.2. Kernel SHAP – linh hoạt cho mô hình phi tuyến
Kernel SHAP là một phương pháp **model-agnostic** giúp tính toán giá trị SHAP – giá trị đóng góp công bằng của từng đặc trưng – cho mọi mô hình học máy, kể cả những mô hình phi tuyến và hộp đen (black-box) mà ta không hiểu rõ cấu trúc bên trong.
### Nguyên lý hoạt động

-   Kernel SHAP biến bài toán tính giá trị Shapley thành một bài toán hồi quy tuyến tính trọng số cục bộ. Thuật toán lấy mẫu các tập con đặc trưng (hoặc "liên minh"), với mỗi tập con biểu diễn một cách mô phỏng việc biến đặc trưng "có mặt" hay "bị ẩn" khi dự đoán.
 -   Để ẩn đặc trưng, Kernel SHAP thay giá trị đặc trưng đó bằng giá trị trung bình hoặc mẫu từ một bộ dữ liệu nền (background dataset), đảm bảo giả định các đặc trưng độc lập.
-   Thuật toán đánh trọng số các tập con dựa trên kích thước tập con theo hàm kernel đặc biệt nhằm giữ tính công bằng của giá trị Shapley.
-   Sau khi lấy mẫu, Kernel SHAP thực hiện hồi quy tuyến tính trọng số giữa giá trị dự đoán của mô hình trên các tập con và các đặc trưng có mặt, từ đó ước lượng trọng số đóng góp (SHAP value) của từng đặc trưng.

### Công thức trọng số Kernel SHAP

Hàm trọng số kernel cho một tập con đặc trưng  $S$, tập đặc trưng đầy đủ  $N$  với  $M=∣N∣$, được tính bằng:
$$
\omega(S) = \frac{M - 1}{\binom{M}{|S|} \times |S| \times (M - |S|)}
$$
Công thức này cân bằng trọng số để đảm bảo mỗi đặc trưng được đánh giá đúng mức qua mẫu tập con.

### Sự khác biệt giữa SHAP và Kernel SHAP
| Tiêu chí             | SHAP (Permutation SHAP)| Kernel SHAP|
|----------------------|-------------------------------------|------------|
| Độ phức tạp tính toán | Tương đương, rất lớn và đột biến theo số đặc trưng (hàm mũ) | Xấp xỉ, tính gần đúng với hiệu quả cao, đặc biệt khi nhiều đặc trưng |
| Giả định tính toán    | Tính toán chính xác mối liên hệ đặc trưng, không giả định độc lập | Giả định đặc trưng độc lập, có thể sai lệch với dữ liệu có tương quan |
| Độ chính xác| Chính xác nhất | Gần bằng chính xác, đôi khi khác biệt với dữ liệu có tương quan đặc trưng |
| Ứng dụng| Dùng khi số đặc trưng nhỏ, ưu tiên độ chính xác     | Phổ biến với số đặc trưng lớn và mô hình hộp đen không biết cấu trúc |


## 6.3. Deep SHAP – giải thích cho mạng nơ-ron sâu

Deep SHAP là một phương pháp giải thích dựa trên khung SHAP được thiết kế đặc biệt cho các mô hình  **mạng nơ-ron sâu**  (Deep Neural Networks - DNNs). Nó kết hợp lý thuyết giá trị Shapley với các kỹ thuật lan truyền ngược (backpropagation) trong mạng thần kinh để ước tính và phân phối công bằng ảnh hưởng của từng đặc trưng đầu vào lên kết quả dự đoán.

### Lý do ra đời của Deep SHAP

-   Mạng nơ-ron sâu với hàng chục đến hàng trăm lớp ẩn có mô hình rất phức tạp, thường được xem như "hộp đen" do khó đoán sự đóng góp của từng đầu vào.
    
-   Các phương pháp SHAP truyền thống (như Kernel SHAP) có thể áp dụng nhưng thường rất tốn kém về tính toán và không tận dụng được cấu trúc đặc thù của mạng nơ-ron sâu.
    
-   Deep SHAP tận dụng đặc điểm cấu trúc mạng nơ-ron để giải thích mô hình một cách nhanh hơn và chính xác hơn.
    

### Nguyên lý hoạt động

-   Deep SHAP dựa trên sự kết hợp của SHAP và  **DeepLIFT**  – một phương pháp phân tích độ đóng góp dự đoán thông qua kỹ thuật lan truyền ngược (backpropagation) trong mạng neural.
    
-   DeepLIFT so sánh giá trị của các nơ-ron so với một giá trị tham chiếu (baseline) và tính toán sự thay đổi này truyền ngược về đầu vào, thể hiện đóng góp của từng đặc trưng.
    
-   Deep SHAP sử dụng các giá trị này để xấp xỉ  giá trị Shapley, đồng thời đảm bảo tính chất công bằng và các đặc điểm toán học của giá trị SHAP.

### Ưu điểm của Deep SHAP

-   **Hiệu quả tính toán:**  tận dụng việc lan truyền ngược nên nhanh hơn nhiều so với phương pháp mô phỏng tập con hay Kernel SHAP.
    
-   **Chính xác hơn các phương pháp gradient thuần túy:**  vì đánh giá dựa trên thay đổi tương đối so với baseline, giảm sai lệch do phi tuyến.
    
-   **Giữ được tính công bằng của Shapley values:**  đảm bảo mỗi đặc trưng nhận đóng góp phù hợp với ảnh hưởng thực tế.
    
-   **Phù hợp cho nhiều kiến trúc mạng sâu:**  từ mạng đa lớp đơn giản đến mạng tích chập (CNN), mạng hồi quy (RNN)...

## 6.4. Ứng dụng Deep SHAP để giải thích mô hình dự báo giá căn hộ tại Pháp
Trong bài viết này, chúng tôi xây dựng mô hình dự báo giá căn hộ dựa trên dữ liệu lớn chứa hơn 1,5 triệu giao dịch bất động sản tại Pháp. Dữ liệu gồm 62 biến đặc trưng mô tả chi tiết vị trí, diện tích, đặc điểm dân cư khu vực xung quanh.

Mô hình sử dụng là mạng nơ-ron đa lớp (deep neural network), gồm các lớp: 512 → 256 → 128 → 64 → 1 neuron cuối cùng, kết hợp kỹ thuật batch normalization và dropout để tránh quá khớp, giúp mô hình học các mối quan hệ phức tạp và phi tuyến tính trong dữ liệu.

Sau khi huấn luyện thành công, chúng tôi tiến hành giải thích mô hình bằng phương pháp Deep SHAP để xác định rõ từng biến ảnh hưởng như thế nào đến dự báo giá căn hộ. Các đặc trưng như diện tích xây dựng thực tế, thu nhập cư dân trong khu vực xung quanh, thành phần nghề nghiệp dân cư đều được phân tích để hiểu rõ vai trò và ảnh hưởng của từng yếu tố này đối với giá trị căn hộ.

Dưới đây là ví dụ minh họa kèm kết quả của việc sử dụng Deep SHAP để giải thích mô hình :

```python
def explain_model_deep_shap(
    model, 
    X_train_scaled, 
    X_test_scaled, 
    feature_cols, 
    device, 
    num_background=100, 
    num_samples=50, 
    feature_name_mapping=None,
    title_text="Model Explanation - SHAP Values"
):
    """
    Giải thích mô hình PyTorch bằng Deep SHAP.
    
    Args:
        model: PyTorch model đã huấn luyện.
        X_train_scaled: Dữ liệu huấn luyện đã chuẩn hóa.
        X_test_scaled: Dữ liệu kiểm tra đã chuẩn hóa.
        feature_cols: Danh sách tên các biến đặc trưng.
        device: device dùng để tính toán (cpu/cuda).
        num_background: số mẫu dùng làm background cho SHAP.
        num_samples: số mẫu để giải thích.
        feature_name_mapping: dict ánh xạ tên biến sang tên hiển thị (tùy chọn).
        title_text: Tiêu đề tự do hiển thị trên biểu đồ.
    """
    model.eval()
    
    background = torch.tensor(X_train_scaled[:num_background], dtype=torch.float32).to(device)
    explainer = shap.DeepExplainer(model, background)
    
    samples = torch.tensor(X_test_scaled[:num_samples], dtype=torch.float32).to(device)
    shap_values = explainer.shap_values(samples)
    
    # Chuyển sang numpy array
    if isinstance(shap_values, list):
        shap_values_np = shap_values[0]
    else:
        shap_values_np = shap_values
        
    if isinstance(shap_values_np, torch.Tensor):
        shap_values_np = shap_values_np.cpu().detach().numpy()
    
    if shap_values_np.ndim == 3:
        shap_values_np = shap_values_np.squeeze(-1)
    
    # Chuyển tên biến nếu có mapping
    if feature_name_mapping:
        feature_names_vn = [feature_name_mapping.get(col, col) for col in feature_cols]
    else:
        feature_names_vn = feature_cols
    
    # Cấu hình font tiếng Việt, kích thước
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    
    # Vẽ biểu đồ SHAP
    shap.summary_plot(
        shap_values_np,
        X_test_scaled[:num_samples],
        feature_names=feature_names_vn,
        max_display=20,
        show=False
    )
    
    # Thêm tiêu đề tự do phía trên
    plt.gcf().text(
        0.5, 0.98, title_text, 
        fontsize=16, color='crimson', 
        fontweight='bold', ha='center', va='top'
    )
    plt.show()


if __name__ == "__main__":
    
    df = df_dummies[df_dummies['code_type_local'] == 2]

    df.drop(columns=['code_type_local' , 'prix_m2_iris'],inplace = True)
    
    print("DataFrame mẫu được tạo:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]}...\n")
    
    # Huấn luyện mô hình
    model, scaler_X, scaler_y, X_train_scaled, X_test_scaled, feature_cols, device = \
        train_model_optim_gpu(df, epochs=300)
    
    # Giải thích mô hình bằng Deep SHAP
    shap_values = explain_model_deep_shap(
        model, 
        X_train_scaled, 
        X_test_scaled, 
        feature_cols,
        device,
        num_background=100,
        num_samples=50
    )
```

![Sử dụng Deep SHAP để giải thích mô hình dự báo giá nhà](images/deepshap.png)


# 7. Kết luận

Sự kết hợp giữa **Genetic Algorithm (GA)** và **SHAP** mở ra một hướng đi mới cho tối ưu hóa có thể giải thích (**Explainable Optimization**) — nơi mà mô hình không chỉ tìm ra kết quả tốt nhất, mà còn cho chúng ta biết lý do vì sao kết quả đó tốt.

GA mang lại khả năng tìm kiếm toàn cục mạnh mẽ trong những không gian thiết kế phức tạp, phi tuyến và không khả vi — điều mà các phương pháp cổ điển như Gradient Descent khó đạt được. Trong khi đó, SHAP và các biến thể của nó như Tree SHAP hay Kernel SHAP đóng vai trò “giải phẫu” mô hình AI, giúp làm rõ mức độ ảnh hưởng của từng yếu tố đầu vào đến kết quả tối ưu mà GA tìm được. Nhờ đó, hệ thống GA + SHAP không chỉ tối ưu được thiết kế, cấu hình hoặc chiến lược, mà còn cung cấp cái nhìn sâu sắc, minh bạch và đáng tin cậy cho các nhà kỹ sư, nhà khoa học và chuyên gia ra quyết định.

Từ việc tối ưu khí động học xe đua, thiết kế turbine gió, lựa chọn danh mục đầu tư, đến phát triển thuốc và vật liệu mới, sự kết hợp giữa sức mạnh tiến hóa của GA và khả năng giải thích của SHAP đang trở thành xu hướng tất yếu trong thế hệ AI kỹ thuật và khoa học ứng dụng hiện đại — nơi hiệu quả, hiểu biết và minh bạch đi cùng nhau.