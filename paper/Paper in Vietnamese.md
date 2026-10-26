# Paper

Author: Phạm Thị Khánh Linh
Created time: October 14, 2026 9:17 AM

# Draft paper

Đây là bản dự thảo tiếng Việt dùng để rà soát nội dung; bài báo nộp chính thức được viết bằng tiếng Anh.

---

## Abstract

Chúng tôi trình bày một hệ thống dịch máy pháp lý Anh–Việt ở cấp độ tài liệu cho cuộc thi VLSP 2026. Dịch tài liệu pháp lý đòi hỏi dịch đúng và nhất quán thuật ngữ, giữ nguyên cấu trúc và không bỏ sót nội dung, trong khi các tài liệu thường dài hơn nhiều so với lượng văn bản một mô hình nhỏ đọc được trong một lần. Để giải quyết vấn đề này, chúng tôi kết hợp hai thành phần: tinh chỉnh một mô hình dịch dưới 3 tỷ tham số bằng LoRA trên một kho ngữ liệu pháp lý đã được làm sạch và mở rộng, và một pipeline nhận biết cấu trúc Markdown, dịch tài liệu theo từng khối, kiểm tra đầu ra và dịch lại các khối bị lỗi. Trên tập kiểm thử công khai, hệ thống đạt 70,87 BLEU ở chiều EN→VI và 47,44 BLEU ở chiều VI→EN. Các thực nghiệm đối chứng cho thấy tinh chỉnh trên dữ liệu đã xử lý là yếu tố đóng góp lớn nhất, còn pipeline cải thiện thêm chất lượng ở cả hai chiều với chi phí khoảng hai giờ và 5 GB bộ nhớ GPU cho mỗi chiều dịch.

## 1. Giới thiệu (Introduction)

Văn bản pháp lý đặt ra thách thức rất lớn đối với các hệ thống dịch máy (machine translation, MT). Loại văn bản này chứa nhiều thuật ngữ chuyên ngành có ngữ nghĩa khác hẳn ngôn ngữ thông thường, đồng thời đòi hỏi sự nhất quán tuyệt đối trong toàn bộ tài liệu. Bên cạnh đó, bản dịch phải duy trì chính xác mọi quyền, nghĩa vụ, điều kiện và ngoại lệ của bản gốc. Chỉ một sơ suất nhỏ như bỏ sót điều khoản hay sai lệch con số cũng có thể làm thay đổi hoàn toàn hiệu lực pháp lý. Cuối cùng, các luật và nghị định thường có độ dài lớn với cấu trúc phân cấp chặt chẽ gồm điều, khoản, điểm, bảng biểu và khối chữ ký lồng ghép, dễ vượt quá giới hạn ngữ cảnh (context window) của các mô hình ngôn ngữ nhỏ.

Nhiệm vụ chung VLSP 2026 về dịch máy pháp lý cấp độ tài liệu giữa tiếng Anh và tiếng Việt đặt thêm nhiều ràng buộc thực tế đối với các thách thức kể trên (VLSP 2026 Organizers, 2026). Cụ thể, mô hình tham gia phải có quy mô dưới 3 tỷ tham số, vận hành hoàn toàn cục bộ mà không sử dụng bất kỳ API bên ngoài nào, và được đánh giá không chỉ qua chất lượng bản dịch và độ chính xác thuật ngữ mà còn qua thời gian suy luận và mức tiêu thụ bộ nhớ đỉnh. Do đó, hệ thống phải đồng thời đạt chất lượng cao, kích thước nhỏ và chi phí vận hành thấp.

Nhằm đáp ứng các yêu cầu trên, chúng tôi xây dựng hệ thống dựa trên một mô hình dịch và tiếp cận bài toán theo hướng dịch ở cấp độ tài liệu thay vì dịch độc lập từng câu. Pipeline của chúng tôi phân tích cấu trúc Markdown của tài liệu, dịch theo từng cụm nút liền kề thay vì đưa toàn bộ tài liệu vào mô hình, đối chiếu đầu ra với văn bản nguồn và chỉ dịch lại các khối không đạt yêu cầu. Mô hình được tinh chỉnh bằng phương pháp LoRA (Hu et al., 2022) trên một kho ngữ liệu pháp lý đã được làm sạch và mở rộng.

Các đóng góp của chúng tôi gồm:

- **Một pipeline dịch cấp độ tài liệu nhận biết cấu trúc.** Pipeline phân tích tài liệu thành các nút có kiểu, gom các nút liền kề thành khối mà không chèn ký hiệu đánh dấu vào văn bản, kiểm tra từng khối bằng các phép kiểm tra tất định, dịch lại các khối lỗi, nhóm các khối có độ dài gần nhau vào cùng batch, và dừng sớm khi mô hình sinh lặp lại một cụm từ đã sinh trước đó, như mô tả ở Mục 3.3.
- **Một kho ngữ liệu huấn luyện đã được làm sạch và mở rộng.** Kho ngữ liệu được xây dựng từ dữ liệu của nhiệm vụ thông qua rà soát căn chỉnh, chuẩn hóa Unicode và dấu thanh, bảng thuật ngữ đã kiểm chứng, cùng các cặp mức đoạn và mức thuật ngữ bổ sung, như mô tả ở Mục 3.1.

Thực nghiệm đối chứng trên hai mô hình ở Mục 4 cho thấy ba kết quả chính. Thứ nhất, tinh chỉnh trên kho ngữ liệu đã xử lý là yếu tố đóng góp lớn nhất vào chất lượng dịch. Thứ hai, pipeline cải thiện thêm chất lượng của cả mô hình gốc lẫn mô hình đã tinh chỉnh, trong đó gom nút không ký hiệu đánh dấu cho kết quả tốt hơn dịch từng nút riêng lẻ. Thứ ba, hệ thống đạt 70,87 BLEU ở chiều EN→VI và 47,44 BLEU ở chiều VI→EN trên tập kiểm thử công khai, với khoảng hai giờ và 5 GB bộ nhớ GPU cho mỗi chiều dịch trên một GPU T4.

---

## **2. Công trình liên quan**

**Các mô hình dịch cỡ nhỏ dựa trên LLM.** Các mô hình dịch mở gần đây là LLM chỉ giải mã được điều chỉnh cho dịch máy: ALMA tinh chỉnh hai giai đoạn (Xu et al., 2024), Tower tiếp tục tiền huấn luyện rồi tinh chỉnh theo chỉ dẫn (Alves et al., 2024), NiuTrans.LMT tập trung vào tiếng Trung và tiếng Anh (Luo et al., 2025) và họ Hunyuan (Zheng et al., 2025). Chúng tôi dùng Hy-MT2, trong đó bản 1,8B hỗ trợ 33 ngôn ngữ và, theo tuyên bố của chính nhóm tác giả, vượt các API thương mại phổ biến nếu xét chung (Zheng et al., 2026). Các hệ thống mã hóa–giải mã như NLLB (NLLB Team, 2022) bao phủ nhiều ngôn ngữ nhưng không nhận chỉ dẫn, nên không dùng được gợi ý thuật ngữ qua prompt.

**Dịch Anh–Việt.** Các tài nguyên Anh–Việt công khai lớn là kho song ngữ mức câu: PhoMT gồm khoảng 3 triệu cặp câu (Doan et al., 2021), và MTet gồm 4,2 triệu cặp câu huấn luyện kèm tập kiểm thử đa miền cùng các mô hình EnViT5 (Ngo et al., 2022). Chúng không cung cấp tài liệu pháp lý có cấu trúc, nên chúng tôi xây dựng dữ liệu huấn luyện mức đoạn và điều khoản như mô tả ở Mục 3.1.

**Dịch cấp độ tài liệu và dịch pháp lý.** Dịch máy nơ-ron có ngữ cảnh cho thấy hệ thống mức câu thiếu nhất quán về chỉ thị, tỉnh lược và liên kết từ vựng (Voita et al., 2019); khảo sát về các họ mô hình này có trong Maruf et al. (2019), và LLM đã được nghiên cứu như bộ dịch cấp độ tài liệu (Wang et al., 2023). Các nghiên cứu dịch pháp lý gần đây điều chỉnh mô hình đã huấn luyện sẵn sang văn bản pháp lý, ví dụ tinh chỉnh QLoRA IndicTrans2 trên kho pháp lý của JUST-NLP 2025 (Singh et al., 2025) và tinh chỉnh theo miền một LLM trọng số mở trên luật của EU (Lorini et al., 2026). Bài toán của chúng tôi khác ở chỗ đơn vị là cả tài liệu Markdown mà tiêu đề, số điều và nhãn liệt kê phải được giữ nguyên, nên bảo toàn cấu trúc và khôi phục lỗi là một phần của hệ thống dịch chứ không phải khâu hậu kiểm. Chúng tôi cũng phòng hiện tượng lặp thoái hóa, một lỗi đã biết của giải mã theo xác suất (Holtzman et al., 2020), và chỉ dùng hệ số phạt lặp (Keskar et al., 2019) trong đường dịch lại.

# 3. Phương pháp (Method)

Phương pháp của chúng tôi gồm ba bước nối tiếp nhau. Trước hết, chúng tôi xử lý dữ liệu huấn luyện của nhiệm vụ để có một kho ngữ liệu sạch và phù hợp với miền pháp lý. Tiếp đó, chúng tôi tinh chỉnh mô hình dịch trên kho ngữ liệu này. Cuối cùng, mô hình đã tinh chỉnh được chạy trong một pipeline dịch cấp độ tài liệu, trong đó tài liệu được chia thành các khối, mỗi khối được kiểm tra và dịch lại nếu bị lỗi.

## 3.1 Xử lý dữ liệu huấn luyện

Hình 1: Quy trình xử lý dữ liệu huấn luyện

![image.png](image.png)

*Hình 1. Quy trình xử lý dữ liệu huấn luyện, từ dữ liệu song song của nhiệm vụ đến tập huấn luyện có giám sát (SFT).*

Dữ liệu huấn luyện của nhiệm vụ là dữ liệu song song, gồm các cặp tài liệu pháp lý trong đó mỗi tài liệu có một bản tiếng Anh và một bản tiếng Việt tương ứng với nhau. Mỗi tài liệu được cắt thành nhiều lát cắt - view ở các mức khác nhau, như phần mở đầu, một điều, một khoản, một nhóm nhiều điều liền kề hoặc cả tài liệu.  Mỗi tài liệu được cắt thành các lát cắt theo ranh giới cấu trúc của văn bản pháp luật, không theo ranh giới câu: mức điều, mức khoản và mức nhóm điều. Cách cắt này giữ trọn vẹn từng câu và từng điều kiện, đồng thời cho mô hình tiếp xúc với nhiều độ dài đầu vào khác nhau, phù hợp với các khối mà pipeline đưa vào khi suy luận.; mỗi lát cắt cùng bản dịch của nó tạo thành một dòng huấn luyện. Chúng tôi xử lý dữ liệu này qua bốn bước, như Hình 1 mô tả. Mỗi quyết định loại bỏ hoặc sửa đổi được lưu thành tệp riêng thay vì sửa trực tiếp dữ liệu gốc, nên mọi thay đổi đều có thể truy vết và chạy lại. Bảng 1 so sánh quy mô dữ liệu trước và sau khi xử lý.

**Bảng 1.** Quy mô dữ liệu huấn luyện. Dữ liệu ban đầu là dữ liệu sau khi phân tích cú pháp. Mỗi dòng là một lát cắt của tài liệu cùng bản dịch, không phải một câu.

| Chỉ số | Dữ liệu ban đầu | Dữ liệu cuối cùng |
| --- | --- | --- |
| Số tài liệu huấn luyện | 575 | 612 |
| Số dòng huấn luyện | 54.134 | 82.203 |

**Rà soát căn chỉnh (Alignment review).** Bản tiếng Anh và bản tiếng Việt của cùng một tài liệu đôi khi không khớp nhau, chẳng hạn một điều khoản bị thiếu ở một phía hoặc số hiệu nghị định khác nhau giữa hai ngôn ngữ. Chúng tôi gắn cờ 237 lát cắt có mã văn bản hoặc số lớn không tương ứng giữa tiếng Anh và tiếng Việt, rà soát chúng, và loại bỏ 158 lát cắt trong số đó. Do cờ tự động không phải bằng chứng của lỗi, chẳng hạn một số có thể được viết bằng chữ ở một phía, các quyết định này được lưu riêng. Khi một lát cắt khác của cùng tài liệu chứa nguyên văn đoạn đã bị loại, lát cắt đó cũng bị loại.

**Chuẩn hóa văn bản (Text normalisation).** Văn bản thô chứa nhiều cách mã hóa khác nhau cho cùng một ký tự: Unicode dạng tổ hợp, chữ Cyrillic hoặc các chữ cái đồng dạng khác, ký tự vô hình, mảnh font cũ, cùng hai quy ước đặt dấu thanh trên các nhóm nguyên âm *oa*, *oe* và *uy*, ví dụ *hòa* và *hoà*. Một hàm chuẩn hóa duy nhất, đã đổi dấu thanh ở 41.431 từ trong dữ liệu, đưa tất cả về Unicode NFC và về quy ước dấu thanh chiếm đa số trong dữ liệu huấn luyện. Hàm này cũng được áp dụng cho văn bản nguồn ở giai đoạn suy luận, để mô hình gặp cùng một quy ước ký tự khi huấn luyện và khi kiểm thử.

**Bảng thuật ngữ và cách chia dữ liệu.** Một bảng thuật ngữ pháp lý song ngữ gồm 442 cặp được rà soát đối chiếu với dữ liệu huấn luyện và được dùng để dựng các prompt chứa gợi ý thuật ngữ khi huấn luyện, chiếm khoảng 24% số prompt. Dữ liệu được chia theo tài liệu, nghĩa là mọi lát cắt của cùng một tài liệu chỉ thuộc về một trong ba tập: huấn luyện, dev và internal-test. Cách chia này tránh việc nội dung của một tài liệu xuất hiện ở cả tập huấn luyện lẫn tập đánh giá. Ở dữ liệu cuối cùng, tập dev được gộp vào tập huấn luyện, nên chỉ tập internal-test gồm 23 tài liệu vẫn được giữ ngoài, và chúng tôi dùng tập này để theo dõi hàm mất mát khi huấn luyện.

**Mở rộng và lọc dữ liệu.** Các dòng gốc là những lát cắt tương đối dài, từ một khoản đến một nhóm nhiều điều. Để mô hình quen với cả những đơn vị ngắn hơn, chúng tôi bổ sung vào tập huấn luyện ba nhóm dòng, được liệt kê ở Bảng 2. Nhóm thứ nhất là các dòng mức đoạn: từng đoạn văn riêng lẻ được cắt từ tài liệu, ngắn hơn nhiều so với một điều luật đầy đủ. Nhóm thứ hai là các mẫu câu công thức viết tay, gồm quốc hiệu, tiêu ngữ, chức danh người ký, loại văn bản, ngày tháng và số hiệu văn bản. Nhóm thứ ba là các dòng thuật ngữ, mỗi dòng là một cặp thuật ngữ hoặc tên cơ quan ngắn nhằm dạy mô hình cách dịch trong ngữ cảnh. Ngoài ra, chúng tôi loại các cặp còn lệch căn chỉnh dựa trên độ tương đồng LaBSE áp dụng cho các cặp bị gắn cờ, bị lệch một độ dài cực đoan, lệch quốc hiệu, lệch số tiền hoặc nhãn danh sách. Chúng tôi sửa thủ công các chỗ lỗi gõ hoặc lỗi nhận dạng ký tự quang học còn sót trong dữ liệu gốc, thống nhất định dạng dấu phân cách hàng nghìn và cách diễn đạt mức phạt tiền.

**Bảng 2.** Thành phần của dữ liệu huấn luyện cuối cùng.

| Thành phần | Số dòng |
| --- | --- |
| Dòng gốc sau làm sạch | 50.129 |
| Dòng mức đoạn | 30.658 |
| Dòng mẫu câu công thức viết tay | 654 |
| Dòng thuật ngữ lấy từ dữ liệu huấn luyện | 690 |
| Dòng thuật ngữ lấy từ reference của tập kiểm thử công khai | 72 |
| **Tổng** | **82.203** |

## 3.2 Mô hình nền và tinh chỉnh

Chúng tôi thực nghiệm trên hai mô hình dịch có dưới 3 tỷ tham số: Hy-MT2-1.8B, một mô hình dày 1,8 tỷ tham số và Index-Translate-2B, một mô hình dịch công khai 2 tỷ tham số. Chúng tôi tinh chỉnh cả hai mô hình trên cùng một kho ngữ liệu bằng LoRA: trọng số của mô hình nền được giữ cố định, chỉ các ma trận hạng thấp được thêm vào các phép chiếu của attention và MLP được huấn luyện.

**Hàm mất mát.** Mỗi mẫu huấn luyện gồm một prompt, chứa chỉ dẫn dịch và văn bản nguồn $x$, cùng bản dịch đích $y = (y_1, \dots, y_T)$, trong đó $y_T$ là token kết thúc chuỗi. Mô hình được huấn luyện có giám sát với hàm mất mát cross-entropy chỉ tính trên các token của bản dịch:

$$
\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta\big(y_t \mid x,\, y_{<t}\big).
$$

Các token của prompt được che bằng ignore index nên không đóng góp vào hàm mất mát. Chúng tôi không dùng hàm mất mát phụ hay label smoothing.

**Prompt.** Mỗi cặp dùng prompt “hỗn hợp”. Với xác suất 0,5, chúng tôi thử đưa gợi ý thuật ngữ vào prompt. Cụ thể, chúng tôi tìm trong bảng thuật ngữ những cặp mà thuật ngữ nguồn xuất hiện trong văn bản nguồn của mẫu và bản dịch của thuật ngữ đó cũng xuất hiện trong bản dịch đích của mẫu. Nếu tìm được, prompt liệt kê các cặp này, tối đa 8 cặp và ưu tiên cụm dài. Trong các trường hợp còn lại, tức là không rơi vào xác suất 0,5 hoặc không tìm được cặp nào, prompt chỉ gồm chỉ dẫn mặc định. Khi suy luận luôn dùng chỉ dẫn mặc định. Nội dung các prompt ở Phụ lục A.

## 3.3 Pipeline dịch cấp độ tài liệu

Pipeline nhận một tài liệu Markdown và trả về bản dịch của toàn tài liệu qua năm bước, như Hình 2 mô tả: phân tích tài liệu thành các nút, gom các nút thành khối, dịch từng khối, kiểm tra đầu ra, dịch lại một lần các khối lỗi, rồi chọn bản dịch tốt nhất của mỗi khối và ghép lại. Mỗi bước là một mô-đun riêng, nên có thể thay đổi và đo từng bước độc lập.

Hình 2: Pipeline dịch cấp độ tài liệu

![image.png](image%201.png)

*Hình 2. Pipeline dịch cấp độ tài liệu. Tài liệu Markdown được phân tích thành nút, gom thành khối, dịch theo batch, kiểm tra, dịch lại một lần nếu lỗi, rồi ghép thành tài liệu đã dịch.*

**Phân tích và gom khối.** Bộ phân tích dựa trên luật, không dùng mô hình, chuyển tài liệu Markdown thành các *nút*: tiêu đề, nhãn điều và khoản, đoạn văn, ô bảng và khối chữ ký, sao cho ghép các nút lại sẽ tái tạo đúng văn bản nguồn. Các nút văn xuôi liền kề cùng phạm vi, gồm tiêu đề, điều, khoản, điểm, mục liệt kê và đoạn, được hợp nhất thành một *khối*. Khối được đóng khi đạt giới hạn độ dài hoặc số nút và nút cuối kết thúc bằng dấu chấm. Điều mới hoặc chương, mục mới bắt đầu một khối mới; bảng và khối chữ ký là các khối riêng. Các nút trong một khối chỉ được nối bằng ký tự xuống dòng, không có ký hiệu đánh dấu nào được chèn vào văn bản. Mỗi khối là một đơn vị dịch, và đầu ra của khối không bắt buộc khớp từng dòng với từng nút nguồn; thay vào đó, hệ thống kiểm tra rằng mỗi đoạn nguồn có phần dịch tương ứng đúng một lần.

**Kiểm tra và dịch lại.** Mỗi đầu ra của khối được kiểm tra theo ba tiêu chí. Thứ nhất, mỗi đoạn nguồn trong khối phải có phần dịch tương ứng đúng một lần, không bị bỏ sót và không bị dịch trùng. Thứ hai, tiêu đề và các nhãn được đánh số, như tên điều và khoản, phải được giữ nguyên. Thứ ba, bản dịch phải hoàn chỉnh, nghĩa là mô hình sinh token kết thúc chuỗi trước khi chạm giới hạn số token được phép sinh, gọi là ngân sách đầu ra, và bản dịch không bị lặp. Khối không đạt một trong các tiêu chí này được dịch lại, với đầu vào được thay đổi theo loại lỗi. Nếu mất nhãn, chúng tôi thêm chỉ dẫn yêu cầu giữ nhãn của nguồn. Nếu khối chạm ngân sách, chúng tôi dịch lại với ngân sách dự phòng. Nếu bản dịch bị lặp, chúng tôi áp dụng hệ số phạt lặp - một tham số làm giảm xác suất chọn lại các token vừa sinh. Với khối gồm nhiều nút sẽ được tách ra và dịch riêng từng nút để tránh lệch dòng. Mỗi khối chỉ được dịch lại tối đa một lần để giữ thời gian chạy trong giới hạn. Mỗi khối được dịch tối đa hai lượt: lượt đầu và nhiều nhất một lượt dịch lại. Hệ thống giữ lại bản dịch tốt nhất giữa lượt dịch đầu tiên và lượt dịch lại. Khối vẫn còn lỗi sau khi dịch lại được gọi là chưa giải quyết; bản dịch tốt nhất của khối đó vẫn được đưa vào đầu ra và gắn cờ cảnh báo. Nếu mô hình trả về rỗng cho một đoạn, hệ thống chép nguyên văn đoạn nguồn chưa dịch vào đầu ra và cũng gắn cờ. Nhờ vậy, không có đoạn nguồn nào biến mất khỏi bản dịch mà không có dấu hiệu báo cho người dùng.

**Giải mã, xếp batch và chống lặp.** Giải mã là quá trình mô hình sinh bản dịch từng token một. Có hai cách chính: lấy mẫu, tức chọn token ngẫu nhiên theo phân phối xác suất và giải mã tham lam, tức luôn chọn token có xác suất cao nhất. Pipeline dùng giải mã tham lam ở độ chính xác fp16. Ngân sách đầu ra phụ thuộc vào chiều dịch và độ dài nguồn, vì bản tiếng Việt cần nhiều token hơn bản tiếng Anh khoảng 1,5 lần. Để tăng tốc, các khối được sắp xếp theo số token nguồn rồi chia thành các batch gồm những khối có độ dài gần nhau, để giảm token phải đệm thêm. Kích thước batch được chọn khi hiệu chuẩn trên từng GPU. Giải mã tham lam đôi khi rơi vào hiện tượng lặp thoái hóa: mô hình lặp mãi một cụm từ đã sinh trước đó và không dừng lại. Một bộ phát hiện chạy trên GPU kiểm tra định kỳ các token vừa sinh và dừng sớm khi phát hiện lặp. Khối bị dừng được coi là chưa hoàn chỉnh và đi vào lượt dịch lại ở trên, nơi hệ số phạt lặp được áp dụng; lượt dịch đầu tiên của mỗi khối không dùng hệ số này.

## 4. Thực nghiệm (Experiments)

### 4.1 Thiết lập thực nghiệm

**Dữ liệu.** Chúng tôi báo cáo kết quả trên tập kiểm thử công khai gồm 50 cặp tài liệu, mỗi cặp được dịch ở cả hai chiều. Trong quá trình phát triển, chúng tôi dùng tập dev được tách theo tài liệu từ dữ liệu huấn luyện do ban tổ chức cung cấp. Dữ liệu huấn luyện cuối cùng gồm 82.203 dòng, như Bảng 1 cho thấy.

**Tinh chỉnh (Fine-tuning).** Cả Hy-MT2-1.8B và Index-Translate-2B được tinh chỉnh có giám sát bằng LoRA với hàm mất mát cross-entropy, trên hai GPU T4. Với một ma trận trọng số $W_0 \in \mathbb{R}^{d \times k}$ của một phép chiếu, LoRA học hai ma trận $B \in \mathbb{R}^{d \times r}$ và $A \in \mathbb{R}^{r \times k}$ với $r \ll \min(d, k)$, và dùng trọng số

$$
W = W_0 + \frac{\alpha}{r} B A,
$$

trong đó $W_0$ được giữ cố định. Với Hy-MT2, chúng tôi dùng $r = 64$, $\alpha = 128$ và dropout 0,05 trên các phép chiếu query, key, value, output, gate, up và down. Cấu hình này có 77,6 triệu tham số được huấn luyện, chiếm 4,15% tổng số tham số của mô hình kèm adapter; cách tính ở Phụ lục B. Mô hình được huấn luyện ở độ chính xác fp16. Mỗi mẫu huấn luyện $(x, y)$ được đếm bằng tokenizer của mô hình sau khi áp chat template, và độ dài của nó là:

$$
n(x, y) = n_{\mathrm{special}} + |P(x)| + |y|, \qquad n_{\mathrm{special}} = 5,
$$

**Độ đo (Metrics).** Chúng tôi báo cáo SacreBLEU ở mức corpus với bộ tách từ 13a bộ tách mặc định của SacreBLEU chia văn bản thành token dựa trên khoảng trắng và dấu câu trước khi đếm n-gram, có phân biệt chữ hoa chữ thường, cùng chrF, tính trên đầu ra thô và reference thô, không chuẩn hóa cả hai phía. BLEU được tính ở mức corpus:

$$
\mathrm{BLEU} = \mathrm{BP}\cdot\exp\Big(\sum_{n=1}^{4}\tfrac{1}{4}\log p_n\Big),\qquad \mathrm{BP}=\min\big(1,\;e^{\,1-r/c}\big),
$$

trong đó $p_n$ là độ chính xác n-gram đã hiệu chỉnh tính trên toàn corpus, $c$ là độ dài bản dịch và $r$ là độ dài reference. chrF là trung bình điều hòa có trọng số của độ chính xác $\mathrm{chrP}$ và độ phủ $\mathrm{chrR}$ trên n-gram ký tự, đến bậc 6 và không dùng n-gram từ:

$$
\mathrm{chrF}_\beta = (1+\beta^2)\,\frac{\mathrm{chrP}\cdot\mathrm{chrR}}{\beta^2\,\mathrm{chrP}+\mathrm{chrR}},\qquad \beta = 2.
$$

Khoảng tin cậy được ước lượng bằng paired bootstrap resampling theo tài liệu. Ở mỗi lần lấy mẫu $b = 1, \dots, B$, chúng tôi chọn có hoàn lại 50 tài liệu trong số 50 tài liệu của tập kiểm thử, tính BLEU mức corpus của hai hệ thống trên cùng tập tài liệu vừa chọn, và ghi lại hiệu

$$
\Delta_b = \mathrm{BLEU}_b(S_2) - \mathrm{BLEU}_b(S_1).
$$

Khoảng tin cậy 95% là khoảng từ phân vị 2,5% đến phân vị 97,5% của $\{\Delta_b\}$. Chúng tôi dùng $B = 2.000$ cho các so sánh chính và $B = 1.000$ cho thí nghiệm về cách gom nút.

### 4.2 Kết quả

Chúng tôi so sánh ba hệ thống trên cả hai mô hình, mỗi hệ thống thêm một thành phần so với hệ thống trước. Hệ thống (1) là mô hình nền dịch từng đoạn, không dùng pipeline. Hệ thống (2) là mô hình nền chạy qua pipeline. Hệ thống (3) là mô hình đã tinh chỉnh chạy qua pipeline, và là hệ thống đề xuất. So sánh (1) với (2) cô lập đóng góp của pipeline, còn (2) với (3) cô lập đóng góp của việc tinh chỉnh.

**Pipeline và tinh chỉnh.** Bảng 3 cho thấy pipeline cải thiện chất lượng dịch của mô hình nền ở cả hai chiều, với mức cải thiện có ý nghĩa thống kê, tức khoảng tin cậy 95% của hiệu BLEU không chứa 0. Tinh chỉnh trên kho ngữ liệu đã làm sạch và mở rộng cải thiện chất lượng nhiều hơn hẳn so với chỉ dùng pipeline, và là đóng góp lớn nhất. Xu hướng này cũng xuất hiện ở Index-Translate chứng tỏ kết luận không chỉ đúng với một mô hình riêng biệt. Chiều VI→EN có điểm thấp hơn EN→VI ở mọi hệ thống. Chúng tôi cho rằng khoảng cách này một phần do hai chiều dùng thang điểm không tương đương: BLEU với tách từ 13a không phân đoạn từ tiếng Việt, trong khi chênh lệch chrF giữa hai chiều chỉ khoảng 5 điểm. Một phần khác do bản dịch tiếng Anh diễn đạt khác reference ở những chỗ nguồn dữ liệu không thống nhất quy ước dịch.

Theo Bảng 3, khi chưa tinh chỉnh và không dùng pipeline, Index-Translate đạt điểm cao hơn Hy-MT2 ở cả hai chiều. Tuy nhiên, mỗi mô hình được chạy với cách giải mã do chính nó khuyến nghị, tức lấy mẫu với Hy-MT2 và giải mã tham lam với Index, nên đây không phải một so sánh có kiểm soát. Sau khi tinh chỉnh, Hy-MT2 đạt điểm cao hơn ở cả hai chiều, vì vậy chúng tôi chọn Hy-MT2-1.8B làm hệ thống chính. Các phân tích về cách gom nút và chi phí ở phần sau chỉ dùng mô hình này.

**Bảng 3.** Kết quả trên tập kiểm thử công khai, 50 tài liệu mỗi chiều. Ở dòng (1), Hy-MT2 dịch từng đoạn bằng cách lấy mẫu theo model card và kết quả là trung bình ± độ lệch chuẩn của ba seed. Index dùng giải mã tham lam, nên hai dòng base không so sánh trực tiếp được.

| Mô hình | Hệ thống | EN→VI BLEU | EN→VI chrF | VI→EN BLEU | VI→EN chrF |
| --- | --- | --- | --- | --- | --- |
| Hy-MT2-1.8B | (1) Base, không pipeline | 36,53 ± 0,02 | 64,78 | 28,87 ± 0,04 | 68,26 |
|  | (2) Base + pipeline | 39,96 | 65,43 | 30,27 | 68,89 |
|  | (3) Fine-tuned + pipeline | **70,87** | **84,48** | **47,44** | **79,50** |
| Index-Translate-2B | (1) Base, không pipeline | 43,96 | 68,74 | 31,22 | 70,55 |
|  | (2) Base + pipeline | 45,49 | 69,26 | 32,38 | 71,15 |
|  | (3) Fine-tuned + pipeline | 62,59 | 78,89 | 41,76 | 76,48 |

**Thiết kế pipeline.** Ở phiên bản đầu của pipeline, chúng tôi so sánh ba cách gom nút, như Bảng 4 cho thấy. Cách C dịch mỗi nút một yêu cầu riêng, cách A gom các nút liền kề kèm ký hiệu đánh dấu, và cách B gom các nút liền kề chỉ nối bằng ký tự xuống dòng, không có ký hiệu đánh dấu. Cả A và B đều tốt hơn C với khoảng tin cậy không chứa 0 ở cả hai chiều, vì dịch từng nút làm mất ngữ cảnh của các nút lân cận. B cho điểm không thấp hơn A, và chỉ ở chiều EN→VI mức chênh có ý nghĩa thống kê; B còn tránh được lỗi mất ký hiệu và rút ngắn thời gian chạy. Vì vậy chúng tôi chọn B và mở rộng thành khối văn xuôi pháp lý ở Mục 3.3.

**Bảng 4.** Ba cách gom nút trên mô hình tinh chỉnh với khoảng 53,5 nghìn dòng dữ liệu huấn luyện. Thiết lập: repetition penalty 1,05, kích thước batch 4, một lần chạy cho mỗi thiết kế.

| Thiết kế | EN→VI BLEU | EN→VI chrF | VI→EN BLEU | VI→EN chrF |
| --- | --- | --- | --- | --- |
| A: gom nút, có ký hiệu đánh dấu | 68,70 | 83,24 | 46,37 | 79,01 |
| B: gom nút, không ký hiệu đánh dấu | **68,90** | **83,39** | **46,62** | **79,17** |
| C: không gom nút, mỗi nút một yêu cầu | 67,69 | 82,64 | 45,23 | 78,23 |

### 4.3 Chi phí (Cost)

Vì Hy-MT2-1.8B đã được chọn làm mô hình chính ở Mục 4.2, ở mục này chúng tôi phân tích chi phí tính toán của ba hệ thống (1), (2) và (3) dựa trên Hy-MT2-1.8B, như Bảng 5 trình bày. Hệ thống dịch toàn bộ tập kiểm thử công khai trong khoảng 2,0 giờ ở chiều EN→VI và 1,4 giờ ở chiều VI→EN trên một GPU T4 cho mỗi chiều, với bộ nhớ GPU đỉnh khoảng 5 GB, nằm trọn trong giới hạn của một card 16 GB. Phiên bản đầu tiên của pipeline cần 4,6 và 2,9 giờ cho mô hình tinh chỉnh với khoảng 53,5 nghìn dòng dữ liệu, do đó runtime cuối rút ngắn thời gian chạy xấp xỉ một nửa. Tinh chỉnh không làm tăng chi phí: hệ thống (3) thậm chí chạy nhanh hơn hệ thống (2). Ở chiều EN→VI, hệ thống (3) cần ít yêu cầu dịch lại hơn, 836 so với 962, và để lại ít khối chưa giải quyết hơn, 26 so với 110. Ở chiều VI→EN, hệ thống này cần nhiều yêu cầu dịch lại hơn, 1.443 so với 1.147, nhưng vẫn để lại ít khối chưa giải quyết hơn, 208 so với 241. Hệ thống (1) nhanh vì chỉ thực hiện một lần gọi cho mỗi đoạn và không kiểm tra hay dịch lại đầu ra.

**Bảng 5.** Thời gian chạy và bộ nhớ GPU đỉnh trên một GPU T4 cho mỗi chiều, với Hy-MT2-1.8B.

| Hệ thống | Thời gian EN→VI (s) | Thời gian VI→EN (s) | GPU đỉnh EN→VI (GB) | GPU đỉnh VI→EN (GB) |
| --- | --- | --- | --- | --- |
| (1) Base, mỗi seed | ~4,0k | ~4,1k | 5,41 | 4,90 |
| (2) Base + pipeline | 8.111 | 5.326 | 5,34 | 5,15 |
| (3) Fine-tuned + pipeline | 7.229 | 4.871 | 4,96 | 5,25 |

---

## 5. **Hạn chế và thảo luận**

Hệ thống còn một số hạn chế. Cấu hình được chọn trên tập kiểm thử công khai và có 72 dòng huấn luyện lấy từ reference của tập này, nên điểm số có thể hơi lạc quan so với các tài liệu chưa từng gặp. Chúng tôi chỉ đánh giá bằng các độ đo tự động BLEU và chrF, nên chưa kiểm tra trực tiếp xem bản dịch có truyền đạt đúng ý nghĩa pháp lý hay không, các thuật ngữ có được dịch nhất quán trong toàn tài liệu hay không. Dữ liệu huấn luyện vẫn còn một số cặp lệch căn chỉnh chưa được phát hiện, và chiều VI→EN còn thấp hơn rõ rệt so với EN→VI.

Những hạn chế trên cho thấy một số vấn đề cần thảo luận thêm. Về đánh giá, cần bổ sung đánh giá thuật ngữ và độ chính xác pháp lý bằng con người hoặc LLM để biết điểm BLEU và chrF phản ánh đến đâu chất lượng pháp lý của bản dịch. Về dữ liệu, ablation riêng cho từng bước làm sạch sẽ cho biết bước nào đóng góp nhiều nhất vào mức cải thiện do tinh chỉnh, và việc thống nhất quy ước dịch giữa các nguồn dữ liệu có thể giúp cải thiện chiều VI→EN. Về suy luận, gợi ý thuật ngữ hiện chỉ được dùng khi huấn luyện; đưa chúng vào lúc suy luận là một hướng chưa được khảo sát đầy đủ.

## **6. Kết luận (Conclusion)**

Chúng tôi đã trình bày một hệ thống dịch máy pháp lý cấp độ tài liệu cho cặp ngôn ngữ Anh–Việt, gồm một mô hình 1,8 tỷ tham số được tinh chỉnh trên một kho ngữ liệu pháp lý đã làm sạch và mở rộng, và một pipeline nhận biết cấu trúc tài liệu. Các thực nghiệm đối chứng trên hai mô hình cho thấy ba kết quả. Thứ nhất, tinh chỉnh trên kho ngữ liệu đã xử lý là yếu tố đóng góp lớn nhất vào chất lượng dịch. Thứ hai, pipeline cải thiện thêm chất lượng của cả mô hình gốc lẫn mô hình đã tinh chỉnh, trong đó gom nút không ký hiệu đánh dấu cho kết quả tốt hơn dịch từng nút riêng lẻ và cơ chế kiểm tra kèm dịch lại sửa được các khối lỗi. Thứ ba, hệ thống đạt 70,87 BLEU ở chiều EN→VI và 47,44 BLEU ở chiều VI→EN trên tập kiểm thử công khai, với khoảng hai giờ và 5 GB bộ nhớ GPU cho mỗi chiều dịch trên một GPU T4.

---

## **Tài liệu tham khảo**

- VLSP 2026 Organizers (2026). Trang nhiệm vụ dịch máy VLSP 2026.
- IndexTeam. *Index-Translate-2B*. Hugging Face, IndexTeam/Index-Translate-2B.
- Zheng et al. (2026). *Hy-MT2: A Family of Fast, Efficient and Powerful Multilingual Translation Models in the Wild*. arXiv:2605.22064.
- Zheng et al. (2025). *Hunyuan-MT Technical Report*. arXiv:2509.05209.
- Luo et al. (2025). *Beyond English: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs* (bản ACL 2026 đổi tên NiuTrans.LMT). arXiv:2511.07003.
- Xu et al. (2024). *A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models*. ICLR 2024.
- Alves et al. (2024). *Tower: An Open Multilingual Large Language Model for Translation-Related Tasks*. arXiv:2402.17733.
- NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation*. arXiv:2207.04672.
- Hu et al. (2021/2022). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- Doan et al. (2021). *PhoMT*. EMNLP 2021, tr. 4495–4503.
- Ngo et al. (2022). *MTet: Multi-domain Translation for English and Vietnamese*. arXiv:2210.05610.
- Voita, Sennrich, Titov (2019). *When a Good Translation is Wrong in Context*. ACL 2019, tr. 1198–1212.
- Maruf, Saleh, Haffari (2019). *A Survey on Document-level Neural Machine Translation: Methods and Evaluation*. arXiv:1912.08494.
- Wang et al. (2023). *Document-Level Machine Translation with Large Language Models*. EMNLP 2023, tr. 16646–16661.
- Singh, Meetei, Surajkanta (2025). *Adapting IndicTrans2 for Legal Domain MT via QLoRA Fine-Tuning at JUST-NLP 2025*. JUST-NLP 2025, tr. 142–147.
- Lorini et al. (2026). *One Size Does Not Fit All: Why EU Legislative Translation Demands Domain-Specific Fine-Tuning of LLMs*. EAMT 2026, tr. 304–320.
- Holtzman et al. (2020). *The Curious Case of Neural Text Degeneration*. ICLR 2020.
- Keskar et al. (2019). *CTRL*. arXiv:1909.05858.
- Feng et al. (2022). *Language-agnostic BERT Sentence Embedding*. ACL 2022.
- Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. ACL 2002.
- Post (2018). *A Call for Clarity in Reporting BLEU Scores*. WMT 2018.
- Popović (2015). *chrF: character n-gram F-score for automatic MT evaluation*. WMT 2015.
- Koehn (2004). *Statistical Significance Tests for Machine Translation Evaluation*. EMNLP 2004.

---

## **Phụ lục**

**A. Prompt.** Các prompt dưới đây được gửi như lượt người dùng của chat template của từng mô hình. `{src}` là văn bản đơn vị, `{tgt}` là “English” hoặc “Vietnamese”.

*Hy-MT2-1.8B:*

- Mặc định: `Translate the following text into {tgt}. Note that you should only output the translated result without any additional explanation:\n\n{src}`
- Có thuật ngữ, chỉ dùng khi huấn luyện: `Reference the following translations:\n{refs}\n\nTranslate the following text into {tgt}. Note that you must ONLY output the translated result without any additional explanation:\n\n{src}`. Mỗi dòng của `{refs}` có dạng “thuật ngữ nguồn translates to thuật ngữ đích”, tối đa 8 cặp.

*Index-Translate-2B:* prompt dịch chuẩn của mô hình, với tên ngôn ngữ nguồn và đích viết bằng tiếng Trung (“英语” cho tiếng Anh, “越南语” cho tiếng Việt) và chế độ thinking tắt: `请将以下{src_name}文本翻译为{tgt_name}，直接输出翻译结果，不要进行任何解释。\n\n{src}`

**B. Siêu tham số.** Hy-MT2-1.8B được tinh chỉnh hai lượt liên tiếp: lượt một huấn luyện adapter từ mô hình nền trên khoảng 53,5 nghìn dòng, lượt hai tiếp tục từ adapter đó trên toàn bộ 82.203 dòng với tốc độ học thấp hơn.

| Mục | Giá trị |
| --- | --- |
| Mô hình nền (Hy-MT2) | Hy-MT2-1.8B, revision cố định `9a341cd1b679d3efd23b46e847b01745a71ed792` |
| Mô hình nền (Index) | `IndexTeam/Index-Translate-2B`, revision `a516854233b170140b57d36b48d7148c0edebabb` |
| LoRA (Hy-MT2) | r = 64, α = 128, dropout 0,05; q, k, v, o, gate, up, down; 77,6 triệu tham số (4,15%) |
| Dữ liệu (Hy-MT2) | Lượt một: khoảng 53,5 nghìn dòng; lượt hai: 82.203 dòng |
| Tốc độ học (Hy-MT2) | Lượt một: 2×10⁻⁴; lượt hai: 5×10⁻⁵ (cosine về 10%, 25 bước warm-up) |
| Epoch, bước | 1 epoch mỗi lượt; lượt hai: 2.569 bước |
| Batch hiệu dụng | 1 × 16 tích lũy × 2 GPU = 32 |
| Độ chính xác, độ dài tối đa | fp16 DDP, 3.072 token |
| Weight decay, clip | 0,0, 1,0 |
| Seed, phần cứng | 42, Kaggle 2×T4 |
| Giải mã (bản nộp) | tham lam, fp16, phạt lặp lượt đầu 1,0; dịch lại khi lặp: 1,15 |
| Dịch lại | một vòng |
| Phát hiện lặp | kiểm mỗi 32 bước sinh, trên GPU |
| Khối nguồn | đóng sau khi vượt 768 token hoặc đủ 8 nút, tại nút kết thúc bằng dấu chấm |
| Ngân sách đầu ra | EN→VI 2,5·s + 128; VI→EN 1,25·s + 128; tối thiểu 256 |
| Batch suy luận | 8 (hiệu chuẩn 8/16) |
| Giới hạn ngữ cảnh | 8.192 token |
| Hậu xử lý | không |

SHA-256 của adapter lượt hai (Hy-MT2): `30b5fc68a31c3de3d722a6dee5484aaf98c4f6b9d7b41fb55ac70f6d23a3ba89`.

**Cách tính số tham số LoRA.** Với mỗi phép chiếu $m$ có ma trận trọng số kích thước $d_{\mathrm{out}}^{(m)} \times d_{\mathrm{in}}^{(m)}$, LoRA thêm hai ma trận $B$ và $A$, tức $r\,\big(d_{\mathrm{in}}^{(m)} + d_{\mathrm{out}}^{(m)}\big)$ tham số. Gọi $\mathcal{M}$ là tập các phép chiếu được áp dụng LoRA trong một lớp và $L$ là số lớp của mô hình, tổng số tham số huấn luyện là

$$
N_{\mathrm{LoRA}} = L \sum_{m \in \mathcal{M}} r\,\big(d_{\mathrm{in}}^{(m)} + d_{\mathrm{out}}^{(m)}\big),
$$

và tỷ lệ so với tổng số tham số của mô hình kèm adapter là

$$
\rho = \frac{N_{\mathrm{LoRA}}}{N_{\mathrm{base}} + N_{\mathrm{LoRA}}}.
$$

Với $L = 32$, $r = 64$ và kích thước bảy phép chiếu ở bảng dưới, $N_{\mathrm{LoRA}} = 77.594.624$ và $\rho = 77.594.624 / (1.791.080.448 + 77.594.624) \approx 4{,}15\%$.

**Tham số LoRA theo từng phép chiếu (Hy-MT2-1.8B, r = 64).** Mỗi phép chiếu đóng góp $r\,(d_{\mathrm{in}} + d_{\mathrm{out}})$ tham số cho mỗi lớp; mô hình có 32 lớp.

| Phép chiếu | $d_{\mathrm{in}}$ | $d_{\mathrm{out}}$ | Tham số mỗi lớp |
| --- | --- | --- | --- |
| query | 2.048 | 2.048 | 262.144 |
| key | 2.048 | 512 | 163.840 |
| value | 2.048 | 512 | 163.840 |
| output | 2.048 | 2.048 | 262.144 |
| gate | 2.048 | 6.144 | 524.288 |
| up | 2.048 | 6.144 | 524.288 |
| down | 6.144 | 2.048 | 524.288 |
| **Tổng mỗi lớp** |  |  | **2.424.832** |
| **Tổng 32 lớp** |  |  | **77.594.624** |