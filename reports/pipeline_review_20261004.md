**Đánh giá pipeline — 04/10/2026**

Khuyến nghị giữ B v1.2 làm baseline, ưu tiên giảm chi phí batch và lỗi căn dòng; sau đó thử glossary pháp lý theo HY_TERMS. Kiến trúc tách bước, roundtrip nguồn, trace, cache và replay đã tạo nền tốt để làm ablation. Chưa có bằng chứng một thay đổi mới sẽ tăng bao nhiêu BLEU hoặc giảm bao nhiêu thời gian GPU.

Đã đọc code inference, recovery, selection, config và các báo cáo A/B/C; chạy audit CPU trên CSV public test nguyên bản, 50 tài liệu × 2 chiều. Không chạy thêm inference GPU hay chấm lại các prediction A/B/C vì các output đầy đủ không có trong workspace này. Số BLEU dưới đây lấy từ báo cáo đã lưu.

| Hệ thống | BLEU EN→VI | BLEU VI→EN | Thời gian EN→VI / VI→EN |
|---|---:|---:|---|
| A: gom + marker | 68,70 | 46,37 | 5,53 h / 3,10 h |
| B v1.1: gom + newline | 68,90 | 46,62 | 4,57 h / 2,89 h |
| C: từng node | 67,69 | 45,23 | 4,35 h / 2,90 h |
| B v1.2: recovery từng node | 68,90 | 46,61 | Chưa đo GPU; replay |

Nguồn: [A/B/C](public_test_abc_20261004.json), [v1.2 replay](public_test_b_v12_replay_20261004.json). B−C = +1,21 / +1,39 BLEU; khoảng bootstrap 95% đều lớn hơn 0. B−A ở VI→EN = +0,25 với khoảng [−0,05; +0,65], nên chưa đủ bằng chứng B tăng chất lượng chiều này so với A. Mỗi biến thể chỉ chạy một lần trên các máy Kaggle khác nhau. v1.2 giảm unresolved từ 7/1 xuống 2/0 unit, nhưng không tăng BLEU và chưa có số runtime thật.

**Các phát hiện mới từ audit local**

- CSV MD5 khớp `d86e055d3f84436ae07e6c27fe686bcc`. Tokenizer SHA khớp `b475bbef…`; SHA256 của file `model.safetensors` khớp A1 Kaggle `5dc305ae…`. Giá trị `6c35a145…` trong smoke local là hash tổng hợp thư mục model theo `core/hashing.py`, không phải hash file weights.
- Parser roundtrip 100/100; 0 unit không vừa budget. B có đúng 2.213 / 2.296 unit lượt đầu, khớp báo cáo GPU. Điều này kiểm tính toàn vẹn nguồn, không chứng minh output dịch đúng hay căn dòng đúng.
- Prompt p50/p95: 241/491 token EN→VI và 370/740 token VI→EN. Độ dài prompt biến thiên từ 25/28 đến 792 token. Trung vị là 8 node/unit, nên tăng source_cap đơn thuần có thể không giải quyết giới hạn số node.
- Mô phỏng lượt đầu theo thứ tự tài liệu/unit, batch 4: token hữu ích chiếm 70,5% / 73,0% tensor padded. Sắp theo độ dài đưa tỷ lệ này lên 99,8% / 99,9%. Đây là proxy cho padding input, không phải trace batch thật: chưa tính recovery, cache, scheduler hay decode. Không suy ra tăng tốc GPU 27–29% từ con số này.
- `Quyen_dan_su/390`: EN 11.712 ký tự, VI 23.277 ký tự. EN kết thúc ở chữ ký, VI còn các biểu mẫu. `Vi_pham_hanh_chinh/419` có dấu hiệu tương tự. Điểm thấp tại đây cần được phân tích với phạm vi source/reference; không dùng reference để thêm phần không có trong source.
- Smoke local chỉ có 1 tài liệu, batch 1, config A; môi trường Transformers 5.18.0 khác Kaggle 5.8.1. Smoke này không đo hiệu suất B hiện tại. `reports/local_public_test_estimate.json` được IDE nhắc tới nhưng không tồn tại trên đĩa tại thời điểm audit.

Chi tiết máy đọc được: [source audit](pipeline_review_source_audit_20261004.json).

**Thứ tự cải tiến đề xuất**

| Ưu tiên | Thay đổi | Mục tiêu và kiểm chứng |
|---|---|---|
| 1 | Scheduler gom request theo prompt_tokens và output budget; thử batch 4 rồi 8 trên GPU thực tế | Giảm padding và chờ request dài. Giữ request_id và quan hệ recovery; đo end-to-end, OOM, BLEU và cấu trúc vì thay batch fp16 có thể đổi token. |
| 2 | Packing theo rủi ro: tránh trộn heading/label ngắn với prose dài; bảng theo hàng hoặc nhóm nhỏ | Giảm lệch dòng và lượt dịch lại mà giữ context. Thử max_nodes 4/6/8; table max_nodes 4/8/16 riêng, mỗi lần một yếu tố. |
| 3 | Glossary-only trên bản sao config B: source_context=false, soft_glossary=true | Dùng HY_TERMS mà A1 đã học; 4–8 thuật ngữ thực sự có trong nguồn, lấy từ train/glossary đã xác minh. Ưu tiên VI→EN và thuật ngữ pháp lý/cách diễn đạt nhất quán. |
| 4 | Recovery có context nguồn cho node đích, hoặc nhóm nhỏ với prompt căn dòng đã huấn luyện | Giữ context ở khoảng 21% unit đang phải dịch lại. So trực tiếp với v1.2 từng node; bảo đảm model chỉ xuất node đích và không lặp phần context. |
| 5 | Retry có chọn lọc cho lỗi nội dung; selection ưu tiên coverage và mức nghiêm trọng | Hiện loop, số, label, thiếu độ dài đều chỉ warning; selection chỉ đếm hard, hòa giữ bản đầu. Đánh giá precision của từng warning trên dev trước khi kích hoạt retry để tránh dịch lại hàng loạt. |
| 6 | SFT bổ sung khớp format suy luận nếu lỗi căn dòng vẫn cao | Chỉ dùng train, các nhóm node có alignment target đáng tin; target có đúng số dòng/nhãn, có heading/bảng/chữ ký. Không suy ra target alignment bằng cách ghép dòng source/reference một cách máy móc. |

`HFBackend._generate_group` hiện dùng budget lớn nhất cho toàn batch rồi cắt theo budget riêng sau generate. Request có budget nhỏ và chưa EOS có thể sinh phần sẽ bị bỏ đi. Length bucketing là bước đầu ít phức tạp; tiếp theo có thể thêm stop theo request hoặc backend continuous batching. Model card Tencent có hướng dẫn vLLM/SGLang, nhưng vẫn cần kiểm phiên bản, GPU, prompt/decode parity và chạy benchmark thực tế trước khi đổi backend.

Recovery B v1.1 chiếm 3.609/5.822 và 3.860/6.156 request logic (~62%). Không diễn giải thành 62% thời gian GPU: các counter recovery có cả cache hit, còn requests_sent chỉ đếm request gửi. Từ báo cáo suy ra 602/861 cache hit. Peak allocated VRAM là khoảng 3,94 GiB trên T4; chưa đủ để khẳng định batch 8/16 sẽ vừa mọi workload, đặc biệt GPU laptop local.

Hai bẫy config: `do_sample` bị backend hardcode false; chuỗi `output_budget` trong JSON chỉ mô tả công thức, không được thực thi. `prompt_style=delimiter` chỉ có tác dụng khi unit có marker, nên đổi trường này trên B không tạo ablation prompt mới. Muốn thử sampling, beam hoặc budget khác phải triển khai và ghi lại decode/config thực tế.

Preset source_context/glossary hiện có còn dùng marker mặc định. Để ablation sạch, bắt đầu từ config B và chỉ bật feature đang thử. Source-context-only cũng đang đi tuần tự trong từng doc ở `doc_job.flow`; có thể xem xét song song khi evidence chỉ dựa vào nguồn tĩnh. Target history có phụ thuộc thứ tự và cần giữ dependency.

**Benchmark cho vòng tiếp theo**

1. Đo baseline B v1.2 trên đúng model/GPU/env mục tiêu; khóa weights/tokenizer/template hash, phiên bản thư viện, batch và decode. Dùng work dir mới để đo cold run, báo riêng warm-cache run. Không lấy runtime replay hoặc smoke 1 request làm dự báo full test.
2. Chọn tham số trên dev tách theo tài liệu. Public test dùng xác nhận sau khi chốt; kiểm train/dev/test không trùng tài liệu. Chạy từng ablation trước khi kết hợp.
3. Giữ headline corpus BLEU raw: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`. Không lấy trung bình BLEU từng doc làm headline; không tự normalize reference. Ghi raw/NFC/text riêng khi cần chẩn đoán.
4. Báo riêng EN→VI, VI→EN: BLEU, chrF/chrF++, BP, hyp/ref token ratio, domain scores, 0 missing output, unresolved, số unit cần recovery, request gửi/cache hit, runtime và VRAM. Resample cặp theo tài liệu cho CI; runtime cùng máy, tối thiểu lặp lại finalist để thấy biến động.
5. Giữ đủ 50 doc trong headline. Có thể thêm kết quả chẩn đoán trên tập bỏ các ca lệch phạm vi đã xác minh, công bố rõ danh sách và lý do; không thay thế điểm benchmark nguyên bản. Script evaluator `benchmark/eval_markdown.py` nằm ngoài checkout này; nên đưa evaluator/version/hash vào bundle tái lập.

Điều kiện chọn: 0 missing; unresolved không tăng so với baseline v1.2; thay đổi nhằm tăng BLEU cần CI của delta đủ thuyết phục trên dev/held-out; thay đổi nhằm tăng tốc cần ngưỡng suy giảm chất lượng định trước. Thu nhỏ budget chỉ sau khi đo output/source token ratio và no_eos theo chiều/kind; cap lớn không gây chậm cho request đã EOS sớm.

Nguồn kỹ thuật tham khảo: [SacreBLEU — signature và paired significance tests](https://github.com/mjpost/sacrebleu), [Tencent Hy-MT2-1.8B — prompt, inference và deployment](https://huggingface.co/tencent/Hy-MT2-1.8B/blob/main/README.md). Các đề xuất tăng tốc/tăng BLEU ở trên là suy luận từ code và số liệu local; chưa phải kết quả ablation mới.
