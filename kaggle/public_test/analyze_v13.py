"""Paired document analysis of downloaded v1.3 evidence vs historical B."""
import csv
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from sacrebleu.metrics import BLEU

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/v13_kaggle_20261004'
OLD = ROOT / 'results/public_test_20261004/B_gom_node_khong_marker'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def statuses(path):
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    warnings = Counter(w for r in rows for w in set(r.get('doc_check', {}).get('warnings', [])))
    return {'documents': len(rows), 'statuses': dict(Counter(r['status'] for r in rows)),
            'warning_documents': dict(warnings),
            'hard_failure_documents': sum(bool(r.get('doc_hard_failures')) for r in rows),
            'unresolved': [{'sample_id': r['sample_id'], 'units': r['unresolved_units'],
                            'hard': r.get('selected_hard_failures', [])}
                           for r in rows if r.get('unresolved_units')],
            'roundtrip_failures': sum(not r.get('roundtrip', {}).get('ok', False) for r in rows)}

def main():
    csv.field_size_limit(2**31-1)
    with (ROOT/'kaggle/public_test/build/original/benchmark/data/en-vi-public_test.csv').open(encoding='utf-8', newline='') as f:
        docs = list(csv.DictReader(f))
    assert len(docs) == 50
    result = {'bootstrap': {'resamples': 2000, 'seed': 42, 'unit': 'paired document',
                          'metric': 'raw corpus BLEU delta', 'interval': 'percentile 95%; conditional on this test set'}, 'directions': {}}
    ids = [d['sample_id'] for d in docs]
    samples = np.random.default_rng(42).integers(0, 50, size=(2000, 50))
    for direction in ('en-vi', 'vi-en'):
        tgt = direction.split('-')[1]
        refs = [d['vietnamese_markdown' if tgt == 'vi' else 'english_markdown'] for d in docs]
        paths = [d['vietnamese_path' if tgt == 'vi' else 'english_path'] for d in docs]
        roots = {'baseline': OLD/'predictions', 'batch_length': OUT/'hyp_batch_length', 'glossary': OUT/'hyp_glossary'}
        hyps = {k: [(p/x).read_text(encoding='utf-8') for x in paths] for k,p in roots.items()}
        metric = BLEU(tokenize='13a')
        stats = {k: np.asarray(metric._extract_corpus_statistics(h, [refs]), dtype=np.int64) for k,h in hyps.items()}
        score = lambda s: metric._compute_score_from_stats(s.tolist()).score
        boot = {k: np.array([score(s) for s in a[samples].sum(axis=1)]) for k,a in stats.items()}
        sentence = BLEU(tokenize='13a', effective_order=True)
        pd = {k: np.array([sentence.sentence_score(h,[r]).score for h,r in zip(hh, refs)]) for k,hh in hyps.items()}
        entry = {'corpus_bleu': {k:score(a.sum(axis=0)) for k,a in stats.items()}, 'variants': {}, 'categories': {}}
        for k in roots:
            stpath = OLD/'logs'/f'document_status_{direction}.jsonl' if k=='baseline' else OUT/f'work_{k}_{direction}'/'logs/document_status.jsonl'
            details = statuses(stpath)
            ratios = [len(h.split())/max(1,len(r.split())) for h,r in zip(hyps[k],refs)]
            details['short_documents_under_70pct'] = [{'sample_id':sid,'ratio':ra} for sid,ra in zip(ids,ratios) if ra<.7]
            details['identical_to_baseline'] = sum(a==b for a,b in zip(hyps[k],hyps['baseline']))
            if k!='baseline':
                delta=pd[k]-pd['baseline']; bd=boot[k]-boot['baseline']
                details.update({'bootstrap_bleu_delta_95ci': np.quantile(bd,[.025,.975]).tolist(),
                                'docs_improved':int((delta>.005).sum()), 'docs_worse':int((delta<-.005).sum()),
                                'docs_equal_within_005':int((abs(delta)<=.005).sum()),
                                'mean_document_bleu_delta':float(delta.mean()),
                                'largest_improvements':[{'sample_id':ids[i], 'baseline':float(pd['baseline'][i]), 'new':float(pd[k][i]), 'delta':float(delta[i])} for i in np.argsort(delta)[-5:][::-1]],
                                'largest_regressions':[{'sample_id':ids[i], 'baseline':float(pd['baseline'][i]), 'new':float(pd[k][i]), 'delta':float(delta[i])} for i in np.argsort(delta)[:5]]})
            entry['variants'][k]=details
        for cat in sorted(set(s.split('/')[0] for s in ids)):
            indexes=[i for i,s in enumerate(ids) if s.startswith(cat+'/')]
            entry['categories'][cat]={'documents':len(indexes), 'corpus_bleu':{k:score(a[indexes].sum(axis=0)) for k,a in stats.items()}}
        result['directions'][direction]=entry
    dest=ROOT/'reports/v13_detailed_analysis.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    render_report(result)
    print(dest)

def render_report(analysis):
    gpu = read(ROOT/'reports/v13_kaggle_results.json')
    old = read(ROOT/'reports/public_test_abc_20261004.json')
    lines = ['# Đánh giá toàn diện v1.3: batching và glossary', '',
             '## Kết luận', '',
             'Nên chọn batching theo độ dài làm cấu hình mặc định tiếp theo. Thời gian dịch giảm khoảng 21% so với B v1.1 lịch sử, điểm BLEU/chrF gần như giữ nguyên. Chưa nên bật glossary hiện tại cho cả hai chiều: lợi ích EN→VI nhỏ và chưa chắc chắn, VI→EN giảm rõ hơn, đồng thời tốn thêm thời gian và recovery.', '',
             '## Dữ liệu và độ tin cậy', '',
             'Đã tải kết quả private kernel wann179/vlsp26-v13-batching-glossary-pt50 về results/v13_kaggle_20261004. Đủ 200/200 file: mỗi bản 50 tài liệu × hai chiều. Cả bốn lượt chấm lại local bằng evaluator chuẩn khớp Kaggle ở raw/NFC/text BLEU và chrF, missing=0. Đây là số đo GPU thật, không phải mock hay replay.', '',
             'T4 ×2, GPU 0 batching; GPU 1 glossary; mỗi GPU EN→VI rồi VI→EN; batch size 4, cùng model và public CSV. Glossary có 368 cặp từ nguồn train, soft_terms=4, giữ batch FIFO, không dùng source context/history. Audit không thấy trùng sample ID train với dev/public. Co-occurrence trong train không bảo đảm mỗi cặp đúng mọi ngữ cảnh.', '',
             'Baseline ban đầu là B v1.1 (gom node, newline, không glossary). B v1.2 là replay và có thay đổi recovery, chưa đo GPU. Không có baseline v1.3 FIFO/no-glossary chạy mới cùng phiên: tốc độ so lịch sử mang tính tham khảo; cải thiện unresolved không thể quy hoàn toàn cho batching hoặc glossary.', '',
             '## Điểm dịch và tốc độ', '',
             '| Bản | Chiều | BLEU raw | Δ BLEU | chrF | Giờ dịch | Δ thời gian vs B v1.1 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for d in ('en-vi','vi-en'):
        t=gpu['runs'][f'batch_length/{d}']['historical_B_seconds']
        s=old['scores']['B'][d]['raw']
        lines.append(f"| B v1.1 | {d} | {s['bleu']:.2f} | — | {s['chrf']:.2f} | {t/3600:.3f} | — |")
        for v in ('batch_length','glossary'):
            r=gpu['runs'][f'{v}/{d}'];s=r['score']['raw']
            lines.append(f"| {v} | {d} | {s['bleu']:.2f} | {r['bleu_delta_vs_B_v11']:+.2f} | {s['chrf']:.2f} | {r['runtime_seconds']/3600:.3f} | {(r['runtime_seconds']/t-1)*100:+.2f}% |")
    lines += ['', 'Batching tiết kiệm 57,1 phút EN→VI và 36,1 phút VI→EN; tổng 93,2 phút/GPU cho 100 tài liệu. Tổng thời gian dịch trên một GPU: baseline 7,413 giờ; batching 5,859 giờ; glossary 7,698 giờ. Throughput batching tương đương khoảng 1,265× baseline. Glossary tăng thời gian khoảng 3,4%/4,6%. Đây là runtime pipeline; thời gian toàn kernel còn gồm load model, chấm và đóng gói.', '',
              'BLEU đo mức khớp với bản tham chiếu, chrF đo mức khớp ký tự. Chúng không trực tiếp bảo đảm đúng nghĩa pháp lý. NFC và điểm text cũng giữ cùng xu hướng: batching text 68,46/45,82 so baseline 68,48/45,79; glossary text 68,55/45,43. Sự giảm VI→EN không chỉ xuất hiện ở Markdown raw.', '',
              '## Mức chắc chắn và từng tài liệu', '',
              'Bootstrap ghép cặp: lấy lại 50 tài liệu có hoàn lại 2.000 lần, seed 42, tính lại corpus BLEU từ thống kê SacreBLEU 2.6.0. Khoảng dưới đây mô tả biến động theo tài liệu của tập này, không bao gồm biến động nhiều lần chạy GPU hoặc khái quát mọi dữ liệu.', '',
              '| Bản | Chiều | Tăng / giảm / gần bằng | Khoảng 95% của Δ corpus BLEU |',
              '|---|---|---:|---:|']
    for d,e in analysis['directions'].items():
        for v in ('batch_length','glossary'):
            a=e['variants'][v];lo,hi=a['bootstrap_bleu_delta_95ci']
            lines.append(f"| {v} | {d} | {a['docs_improved']} / {a['docs_worse']} / {a['docs_equal_within_005']} | [{lo:+.3f}; {hi:+.3f}] |")
    lines += ['', 'Gần bằng nghĩa là chênh điểm từng tài liệu ≤0,005. Khoảng batching hai chiều và glossary EN→VI đều chứa 0; chưa có bằng chứng chắc chắn rằng điểm tăng/giảm. Glossary VI→EN có khoảng hoàn toàn âm. Số tài liệu tăng điểm không quyết định corpus BLEU: tài liệu dài đóng góp nhiều thống kê hơn.', '',
              'Batching không bảo đảm output giống hệt: chỉ 8/50 file EN→VI và 11/50 VI→EN giống baseline theo nội dung Unicode đã đọc. Đổi thành phần batch có thể làm phép tính fp16 và chọn token/recovery khác đi; đây là khả năng giải thích, chưa được tách kiểm chứng. Chênh điểm vẫn nhỏ.', '',
              '## Kết quả theo nhóm pháp luật', '',
              'Mỗi ô: BLEU của baseline → batching → glossary. Đây là corpus BLEU tính riêng trong nhóm, không phải trung bình điểm tài liệu.', '',
              '| Nhóm | Số tài liệu | EN→VI | VI→EN |', '|---|---:|---|---|']
    for cat,entry in analysis['directions']['en-vi']['categories'].items():
        cells=[]
        for d in ('en-vi','vi-en'):
            s=analysis['directions'][d]['categories'][cat]['corpus_bleu']
            cells.append(' → '.join(f'{s[k]:.2f}' for k in ('baseline','batch_length','glossary')))
        lines.append(f"| {cat} | {entry['documents']} | {cells[0]} | {cells[1]} |")
    lines += ['', 'Glossary giảm VI→EN ở cả 7 nhóm; mức giảm lớn nhất trong nhóm trách nhiệm hình sự khoảng −0,65 BLEU. EN→VI cải thiện rõ hơn ở trách nhiệm hình sự (+0,33), nhưng quyền dân sự giảm −0,21. Tập nhóm nhỏ (4–12 tài liệu), chưa đủ để kết luận chọn glossary riêng theo domain.', '',
              '## Tài liệu cần xem và ví dụ thực tế', '']
    for d in ('en-vi','vi-en'):
        a=analysis['directions'][d]['variants']['glossary']
        lines += [f'Glossary {d}, năm tài liệu giảm mạnh nhất:', '', '| Sample ID | Baseline | Glossary | Δ BLEU |', '|---|---:|---:|---:|']
        for r in a['largest_regressions']:
            lines.append(f"| {r['sample_id']} | {r['baseline']:.2f} | {r['new']:.2f} | {r['delta']:+.2f} |")
        lines.append('')
    lines += ['Ở Quyen_dan_su/164 VI→EN, tiêu đề nguồn có “cấp thị thực, cấp thẻ tạm trú”. Baseline dịch “VISAS, TEMPORARY RESIDENCE PERMITS”; glossary đổi thành “LAISSEZ-PASSER, PERMANENT RESIDENCE CARD”, không khớp loại giấy tờ trong nguồn và reference. Trace của recovery unit u0000.r7 có gợi ý “xuất nhập cảnh translates to immigration”. Không có cặp gợi ý trực tiếp thị thực→laissez-passer ở trace này; chưa thể kết luận một cặp glossary cụ thể gây lỗi. Điểm tài liệu giảm 44,00→40,87.', '',
              'Ở So_huu_tri_tue/118, nhiều chỗ đổi “Intellectual Property Law” thành “Law on Intellectual Property”. Đây là ví dụ thay cách diễn đạt có thể giảm BLEU dù nghĩa gần nhau; không phải mọi phần giảm đều là sai nghĩa. Tài liệu giảm 53,57→50,89.', '',
              'Ở Dich_vu_phap_ly/105, đổi “law partnership companies”→“partnership law firms”, “Ministry of Justice”→“Justice Ministry”, cùng nhiều câu được viết lại. Điểm giảm 52,11→48,74. Cần đọc đối chiếu ngữ cảnh để tách đồng nghĩa khỏi sai nội dung.', '',
              '## Recovery, cấu trúc và cảnh báo', '',
              '| Bản | Chiều | Request gửi GPU | Recovery request | Retry | Unresolved unit |', '|---|---|---:|---:|---:|---:|']
    for d in ('en-vi','vi-en'):
        r=old['runs']['B'][d]
        lines.append(f"| Baseline | {d} | {r['requests_sent']} | {r['recovery_requests']} | {r['retry_count']} | {r['unresolved_units']} |")
        for v in ('batch_length','glossary'):
            r=gpu['runs'][f'{v}/{d}']
            lines.append(f"| {v} | {d} | {r['requests_sent']} | {r['recovery_requests']} | {r['retry_count']} | {r['unresolved_units']} |")
    lines += ['', 'Recovery là request dịch lại sau kiểm tra; không phải số tài liệu hỏng. Nó có thể trùng retry hoặc tận dụng cache, nên không cộng các cột để tính tổng request. Glossary so batching tăng recovery 8,5% EN→VI và 14,7% VI→EN. Cả hai bản mới không có OOM/lỗi vận hành, không có node rỗng unresolved, parser roundtrip 50/50 mỗi chiều.', '',
              'Baseline v1.1 unresolved 7/1 unit; batching 2/0; glossary 3/0. Tuy nhiên B v1.2 replay đã đạt 2/0: không gán toàn bộ cải thiện này cho batching. Hai lỗi batching EN→VI còn ở So_huu_tri_tue/118 (protected_heading_lost) và Vi_pham_hanh_chinh/20 (line_count_mismatch 4/5). Glossary còn thêm Thu_tuc_to_tung/111 (line_count_mismatch 11/8).', '',
              '| Cảnh báo (số tài liệu) | EN→VI baseline / batching / glossary | VI→EN baseline / batching / glossary |', '|---|---:|---:|']
    for warning in ('number_diff','number_diff_big','loop_not_in_source','heading_order','heading_not_as_heading'):
        cells=[]
        for d in ('en-vi','vi-en'):
            cells.append(' / '.join(str(analysis['directions'][d]['variants'][v]['warning_documents'].get(warning,0)) for v in ('baseline','batch_length','glossary')))
        lines.append(f'| {warning} | {cells[0]} | {cells[1]} |')
    lines += ['', 'Các cảnh báo số liệu/heading là tín hiệu tự động để rà soát, chưa phải xác nhận sai nghĩa; cách viết số/ngày và nội dung lặp có thể gây báo nhầm. Glossary VI→EN tăng number_diff_big từ 4→6 tài liệu và heading_order từ 3→4; không thể coi bản này cải thiện toàn diện dù unresolved=0.', '',
              'Quyen_dan_su/390 EN→VI chỉ dài khoảng 63,5% reference ở cả ba bản. Lịch sử ABC đã ghi nhận nguồn EN khoảng 11,7k ký tự nhưng reference VI khoảng 27,5k ký tự; đây là dấu hiệu lệch nguồn/tham chiếu, không đủ căn cứ quy cho thiếu dịch. VI→EN không có tài liệu dưới 70%.', '',
              '## Hiệu quả batching và tác động glossary', '',
              'Padding đầu vào batching 4,53%/3,56%; glossary FIFO 29,75%/29,59%. Trung bình batch gần đủ 4 request ở cả hai bản. Batching gửi hơi nhiều request hơn baseline (5.295/5.369 so 5.220/5.295) mà vẫn nhanh hơn; tăng tốc không đến từ giảm số request. Prompt token và output token gần baseline. Điều này phù hợp với mục tiêu giảm tính toán đệm.', '',
              'Baseline cũ không ghi padding, do đó không thể khẳng định mức giảm padding đối chứng. So trực tiếp batching với glossary còn trộn khác biệt prompt/recovery; không dùng chênh padding đó làm ước lượng thuần tác động batching.', '',
              'Glossary so baseline tăng tổng prompt token 13,9% EN→VI và 14,1% VI→EN; generated token tăng khoảng 2,2%/3,0%. Phần tăng bao gồm gợi ý thuật ngữ và request phát sinh, không phải chỉ độ dài mỗi prompt. Có thêm chi phí nhưng chưa có lợi ích chất lượng tương xứng.', '',
              '## Quyết định tiếp theo', '',
              '1. Giữ batch_size=4 và dùng length batching cho bản tiếp theo; giữ glossary tắt mặc định.',
              '2. Rà soát các cặp glossary theo ngữ cảnh, đặc biệt VI→EN và những thuật ngữ có nhiều cách dịch; đánh giá trên dev trước khi dùng public để xác nhận.',
              '3. Nếu cần kết luận nhân quả về tốc độ, chạy thêm v1.3 FIFO/no-glossary cùng môi trường và lặp đo; đây là đề xuất, chưa chạy thêm.',
              '4. Chưa thử kết hợp batching+glossary, nên không suy ra điểm hoặc tốc độ của bản kết hợp. Ưu tiên xử lý lỗi tiêu đề và số liệu; tách kiểm tra chất lượng cặp nguồn/tham chiếu Quyen_dan_su/390.', '',
              'Dữ liệu chi tiết: reports/v13_kaggle_results.json và reports/v13_detailed_analysis.json. Script tái tạo phân tích: kaggle/public_test/analyze_v13.py.']
    (ROOT/'reports/v13_detailed_assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
