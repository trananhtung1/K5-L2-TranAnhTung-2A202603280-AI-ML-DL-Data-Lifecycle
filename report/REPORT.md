# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-07

**Runtime Colab:** GPU (CUDA)

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.14.1+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. `class_id`=468; `class_name`=`cab`; `rank`=1; `score`=0.510915; `taxonomy_name`=`ImageNet-1K`.
2. Record này nói về cả bức ảnh `traffic`, không phải một vật thể riêng, vì tác vụ image classification chỉ xếp hạng một lớp cho toàn bộ ảnh và record không có bounding box hay tọa độ object.
3. Danh sách class đến từ taxonomy `ImageNet-1K` đi kèm checkpoint `yolo11n-cls.pt`, không phải model tự đặt ra tên lớp.
4. `class_id` giúp xác định mã lớp chính xác, `class_name` giúp đọc hiểu nhãn, còn `taxonomy_name` cho biết lớp đó thuộc danh sách taxonomy nào.
5. Guideline cần quy định cách chọn một nhãn chính khi ảnh có nhiều vật thể, ví dụ ưu tiên chủ đề/cảnh chính hoặc vật thể trung tâm.
6. Score 0.510915 chỉ là mức tin của model cho prediction `cab`; nó không phải ground truth do con người kiểm tra và gán theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. `class_name`=`bowl`; `score`=0.719063; `bbox_xyxy`=[32.65, 342.12, 100.16, 384.93]; `bbox_width`=67.51; `bbox_height`=42.81.
2. Box bắt đầu ở góc trên-trái `(32.65, 342.12)` và kết thúc ở góc dưới-phải `(100.16, 384.93)`. Box rộng 67.51 pixel và cao 42.81 pixel. Nó nằm ở vùng phía dưới-bên trái của ảnh kitchen.
3. Ở threshold 0.20 có 17 prediction; ở threshold 0.60 có 6 prediction.
4. Khi hạ threshold, reviewer phải kiểm tra thêm các prediction score thấp: có phải object thật không, class có đúng không, box có bị lệch/trùng hoặc bỏ sót object nào không.
5. Quy tắc box: mỗi object được gán một box riêng, box ôm sát phần object nhìn thấy và không bao sang object bên cạnh hoặc nền không liên quan.
6. Prediction `oven 0.69` ở phía trái ảnh bị cắt bởi mép ảnh. Tôi dùng guideline: vẫn vẽ box cho phần object nhìn thấy, để cạnh box dừng ở mép khung ảnh và không suy đoán phần object nằm ngoài ảnh. Đây là quy tắc có thể áp dụng nhất quán cho nhiều ảnh.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. `instance_id`=`kitchen-007`; `class_name`=`oven`; `score`=0.489667; `polygon_point_count`=213; 5 điểm đầu của `polygon_xy`=[[1.0, 189.0], [1.0, 218.0], [1.0, 214.0], [2.0, 213.0], [2.0, 201.0]].
2. So với box, polygon mô tả biên và hình dạng sát hơn của object theo pixel; box chỉ là hình chữ nhật bao quanh object.
3. `instance_id`=`kitchen-007` dùng để phân biệt instance lò nướng này với các instance khác trong output. Nó không phải class ID (`69`) và không phải tracking ID qua nhiều ảnh/video.
4. Quy tắc mask: vẽ polygon theo biên pixel của phần object nhìn thấy; mask của một instance không được lấn sang pixel thuộc object khác hoặc nền.
5. Prediction `spoon 0.49` ở góc trên-phải là object mảnh, nằm gần nhiều dụng cụ treo và nền có chi tiết tương tự. Tôi chọn escalation nếu guideline chưa quy định mức độ chi tiết khi vẽ mask cho object mảnh hoặc bị che một phần, vì annotator có thể vẽ biên khác nhau.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho toàn bộ ảnh, gồm class_id, class_name và taxonomy theo guideline của dự án. | Với sample_id=traffic, top-1 là cab, score 0.510915. Ảnh có thể có nhiều vật thể/chủ đề nhưng classification chỉ chọn một nhãn, nên cần tiêu chí chọn nội dung chính. Prediction này không phải ground truth. | Xem toàn cảnh và chọn một nhãn chính theo guideline đã quy định. | Kiểm tra nhãn có đúng taxonomy và tiêu chí chọn nội dung chính không; không chấp nhận việc annotator chỉ sao chép prediction của model. |
| Phát hiện vật thể | Mỗi object nhìn thấy có một class và một bounding box bbox_xyxy theo pixel. | Ở vùng dưới-trái, prediction person 0.61 tương ứng với phần tay/người chỉ nhìn thấy một phần. Đây là ca object bị che khuất, cần guideline nêu rõ khi nào vẫn gán nhãn phần object còn nhìn thấy. Ngoài ra, số prediction tăng từ 6 ở threshold 0.60 lên 17 ở 0.20, nên prediction score thấp cần được kiểm tra thêm. | Xác định từng object thuộc phạm vi gán nhãn, vẽ box cho phần nhìn thấy và áp dụng nhất quán quy tắc với object chỉ lộ một phần. | Kiểm tra object bị che có được gán nhãn đúng và đủ không; kiểm tra box có bao đúng phần object nhìn thấy, có thiếu object, sai class hoặc trùng box không. |
| Instance segmentation | Mỗi instance có class và mask/polygon riêng bám theo biên object; hai object cùng class vẫn là hai instance khác nhau. | Các bowl ở góc dưới-trái đã được tách khỏi vùng dining table, đây là điểm cần reviewer kiểm tra để tránh mask lấn nhau. Tuy nhiên, detection có cup 0.45 trong khi output segmentation không có instance cup; reviewer cần đối chiếu ảnh gốc để xác nhận model segmentation có bỏ sót instance hay không. | Vẽ mask riêng cho từng instance nhìn thấy, tách bowl khỏi mặt bàn và vẫn gán nhãn cup nếu cup thuộc phạm vi class và nhìn thấy trong ảnh; không phụ thuộc hoàn toàn vào prediction của model. | Đối chiếu ảnh gốc với tất cả mask để tìm object bị bỏ sót, như cup; kiểm tra mask của bowl và dining table không chồng/lấn sai và các instance cùng class được tách riêng. |

## 5. An toàn dữ liệu

1. Tôi chỉ sử dụng ba ảnh COCO công khai do notebook cung cấp; không tải lên Colab hoặc GitHub ảnh cá nhân, ảnh có khuôn mặt/biển số, dữ liệu khách hàng hay dữ liệu nội bộ.
2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi bài lab, tôi sẽ dừng sử dụng dữ liệu đó và báo Lab Coach/giảng viên để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
