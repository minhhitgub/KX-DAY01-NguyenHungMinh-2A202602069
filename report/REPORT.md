# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 
class_id: 468, class_name: "cab" (xe taxi), rank: 1,score: 0.510915, taxonomy_name: "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
Record đang thể hiện là trong bức ảnh này thì cab đang chiếm phần lớn hình ảnh, tuy nhiên không mô tả được toàn cảnh các chi tiết khác như xe bus, người đi đường. Điểm số 0,51 cho thấy ngoài cab ra thì còn nhiều vật thể khác.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Class list do các nhà nghiên cứu, kỹ sư dữ liệu hay người thiết kế bài toàn định nghĩa và chuẩn bị dữ liệu trước khi mô hình được huấn luyện
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
Cần ID vì nó là dạng số nguyên, tối ưu cho máy tính lưu trữ, tính toán ma trận và xử lí nhanh gọn. Cần tên lớp vì nó là dạng. văn bản giúp con người đọc và hiểu kết quả là vật thể gì. Cần tên taxonomy vì đây là hệ quy chiếu để đối chiếu. Điều này rất quan trọng để tránh trùng lặp hoặc hiểu sai bối cảnh.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Guideline sẽ cần quy định những điều sau:
+Chỉ gán nhãn cho cho vật thể chiếm diện tích lớn nhất.
+Chỉ gán nhãn cho vật thể nằm ở vị trí trung tâm bức ảnh.
+Ưu tiên theo bài toàn nghiwwpj vụ.
- Vì sao model score không phải ground truth?
Model score chỉ là xác suất dự đoán sinh ra từ hàm toán học của thuật toán bên trong checkpoint. Máy móc có thể tính toán sai, còn ground truth là sự thật do con người xác nhận dựa trên thực tế bức ảnh và quy định của guideline.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
class_name: "person", score: 0.912625, bbox_xyxy: [385.33, 69.24, 498.92, 348.92], bbox_width: 113.58, bbox_height: 279.68.
- Diễn giải vị trí box bằng lời:
Bức ảnh có kích thước rộng 640px, cao 427px. Gốc tọa độ (0,0) nằm ở góc trên cùng bên trái. Trục ngang: Bounding box bắt đầu từ pixel thứ 385.33 và kết thúc ở pixel 498.92. Điều này cho thấy người này đang đứng lệch về phía nửa bên phải của bức ảnh. Trục dọc: Bounding box bắt đầu từ pixel thứ 69.24 và kết thúc ở 348.92. Kích thước chiều cao (279.68) lớn hơn nhiều so với chiều rộng (113.58), phù hợp với hình dáng của một người đang trong tư thế đứng.
- So sánh số prediction ở hai threshold:
Ngưỡng >0.6 có 4 vật thể, >0.9 có 1 vật thể.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Khi hạ thấp ngưỡng: Độ bao phủ tăng lên, mô hình bắt được nhiều vật thể hơn và ít bỏ sót. Tuy nhiên, khối lượng công việc của reviewer sẽ tăng mạnh do phải kiểm tra và xóa bỏ rất nhiều các false positives. Khi tăng ngưỡng thì precision cao hơn, các box giữ lại thường rất chắc chắn. Khối lượng công việc xóa false positive giảm, nhưng reviewer lại tốn thời gian để tự vẽ tay bổ sung các vật thể bị bỏ sót false negatives do mô hình không đạt đủ ngưỡng tự tin.
- Đề xuất một quy tắc box chặt:
Quy tắc vẽ box: Bốn cạnh của bounding box phải chạm khít vào các điểm cực biên của vật thể, không được cắt lẹm vào vật thể và cũng không được để chừa quá nhiều khoảng trống bên trong box.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Bị che khuất: Guideline cần quy định rõ là vẽ box bao trọn chỉ phần nhìn thấy được hay vẽ bao trọn cả phần bị che khuất theo suy đoán logic. Ngoài ra, cần quy định mức độ che khuất tối đa, ví dụ như bị che quá 80% thì không cần gán nhãn.
Bị cắt mép ảnh: Guideline cần làm rõ cạnh của bounding box sẽ dừng lại đúng tại sát mép ảnh hay không. Đồng thời, cần quy định vật thể bị cắt mất bao nhiêu phần trăm thì vẫn được coi là hợp lệ để vẽ (ví dụ: chỉ thấy mỗi bánh xe thì có gán nhãn là ô tô không). Những trường hợp mơ hồ này cần được đưa lên cho quản lý dự án quyết định nhằm đảm bảo tính nhất quán của bộ dữ liệu.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
instance_id: "kitchen", class_name: "person", score: 0.899318, số điểm: 348 điểm

Một phần polygon_xy: [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], ...]
Mô tả hình dáng polygon: vùng mask màu xanh lam của người này không phải là một hình khối cơ bản. Đường biên polygon uốn lượn men theo chính xác hình dáng cơ thể: bám sát phần đầu, viền qua vai, chạy dọc theo thân mình đang mặc tạp dề và bao quanh hai chân. Vùng giữa hai chân và bối cảnh phía sau nách được khoét ra, không bị gộp vào mặt nạ.
- Polygon bổ sung chi tiết gì so với box?
Bounding box chỉ là một hình chữ nhật bao quanh các điểm cực biên của vật thể (x_min, y_min, x_max, y_max), do đó nó bao gồm cả một lượng lớn các điểm ảnh thuộc về phần nền (background). Polygon đa giác khắc phục điểm yếu này bằng cách cung cấp chuỗi các tọa độ bám sát chính xác đường viền thực tế của vật thể. Điều này giúp loại bỏ phần nền, mang lại thông tin chi tiết về hình dáng, tư thế và diện tích thực của vật thể ở cấp độ điểm ảnh.
- `instance_id` dùng để làm gì và không phải loại ID nào?
instance_id là mã định danh độc nhất dùng để phân biệt các cá thể (instance) độc lập bên trong cùng một bức ảnh (ví dụ: để phân biệt "người thứ nhất" là kitchen-001 với "người thứ hai" là kitchen-002). Nó không phải là class_id (mã danh mục phân loại chung, ví dụ: 0 là "person"). Nó cũng không phải là tracking_id (mã theo dõi một vật thể duy nhất di chuyển qua nhiều khung hình trong một đoạn video).
- Đề xuất một quy tắc biên mask:
Quy tắc bám sát viền: Các điểm neo anchor points của polygon phải được đặt chính xác tại ranh giới chuyển màu giữa vật thể và nền hoặc vật thể khác. Tuyệt đối không vẽ lấn ra ngoài phần nền và không vẽ lẹm vào bên trong vật thể. Trừ khi có quy định riêng, bóng đổ không được tính vào vùng mask.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Vùng mờ: Guideline cần quy định rõ ràng rằng điểm neo sẽ nằm ở mép ngoài cùng của dải mờ, mép trong cùng (vùng sắc nét), hay đi qua điểm giữa của dải chuyển màu (gradient).
Tiếp xúc/Che khuất: Cần quy định việc gán nhãn chỉ vẽ bề mặt thực tế nhìn thấy được (visible mask) hay được phép nội suy/vẽ bù phần bị che khuất (amodal mask). Khi hai vật thể chồng lên nhau, cần có quy tắc quy định mask nào đè lên mask nào.
Escalation: Trong trường hợp vật thể bị che khuất quá mức dẫn đến đứt gãy thành nhiều mảnh nhỏ, hoặc vùng mờ khiến người gán nhãn không thể phân biệt được hình thù, các trường hợp này cần được báo cáo lên Quản lý dự án hoặc chuyên gia dữ liệu để đưa ra quyết định có gán nhãn tiếp hay bỏ qua nhằm đảm bảo chất lượng dữ liệu.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh**<br>(Image Classification) | Một nhãn (Class ID/Name) đại diện cho toàn ảnh, hoặc một mảng nhãn nếu là multi-label. | Ảnh chứa nhiều chủ thể khác nhau nên không biết chọn chủ thể nào; chủ thể quá nhỏ hoặc không rõ ràng. | Quan sát toàn ảnh và chọn nhãn đúng nhất dựa trên quy tắc ưu tiên (priority rule) của guideline. | Kiểm tra nhãn có thuộc đúng Taxonomy không và có tuân thủ quy tắc ưu tiên khi ảnh có nhiều vật thể không. |
| **Phát hiện vật thể**<br>(Object Detection) | Tọa độ bounding box `[x_min, y_min, x_max, y_max]` kèm Class ID cho từng vật thể. | Box bị rộng, thừa nhiều nền hoặc cắt lẹm vật thể; vật thể bị che khuất (occluded) hoặc bị cắt ở mép ảnh. | Vẽ khung chữ nhật bám sát 4 điểm cực biên (trên, dưới, trái, phải) của phần vật thể nhìn thấy được. | Kiểm tra độ khít (tightness) của box; kiểm tra các trường hợp vật thể bị che khuất hoặc cắt mép có được xử lý đúng guideline không. |
| **Instance segmentation** | Polygon (đa giác) gồm mảng các cặp tọa độ `[x, y]` bám sát viền, kèm Class ID và `instance_id`. | Đường viền lấn ra nền hoặc lẹm vào trong; ranh giới bị mờ nhòe; các vật thể chồng chéo hoặc che lấp nhau. | Chấm các điểm neo (anchor points) theo đúng ranh giới cấp độ pixel của từng cá thể riêng biệt. | Phóng to (zoom in) để kiểm tra độ chính xác của đường biên (pixel-perfect); kiểm tra sự phân tách của các mask chồng lên nhau. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: 
Tuyệt đối không sao chép, tải xuống thiết bị cá nhân, chụp màn hình hay chia sẻ bất kỳ dữ liệu nào (hình ảnh thô, file dự đoán JSON, nhãn ground truth) ra bên ngoài nền tảng/không gian làm việc nội bộ đã được phê duyệt của dự án. Mọi tác vụ xử lý dữ liệu phải tuân thủ nghiêm ngặt các quy định về bảo mật thông tin (ví dụ: các yêu cầu về ẩn danh PII - làm mờ khuôn mặt, biển số xe nếu có).

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
Quản lý dự án (Project Manager / PM), Trưởng nhóm (Team Lead), hoặc Cán bộ phụ trách chất lượng/bảo mật dữ liệu để nhận chỉ thị.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
