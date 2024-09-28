import cv2
from ultralytics import YOLO
import pandas as pd

# YOLOv8 modelini yükle
model = YOLO('yolov8s.pt')

# Araç sınıflarını tanımla
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']

# Sayım çizgisinin Y koordinatı (hiza)
counting_line_y = 460
line= 490

# Araç sayımı
vehicle_count = {
    'car': 0,
    'bus': 0,
    'truck': 0,
    'motorcycle': 0
}

# Araçların izini tutmak için bir yapı
vehicle_tracker = {}

# Video işleme fonksiyonu
def process_video(video_path):
    global vehicle_count, vehicle_tracker

    # VideoCapture nesnesi oluştur
    cap = cv2.VideoCapture(video_path)
    
    # Video yazıcı (output) ayarları
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    vehicle_id = 0  # Araç ID'si izlemek için

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO modelini kullanarak tahmin yap
        results = model(frame)

        # Sonuçları al
        results_list = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        # Mevcut çerçevede görülen araçların ID'lerini takip etmek için geçici bir set
        current_frame_vehicles = set()
        
        # Tahmin edilen araçları gruplandır ve ekranda göster
        for i, (box, cls, conf) in enumerate(zip(results_list, classes, confidences)):
            if model.names[int(cls)] in vehicle_classes:
                vehicle_type = model.names[int(cls)]
                xmin, ymin, xmax, ymax = map(int, box)
                confidence = conf

                # Yeni bir araçsa ID ata ve izleyiciye ekle
                if (xmin, ymin, xmax, ymax) not in vehicle_tracker:
                    vehicle_tracker[(xmin, ymin, xmax, ymax)] = {'id': vehicle_id, 'counted': False}
                    vehicle_id += 1

                vehicle_info = vehicle_tracker[(xmin, ymin, xmax, ymax)]
                current_frame_vehicles.add(vehicle_info['id'])
                
                # Araç hiza çizgisini geçti mi kontrol et
                if ymin <= counting_line_y <= ymax and  vehicle_info['counted']==False:
                    # Kutunun tam orta noktasını hesapla
                    y_center = (ymin + ymax) // 2
                    # Eğer kutunun orta noktası, sayım çizgisine değdiyse
                    if y_center >= counting_line_y and y_center<=line  and y_center< line:
                        vehicle_count[vehicle_type] += 1
                        vehicle_info['counted'] = True

                # Sınıf adı ve güven ile sınır kutusunu çiz
                label = f"{vehicle_type} {confidence:.2f}"
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Araçları izleme listesinden güncelle
        vehicle_tracker = {box: info for box, info in vehicle_tracker.items() if info['id'] in current_frame_vehicles}
        
        # Hiza çizgisini çiz
        #cv2.line(frame, (0, counting_line_y), (frame.shape[1], counting_line_y), (0, 0, 255), 2)
        #cv2.line(frame, (0, line), (frame.shape[1], counting_line_y), (0, 0, 255), 2)
        cv2.line(frame,(0,450),(1280,450),(0,0,255),3)
        cv2.line(frame,(0,470),(1280,470),(0,0,255),3)
        # Araç sayımını sağ üst köşeye yazdır
        text_y = 30
        for vehicle_type, count in vehicle_count.items():
            text = f"{vehicle_type}: {count}"
            cv2.putText(frame, text, (frame.shape[1] - 200, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            text_y += 30
        
        # Sonuçları video dosyasına yaz
        out.write(frame)

        # Sonuçları ekranda göster
        cv2.imshow('frame', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    # Excel dosyasına kaydet
    save_to_excel(vehicle_count)

    # Kaynakları serbest bırak
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def save_to_excel(vehicle_count):
    # DataFrame oluştur
    df = pd.DataFrame(list(vehicle_count.items()), columns=['Vehicle Type', 'Count'])
    
    # Excel dosyasına yaz
    df.to_excel('vehicle_count.xlsx', index=False)

# Test işlemi
video_path = 'cars.mp4'  # Buraya test edeceğiniz videonun yolunu ekleyin
process_video(video_path)
