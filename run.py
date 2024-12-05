import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from ultralytics import YOLO
import cv2
import time
import schedule

# 发送带附件的邮件函数
def send_email_with_attachment(subject, body, attachment_path=None):
    from_address = ''
    to_address = ''

    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = to_address
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    if attachment_path:
        attachment = open(attachment_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(attachment_path)}')
        msg.attach(part)

    try:
        server = smtplib.SMTP_SSL('smtp.163.com', 465)
        email_password = ''

        server.login(from_address, email_password)
        text = msg.as_string()
        server.sendmail(from_address, to_address, text)
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# 创建保存火焰图片的文件夹
def create_fire_images_folder():
    folder_name = 'fire_images'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

# 发送每天早上8点的安全状态邮件
def send_daily_safety_email():
    now = time.localtime()
    if now.tm_hour == 8 and now.tm_min == 0:
        timestamp = time.strftime("%Y%m%d_080000")
        filename = f"fire_detected_frame_{timestamp}.jpg"
        image_path = os.path.join(folder_name, filename)

        if os.path.exists(image_path):
            # 发送火焰检测邮件
            subject = "Fire Detected!"
            body = "Fire has been detected at 8:00 AM. Please take necessary actions."
            send_email_with_attachment(subject, body, image_path)
        else:
            subject = "实验室状态：安全"
            body = "实验室当前状态正常，无异常情况。"
            send_email_with_attachment(subject, body)

# 初始化模型
model = YOLO('best1.pt')

# 创建保存火焰图片的文件夹
folder_name = create_fire_images_folder()

# 设置邮件发送的时间间隔（秒）
min_email_interval = 60  # 每隔60秒发送一次邮件
last_email_time = 0

# 读取视频
video_source = "./1.mp4"  # 替换为您的本地视频路径或0以使用摄像头
cap = cv2.VideoCapture(video_source)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=320, conf=0.5)

        for result in results:
            boxes = result.boxes

            # 检查是否检测到火焰
            fire_detected = any(model.names[int(box.cls)] == 'fire' for box in boxes)

            # 如果检测到火焰，保存图片并发送邮件
            if fire_detected:
                current_time = time.time()
                if current_time - last_email_time > min_email_interval:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"fire_detected_frame_{timestamp}.jpg"
                    filepath = os.path.join(folder_name, filename)
                    cv2.imwrite(filepath, result.orig_img)
                    print(f"Saved {filename}")

                    # 发送邮件通知
                    subject = "Fire Detected!"
                    body = "Fire has been detected. Please take necessary actions."
                    send_email_with_attachment(subject, body, filepath)

                    # 更新上次发送邮件的时间
                    last_email_time = current_time

            # 在原始帧上绘制检测结果
            annotated_frame = result.plot()

            # 显示带有检测结果的帧
            cv2.imshow('检测结果', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()

# 每天早上8点发送安全状态邮件的定时任务
schedule.every().day.at("08:00").do(send_daily_safety_email)

# 循环执行定时任务
while True:
    schedule.run_pending()
    time.sleep(1)
