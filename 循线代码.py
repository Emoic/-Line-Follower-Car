import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
from timeout_decorator import timeout

GPIO.setmode(GPIO.BOARD) 
GPIO.setup(22,GPIO.OUT)	
pwm1 = GPIO.PWM(22, 1000)
pwm1.start(0)                 
GPIO.setup(16,GPIO.OUT)
GPIO.setup(18,GPIO.OUT)
GPIO.setup(13,GPIO.OUT)
GPIO.setup(15,GPIO.OUT)

GPIO.setup(12,GPIO.OUT)	
pwm2 = GPIO.PWM(12, 1000)
pwm2.start(0)                 
GPIO.setup(7,GPIO.OUT)
GPIO.setup(11,GPIO.OUT)


# center定义
center = 320
# 打开摄像头，图像尺寸640*480（长*高），opencv存储值为480*640（行*列）
cap = cv2.VideoCapture(0)


while (1):
    ret, frame = cap.read()
    # 转化为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 大津法二值化
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # 膨胀，白区域变大
    dst = cv2.dilate(dst, None, iterations=14)
     # 腐蚀，白区域变小
    #dst = cv2.erode(dst, None, iterations=5)

    # 单看第400行的像素值s
    color = dst[419]
    color2 = dst[300]
    # 找到白色的像素点个数
    black_count1 = np.sum(color == 0)
    black_count2 = np.sum(color2 == 0)
    # 找到black的像素点索引
    black_index1 = np.where(color == 0)
    black_index2 = np.where(color2 == 0)

    # 防止black_count=0的报错
    if black_count1 == 0:
        black_count1 = 1
    #if black_count2 == 0:
        #black_count2 = 1

    # 找到黑色像素的中心点位置
    # center = (white_index[0][white_count - 1] + white_index[0][0]) / 2
    center1 = (black_index1[0][black_count1-1] + black_index1[0][0]) /2;
    #center2 = (black_index2[0][black_count2-1] + black_index2[0][0]) /2;
    # 计算出center与标准中心点的偏移量
    direction1 = center1 - 320
    #direction2 = center2 - 320

    print(direction1)

    GPIO.output(16,GPIO.HIGH)
    GPIO.output(18,GPIO.LOW)
    pwm1.ChangeDutyCycle(30)
    GPIO.output(7,GPIO.HIGH)
    GPIO.output(11,GPIO.LOW)
    pwm2.ChangeDutyCycle(30)
    if abs(direction1) > 300:
        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)



    elif direction1 >= 0:
        # 限制在70以内
      
        if 30 < direction1:
            direction = 30
            GPIO.output(16,GPIO.LOW)
            GPIO.output(18,GPIO.HIGH)
            pwm1.ChangeDutyCycle(22)
        
            GPIO.output(11,GPIO.LOW)
            GPIO.output(7,GPIO.HIGH)
            pwm2.ChangeDutyCycle(22)
            GPIO.output(13,GPIO.HIGH)
          
    
    
    
    elif direction1 < 0:
        if direction1 < -30:
            direction1 = -30
            GPIO.output(18,GPIO.LOW)
            GPIO.output(16,GPIO.HIGH)
            pwm1.ChangeDutyCycle(22)
          
            GPIO.output(7,GPIO.LOW)
            GPIO.output(11,GPIO.HIGH)
            pwm2.ChangeDutyCycle(22)
            GPIO.output(15,GPIO.HIGH)
         

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pwm1.stop()
pwm2.stop()
GPIO.output(13,GPIO.LOW)
GPIO.output(15,GPIO.LOW)
GPIO.cleanup()
