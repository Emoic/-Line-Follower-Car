
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


GPIO.setup(12,GPIO.OUT)	
pwm2 = GPIO.PWM(12, 1000)
pwm2.start(0)                 
GPIO.setup(7,GPIO.OUT)
GPIO.setup(11,GPIO.OUT)



center = 320
cap = cv2.VideoCapture(0)


while (1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dst = cv2.dilate(dst, None, iterations=5)
    dst = cv2.erode(dst, None, iterations=6)

    color = dst[400]
    color2 = dst[300]
    black_count1 = np.sum(color == 0)
    black_count2 = np.sum(color2 == 0)
    black_index1 = np.where(color == 0)
    black_index2 = np.where(color2 == 0)

    if black_count1 == 0:
        black_count1 = 1

    center1 = (black_index1[0][black_count1-1] + black_index1[0][0]) /2;
    direction1 = center1 - 320

    print(direction1)

    GPIO.output(16,GPIO.HIGH)
    GPIO.output(18,GPIO.LOW)
    pwm1.ChangeDutyCycle(20)
    GPIO.output(7,GPIO.HIGH)
    GPIO.output(11,GPIO.LOW)
    pwm2.ChangeDutyCycle(20)
   
    if abs(direction1) > 300:
        pwm1.ChangeDutyCycle(0)
        pwm2.ChangeDutyCycle(0)

    elif direction1 >= 0:
      
        if 25 < direction1:
            direction = 25
            GPIO.output(16,GPIO.LOW)
            GPIO.output(18,GPIO.HIGH)
            pwm1.ChangeDutyCycle(25)
        
            GPIO.output(11,GPIO.LOW)
            GPIO.output(7,GPIO.HIGH)
            pwm2.ChangeDutyCycle(25)

    
    
    
    elif direction1 < 0:
        if direction1 < -25:
            direction1 = -25
            GPIO.output(18,GPIO.LOW)
            GPIO.output(16,GPIO.HIGH)
            pwm1.ChangeDutyCycle(25)
          
            GPIO.output(7,GPIO.LOW)
            GPIO.output(11,GPIO.HIGH)
            pwm2.ChangeDutyCycle(25)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pwm1.stop()
pwm2.stop()
GPIO.cleanup()
