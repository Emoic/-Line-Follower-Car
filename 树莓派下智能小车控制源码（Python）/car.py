#! /usr/bin/python
# -*- coding:UTF-8 -*-

import RPi.GPIO as GPIO
import time
import ultrasonic as ultrasonic
import motor as motor

'''
L298N驱动马达状态表：
--------------------------------------------
	EN    |   IN1   |   IN2   |   电机状态
--------------------------------------------
	低    |  任意   |   任意  |    停止
	高    |   低    |    低   |    制动
	高    |   高    |    高   |    制动
	高    |   高    |    低   |    正传
	高    |   低    |    高   |    反转
--------------------------------------------
'''

'''
四轮定义: 马达 = [IN1, IN2, EN]
'''
Motor_Right_front = [7, 8, 25]	#右前轮
Motor_Right_rear = [9, 10, 11]	#右后轮
Motor_Left_front = [20, 16, 21]	#左前轮
Motor_Left_rear = [26, 19, 13]	#左后轮

#小车点火
def Car_power_on():
	GPIO.setmode(GPIO.BCM)	#设置GPIO模式
	GPIO.setwarnings(False) 
	#马达上电
	motor.Motor_power_on(Motor_Right_front)
	motor.Motor_power_on(Motor_Right_rear)
	motor.Motor_power_on(Motor_Left_front)
	motor.Motor_power_on(Motor_Left_rear)

#小车运行：前进
def Car_run_forward():
	#四马达正传
	motor.Motor_positive(Motor_Right_front)
	motor.Motor_positive(Motor_Right_rear)
	motor.Motor_positive(Motor_Left_front)
	motor.Motor_positive(Motor_Left_rear)

#小车运行：倒车
def Car_run_back():
	#四马达反转
	motor.Motor_negative(Motor_Right_front)
	motor.Motor_negative(Motor_Right_rear)
	motor.Motor_negative(Motor_Left_front)
	motor.Motor_negative(Motor_Left_rear)

#小车运行：左转
def Car_run_left():
	#右边马达正传，左边马达反转
	motor.Motor_positive(Motor_Right_front)
	motor.Motor_positive(Motor_Right_rear)
	motor.Motor_negative(Motor_Left_front)
	motor.Motor_negative(Motor_Left_rear)

#小车运行：右转
def Car_run_right():
	#右边马达反转，左边马达正转
	motor.Motor_negative(Motor_Right_front)
	motor.Motor_negative(Motor_Right_rear)
	motor.Motor_positive(Motor_Left_front)
	motor.Motor_positive(Motor_Left_rear)

#小车运行：停止
def Car_run_pause():
	#四马达停止
	motor.Motor_pause(Motor_Right_front)
	motor.Motor_pause(Motor_Right_rear)
	motor.Motor_pause(Motor_Left_front)
	motor.Motor_pause(Motor_Left_rear)

#小车运行：制动
def Car_run_brake():
	#四马达制动
	motor.Motor_brake(Motor_Right_front)
	motor.Motor_brake(Motor_Right_rear)
	motor.Motor_brake(Motor_Left_front)
	motor.Motor_brake(Motor_Left_rear)

#小车熄火
def Car_Power_Off():
	GPIO.cleanup()

print("The Car is start...")
Car_power_on()
ultrasonic.initial()
time.sleep(0.5)

#操作手册
print('----------------------------------')
print('w: Car forward.')
print('a: Car left.')
print('d: Car right.')
print('s: Car pause.')
print('x: Car back.')
print('f: Car brake.')
print('c: Check Distance.')
print('e: Take picture.')
print('q: quit the program.')
print('----------------------------------')

while True:
	opt = input('Please choose[w/a/s/d/x/f/q]:') 
	if opt == 'w':
		Car_run_forward()
	elif opt == 'a':
		Car_run_left()
	elif opt == 'd':
		Car_run_right()
	elif opt == 's':
		Car_run_pause()
	elif opt == 'x':
		Car_run_back();
	elif opt == 'f':
		Car_run_brake()
	elif opt == 'c':
		print( 'Distance: ', ultrasonic.checkdist(), ' CM' )
	elif opt == 'e':
		print('Not support.')
	elif opt == 'q':
		break
	else:
		print('Invaild input.')

Car_run_pause()
Car_Power_Off()
