#! /usr/bin/python
# -*- coding:UTF-8 -*-

import RPi.GPIO as GPIO
import time

'''
超声波探头接线方法：
VCC,超声波模块电源脚，接5V电源即可
Trig，超声波发送脚，高电平时发送出40KHZ出超声波
Echo，超声波接收检测脚，当接收到返回的超声波时，输出高电平
GND，超声波模块GND
'''
TRIGPIN = 2 #GPIO2，PIN 3
ECHOPIN = 3	#GPIO3，PIN 5

#计算长度，返回单位为厘米
def checkdist():
	GPIO.output(TRIGPIN,GPIO.HIGH)	#发出触发信号
	time.sleep(0.000015)		#保持10us以上（我选择15us）
	GPIO.output(TRIGPIN,GPIO.LOW)	
	
	while not GPIO.input(ECHOPIN):
		pass
	#发现高电平时开时计时
	t1 = time.time()
	#print ("T1=%f", t1)
	
	while GPIO.input(ECHOPIN):
		pass
	#高电平结束停止计时
	t2 = time.time()
	#print ("T2=%f", t2)
	
	#返回距离，单位为厘米
	return (t2-t1)*34000/2	#空气（15℃）中声音传播速度 340m/s 

def initial():
	GPIO.setmode(GPIO.BCM)
	#第3号针，GPIO2
	GPIO.setup(TRIGPIN,GPIO.OUT,initial=GPIO.LOW)
	#第5号针，GPIO3
	GPIO.setup(ECHOPIN,GPIO.IN)

'''
#test
initial()
time.sleep(1)
try:
	while True:
		#print( 'Distance: %0.2f cm', checkdist() )
		print( 'Distance:', checkdist(), ' cm' )
		
		time.sleep(0.5)
except KeyboardInterrupt:
	print ("\nThis is End.")
	GPIO.cleanup()
'''