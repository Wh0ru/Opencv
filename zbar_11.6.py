
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 6 09:13:34 2018

@author: bongos
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
# from pyzbar.pyzbar import decode
import serial


ser=serial.Serial(
    port='/dev/ttyS4',
    baudrate=115200,)
ser.close()
ser.open()
ser.isOpen()

def decode(img1,img2):
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    MIN_MATCH_COUNT=20
    
    sift=cv2.xfeatures2d.SURF_create(700)
    kp1,des1=sift.detectAndCompute(img1,None)
    kp2,des2=sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE=0
    index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    search_params=dict(checks=50)

    flann=cv2.FlannBasedMatcher(index_params,search_params)

    try:
        matches=flann.knnMatch(des1,des2,k=2)
    except:
        return False
    good=[]
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,2.0)

        h,w=img1.shape
        pts1=np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

        try:
            pts2=cv2.perspectiveTransform(pts1,M)
#             W,H=(pts2[2][0][0]-pts2[0][0][0]),(pts2[2][0][1]-pts2[0][0][1])
#             if W*H<50:
# #                 print('error1!')
#                 return False
            return pts2.reshape(-1,2)
#             if ((H-50<=W<=H+50) and (W-50<=H<=W+50)):
#                 return pts2.reshape(-1,2)
#             else:
# #                 print('error2!')
#                 return False
        except:
            return False
    else:
#         print('Not enough matches are found - %d/%d'%(len(good),MIN_MATCH_COUNT))
        return False

def plot(img,decodes):
    dst=img.copy()
    for d in decodes:
        pts=np.array(d.polygon)
        pts=pts.reshape(-1,1,2)
        dst=cv2.polylines(dst,[pts],True,(0,0,255),2)
    return dst

def plots(img,pts):
    if pts is not False:
        pts=pts.reshape(-1,1,2)
        dst=cv2.polylines(img.copy(),[np.int32(pts)],True,(0,0,255),2,cv2.LINE_AA)
        return dst
    else:
        return None

def zbar_dete(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    decodes=decode(img)
    if len(decodes)>0:
        # for decode in decodes:
        #     if decode.data == b'':
        #         return decode.polygon
        return decodes[0].polygon
    else:
        return False

def sub_lr(pts):
    leftH=pts[1][1]-pts[0][1]
    rightH=pts[2][1]-pts[3][1]
    if leftH<0 or rightH<0:
        return False
    leftW=abs(pts[0][0]-pts[1][0])
    rightW=abs(pts[2][0]-pts[3][0])
    return [leftW,leftH,rightW,rightH]

def calculate(leftW,leftH,rightW,rightH,pts):
    tanHalfView=np.tan(hViewAngle/2)
    leftLen=np.sqrt(leftW*leftW+leftH*leftH)
    rightLen=np.sqrt(rightW*rightW+rightH*rightH)
    leftZ=L*h/(2*leftLen*tanHalfView)
    rightZ=L*h/(2*rightLen*tanHalfView)
    z=(leftZ+rightZ)/2
    
    b=np.arcsin((leftZ-rightZ)/L)
    
    k1=(pts[2][1]-pts[0][1])/(pts[2][0]-pts[0][0])
    b1=(pts[2][0]*pts[0][1]-pts[0][0]*pts[2][1])/(pts[2][0]-pts[0][0])
    k2=(pts[3][1]-pts[1][1])/(pts[3][0]-pts[1][0])
    b2=(pts[3][0]*pts[1][1]-pts[1][0]*pts[3][1])/(pts[3][0]-pts[1][0])
    crossX=-(b1-b2)/(k1-k2)
    a=np.arctan(((2*crossX-w)/h)*tanHalfView)
    return a,b,z

#摄像头上下视角
hViewAngle=0.56
#二维码边长
L=8
#是否开启调试窗口
debug=True

cap=cv2.VideoCapture(1)

img1=cv2.imread('p_18.png')

while cap.isOpened:
    ret,frame=cap.read()
    decode_img=cv2.GaussianBlur(frame,(5,5),0)
    h,w=frame.shape[:2]
    # pts=zbar_dete(decode_img)
    pts=decode(img1,decode_img)

    if pts is not False:
        length=sub_lr(pts)
        if length is not False:
            leftW,leftH,rightW,rightH=length
            a,b,z=calculate(leftW,leftH,rightW,rightH,pts)
            a=a*180/np.pi
            b=b*180/np.pi
            if debug:
                # dst=plot(frame,decode(cv2.cvtColor(decode_img,cv2.COLOR_BGR2GRAY)))
                dst=plots(frame,pts)
                if dst is not None:
                    cv2.imshow('debugui',dst)
                else:
                    cv2.imshow('debugui',frame)
                '''
                a:二维码中心所在铅垂线与O点构成的平面和Z轴形成的夹角
                b:二维码所在平面与X轴构成的夹角
                z:二维码中心到XOY平面的距离
                '''
                if  (-180<=a<=180) and (-180<=b<=180) and z != np.inf:
                    print('a:%f,b:%f,z:%f'%(a,b,z))
                    res='a:{},b:{},z:{}'.format(a,b,z)
                    ser.write(str.encode(res))
            else:
                if  (-180<=a<=180) and (-180<=b<=180) and z != np.inf:
                    print('a:%f,b:%f,z:%f'%(a,b,z))
                    res='a:{},b:{},z:{}'.format(a,b,z)
                    ser.write(str.encode(res))
        else:
            if debug:
                # dst=plot(frame,decode(cv2.cvtColor(decode_img,cv2.COLOR_BGR2GRAY)))
                dst=plots(frame,pts)
                if dst is not None:
                    cv2.imshow('debugui',dst)
                else:
                    cv2.imshow('debugui',frame)
    else:
        if debug:
            # dst=plot(frame,decode(cv2.cvtColor(decode_img,cv2.COLOR_BGR2GRAY)))
            dst=plots(frame,pts)
            if dst is not None:
                cv2.imshow('debugui',dst)
            else:
                cv2.imshow('debugui',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
ser.close()
