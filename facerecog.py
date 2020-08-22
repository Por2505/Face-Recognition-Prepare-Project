import cv2
#import requests
#from firebase import firebase


#url = 'https://notify-api.line.me/api/notify'
#token = 'tZw8sOvcem5GSxxkDNTsqJLdSOSv9WUenjnRrH8PdYw'
#headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+token}
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#firebase = firebase.FirebaseApplication('https://image-cfdb0.firebaseio.com/image-cfdb0', None)

def create_dataset(img,id,img_id):
      cv2.imwrite("data/pic."+str(id)+"."+str(img_id)+".jpg",img)
      

def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,clf): #clf
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      features = classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
      coords=[]
      for(x,y,w,h) in features:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            id,con = clf.predict(gray[y:y+h,x:x+w])
            #print(id)
            
            if id ==1 :
                  
                  if con<=100:
                        cv2.putText(img,"Por",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
                        #result = firebase.put('', 'user1','por')
                        #print(result)
                  else :
                        cv2.putText(img,"Unknow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)

                  if (con < 100) :
                        con = " {0}%".format(round(100 - con))
                  else :
                        con = " {0}%".format(round(100 - con))
                  print(str(con))
                  coords=[x,y,w,h]
            if id ==2 :
                  
                  if con<=100:
                        cv2.putText(img,"Meen",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
                        #result = firebase.put('', 'user1','por')
                        #print(result)
                  else :
                        cv2.putText(img,"Unknow",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)

                  if (con < 100) :
                        con = " {0}%".format(round(100 - con))
                  else :
                        con = " {0}%".format(round(100 - con))
                  print(str(con))
                  coords=[x,y,w,h]
                  
                   
      return img,coords

      
def detect(img,faceCascade,img_id,clf): #clf
      img,coords= draw_boundary(img,faceCascade,1.1,10,(255,0,0),clf)
      #img,coords= draw_boundary(img,faceCascade,1.1,10,(255,0,0),"Face")
      id=1
      if len(coords)==4:
            id =1
            result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
            #y+h
            #create_dataset(result,id,img_id,clf)
      return img

img_id=0

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('rtsp://192.168.1.2:8080/h264_ulaw.sdp')

clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

while(True):
      ret,frame = cap.read()
      if ret == True:
            frame=detect(frame,faceCascade,img_id,clf)
            cv2.imshow('frame',frame)
            img_id+=1
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                  break


cap.release()
#cv2.destoyAllWindows()
cv2.waitKey(0)

