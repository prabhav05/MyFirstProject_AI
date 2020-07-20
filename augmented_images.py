import os,cv2,sys
label_name=sys.argv[1]
freq=int(sys.argv[2])
IMG_SAVE_PATH="image_data"
IMG_class_path=os.path.join(IMG_SAVE_PATH,label_name)
try:
    os.mkdir(IMG_SAVE_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(IMG_class_path)
except FileExistsError:
    print("{} directory already exists".format(IMG_class_path))
cap=cv2.VideoCapture(0)
begin=False
count=0

while(True):
    ret,frame=cap.read()
    if(not ret):
        continue
    if(count==freq):
        break

    cv2.rectangle(frame,(100,100),(500,500),(255,255,255),2)
    if(begin):
        sliced_frame=frame[100:500,100:500]
        save_path=os.path.join(IMG_class_path,"{}.jpg".format(count+1))
        cv2.imwrite(save_path,sliced_frame)
        count+=1

    cv2.putText(frame,"Gathering {}".format(count),(5,50),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,255,255),2,cv2.LINE_8)
    cv2.imshow("Collecting Images",frame)
    k=cv2.waitKey(15)
    if(k==ord("a")):
        begin=not begin
    if(k==ord("q")):
        break
print("\n {} Image saved to {}".format(count,IMG_class_path))
cap.release()
cv2.destroyAllWindows()

