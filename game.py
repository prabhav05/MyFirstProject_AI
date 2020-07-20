from keras.models import load_model
import numpy as np
import cv2
from random import choice

rev_class={0:"rock",1:"paper",2:"scissor",3:"none"}
def rev_class_map(value):
    return rev_class[value]

def winner(move1,move2):

    if(move1==move2):
        return "Tie"
    if(move1=="rock"):
        if(move2=="scissor"):
            return "Prabhav"
        else:
            return "Computer"
    
    if(move1=="paper"):
        if(move2=="rock"):
            return "Prabhav"
        else:
            return "Computer"

    if(move1=="scissor"):
        if(move2=="paper"):
            return "Prabhav"
        else:
            return "Computer"

model=load_model("rock_paper_scissor_model.h5")
cap=cv2.VideoCapture(0)
Winner="yo"
prev_move=None
while True:
        ret,frame=cap.read()
        if not ret:
            continue
        cv2.rectangle(frame,(50,100),(400,430),(255,255,255),2)
        cv2.rectangle(frame,(450,100),(600,430),(255,255,255),2)
        roi = frame[50:400, 100:430]
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))

        pred = model.predict(np.array([img]))
        move_code = np.argmax(pred[0])
        user_move_name = rev_class_map(move_code)

        if(prev_move!=user_move_name):
            if(user_move_name!="none"):
                computer_move_name=choice(["rock","paper","scissor"])
                Winner=winner(user_move_name,computer_move_name)
            else:
                computer_move_name="none"
                Winner="Waiting......."

        prev_move=user_move_name

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Prabhav's Move: " + user_move_name,
                (50, 50), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (350, 50), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Winner: " + Winner,
                    (250, 470), font, 1, (46, 23, 240), 2, cv2.LINE_AA)

        if(computer_move_name!="none"):
                icon=cv2.imread("images/{}.png".format(computer_move_name))
                icon= cv2.resize(icon,(500,500) , interpolation = cv2.INTER_AREA)
        cv2.imshow("Rock Paper Scissor", frame)
        k = cv2.waitKey(10)
        if( k == ord('q')):
                break
cap.release()
cv2.destroyAllWindows()