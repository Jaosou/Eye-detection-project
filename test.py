import cv2 as cv2 #todo : Eye detection
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector 
from cvzone.PlotModule import LivePlot

#* Web Cam Video
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1) #todo : 1 person

#* Plot Graph and Fix window
plotY = LivePlot(640,360,[25,40],invert=True) #Left
plotYRigth = LivePlot(640,360,[25,40],invert=True) #Rigth

#* Point landmark
idList = [22,23,24,26,110,157,158,159,160,161,130] 
idListRigth = [362,381,380,374,373,390,263,388,386,384, 385,387]

listRatio = []
listRatioRigth = []
blinkCount = 0
counter = 0

#* Main
while True :
    

    ret, frame = cap.read()
    frame ,face = detector.findFaceMesh(frame,draw=False)

    if face :
        face = face[0]
        #* Mark point on Face
        for id in idList:
            cv2.circle(frame,face[id],2,(255,0,255),cv2.FILLED)
        for id in idListRigth:
            cv2.circle(frame,face[id],2,(255,0,255),cv2.FILLED)

    #* Left Eye
    try :
        leftUp = face[159]
        leftDown = face[23]
        leftleft = face[130]
        leftRight = face[243]
    except IndexError:
        continue

    #* Rigth Eye
    try :
        rigthUp = face[386]
        rigthDown = face[374]
        rigthleft = face[362]
        rigthRight = face[263]
    except:
        continue

    #*Horizon length Left
    lengthHor,_ = detector.findDistance(leftUp, leftDown)
    """ cv2.line(frame,leftUp,leftDown, (0,200,0),2) """
    #* Vertical Length Left
    lengthVer,_ = detector.findDistance(leftleft,leftRight)
    """ cv2.line(frame,leftleft,leftRight, (0,200,0),2) """

    #* Process Left eye
    ratio = int((lengthHor/lengthVer)*100)
    listRatio.append(ratio)
    if len(listRatio)>= 5 : listRatio.pop(0)
    ratioAvg = sum(listRatio)/len(listRatio)

    #*Horizon length Rigth
    lengthHorRigth,_ = detector.findDistance(rigthUp, rigthDown)
    """ cv2.line(frame,leftUp,leftDown, (0,200,0),2) """
    #* Vertical Length Left
    lengthVerRigth,_ = detector.findDistance(rigthleft,rigthRight)
    """ cv2.line(frame,leftleft,leftRight, (0,200,0),2) """

    #* Process Rigth eye
    ratioRigth = int((lengthHorRigth/lengthVerRigth)*100)
    listRatioRigth.append(ratioRigth)
    if len(listRatioRigth)>= 5 : listRatioRigth.pop(0)
    ratioAvgRigth = sum(listRatioRigth)/len(listRatioRigth)

    #* Blink Count
    if ratioAvg < 31.5 and counter == 0: 
        blinkCount+=1
        counter = 1
    if counter != 0:
        counter +=1 
        if counter > 25:
            counter = 0

    #* PUT Text
    cvzone.putTextRect(frame,f"Bilnk Count : {blinkCount}",(100,100))


    #Todo : Plot Left
    imgPlot = plotY.update(ratioAvg)
    frame = cv2.resize(frame,(700,540))
    imgStack = cvzone.stackImages([frame,imgPlot],1,1)
    cv2.imshow("StackIm",imgStack)

    #Todo : Plot Rigth
    imgPlotRigth = plotYRigth.update(ratioAvgRigth)
    frame = cv2.resize(frame,(700,540))
    imgStackRigth = cvzone.stackImages([frame,imgPlotRigth],1,1)
    cv2.imshow("StaclImRigth",imgStackRigth)

    """ cv2.imshow('Eye Detewction', frame) """
    """ frame = cv2.resize(frame,(1080,720)) """
    """ cv2.imshow("Frame",frame) """


    # ออกจากลูปเมื่อกด 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()