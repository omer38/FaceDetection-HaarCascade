import cv2 as cv

cap = cv.VideoCapture(0)
frontal_face_cascade = cv.CascadeClassifier("frontalface.xml")

while True:
    ret,frame = cap.read()
    frame = cv.flip(frame,1)

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    faces = frontal_face_cascade.detectMultiScale(gray,1.6,6)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow("Frame",frame)

    if cv.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()