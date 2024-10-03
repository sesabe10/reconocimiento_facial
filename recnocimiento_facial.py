import cv2 as cv

cascadePath = 'haarcascade_frontalface_default.xml'
cascadeeyes = 'haarcascade_eye.xml'
faceCascade = cv.CascadeClassifier(cascadePath)
eye_cascade = cv.CascadeClassifier(cascadeeyes)

cap = cv.VideoCapture(0)

while True:
    # Leer un frame del video
    ret, img = cap.read()

    # Convertir el img a escala de grises
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (25, 25),
        flags = cv.CASCADE_SCALE_IMAGE
    )

    #Dibujando el rectangulo sobre el rostro detectado.
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_color = img[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    # Mostrar el video con los rostros detectados
    cv.imshow('Detección de Rostros', img)

    # Salir del bucle si se presiona la tecla 'q'
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()