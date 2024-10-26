import cv2 as cv
import time

# Rutas de los clasificadores
cascadePath = 'haarcascade_frontalface_default.xml'
cascadeeyes = 'haarcascade_eye.xml'
faceCascade = cv.CascadeClassifier(cascadePath)
eye_cascade = cv.CascadeClassifier(cascadeeyes)

cap = cv.VideoCapture(0)

# Establecer una resolución más baja
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    start_time = time.time()  # Iniciar el cronómetro

    # Leer un frame del video
    ret, img = cap.read()
    img = cv.flip(img, 1)

    # Convertir el img a escala de grises
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # Dibujando el rectángulo sobre el rostro detectado
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_color = img[y:y + h, x:x + w]
        roi_gray = gray[y:y + h, x:x + w]

        # Limitar el área de búsqueda para los ojos a la parte superior de la cara
        eyes_region = roi_gray[int(h / 8):int(h / 2), 0:w]
        eyes_color_region = roi_color[int(h / 8):int(h / 2), 0:w]

        # Detectando ojos en la región limitada
        eyes = eye_cascade.detectMultiScale(eyes_region, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            aspect_ratio = ew / eh
            if 0.5 < aspect_ratio < 2.5:
                cv.rectangle(eyes_color_region, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Mostrar el video con los rostros detectados
    cv.imshow('Detección de Rostros', img)

    # Calcular el tiempo de procesamiento
    elapsed_time = time.time() - start_time
    wait_time = max(1, int(1000 / 60 - elapsed_time * 1000))  # Esperar para alcanzar 30 FPS

    # Salir del bucle si se presiona la tecla 'q'
    if cv.waitKey(wait_time) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
