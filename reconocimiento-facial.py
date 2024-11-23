import cv2;
import os;
import face_recognition;

#Codificar los rostros extraídos
rutaRostros = "./rostros";
codigosRostros = [];
nombresRostros = [];

for nombreImagen in os.listdir(rutaRostros):
    imagen = cv2.imread(rutaRostros + "/" + nombreImagen);
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB);
    codigoRostro = face_recognition.face_encodings(imagen, known_face_locations=[(0, 150, 150, 0)])[0];
    codigosRostros.append(codigoRostro);
    nombresRostros.append(nombreImagen.split(".")[0]);

captura = cv2.VideoCapture(0);


detectorRostro = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");

while True:
    ret, frame = captura.read();
    if ret == False:
        break;
    frame = cv2.flip(frame, 1);
    frameCopy = frame.copy();
    rostros = detectorRostro.detectMultiScale(frame, 1.3, 7);
    for (x, y, w, h) in rostros:
        rostro = frameCopy[y:y+h, x:x + w];
        rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB);
        codigoRostroActual = face_recognition.face_encodings(rostro, known_face_locations=[(0, w, h, 0)])[0];
        resultado = face_recognition.compare_faces(codigosRostros, codigoRostroActual);
        if True in resultado:
            indice = resultado.index(True);
            nombre = nombresRostros[indice];
            color = (125, 220, 0);
        else:
            nombre = "Desconocido";
            color= (50, 50, 255);
        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), color, -1);
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2);
        cv2.putText(frame, nombre, (x, y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA);
    cv2.imshow("Frame", frame);
    k = cv2.waitKey(1) & 0xFF;
    if k == 1:
        break;

captura.release();
cv2.destroyAllWindows();