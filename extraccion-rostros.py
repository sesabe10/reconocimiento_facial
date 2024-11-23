import cv2;
import os;

rutaImagenes = "./imagenes"


if not os.path.exists("rostros"):
    os.makedirs("rostros");
    print("Se ha creado la carpeta rostros");


detectorRostro = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");

contador = 0;
for nombreImagen in os.listdir(rutaImagenes):
    print(nombreImagen);
    imagen = cv2.imread(rutaImagenes + "/" + nombreImagen);
    # Detectamos los rostros
    rostros = detectorRostro.detectMultiScale(imagen, 1.1, 5);
    # Guardamos los rostros en la carpeta "rostros"
    for(x, y, w, h) in rostros:
        rostro = imagen[y:y + h, x:x + w];
        rostro = cv2.resize(rostro, (150, 150));
        cv2.imwrite("rostros/"+str(contador)+".jpg", rostro);
        contador += 1;
    
cv2.destroyAllWindows();