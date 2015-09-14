import cv2
import matplotlib.pyplot as plt

def video_reception(classifier1, classifier2, scale_factor=1.1, min_neighbors=5, min_size=(64,64)):
    
    # Cargar el clasificador
    cascade1 = cv2.CascadeClassifier(classifier1)
    cascade2 = cv2.CascadeClassifier(classifier2)    

    # Recibir video de la entrada
    video_capture = cv2.VideoCapture(0)


    while True:
        # Capturar la entrada de video
        ret, frame = video_capture.read()
 
        # Crear la imagen en escala de grises de la entrada de video. 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # Detectar palma en la entrada de video
        features1 = cascade1.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

	# Detectar fist en la entrada de video
	features2 = cascade2.detectMultiScale(
	    gray,
	    scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
 
        # Iterar sobre cada imagen encontrada 
        for (x, y, w, h) in features1:
            # Descomentar la siguiente linea para debug (dibuja una caja en todas las caras )
            feature = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
	# Iterar sobre cada imagen encontrada 
        for (x, y, w, h) in features2:
            # Descomentar la siguiente linea para debug (dibuja una caja en todas las caras )
            feature = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # Mostrar la imagen resultante
        cv2.imshow('Video', frame)
 
        # press any key to exit
        # NOTE;  x86 systems may need to remove: "& 0xFF == ord('q')"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    
video_reception("palm.xml", "fist.xml")
