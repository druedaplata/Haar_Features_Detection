import cv2  # OpenCV Library
 
#-----------------------------------------------------------------------------
#       Cargar y Configurar los clasificadores en Cascada
#-----------------------------------------------------------------------------
 
# Archivos xml describiendo nuestros clasificadores en cascada
faceCascadeFilePath = "haarcascade_frontalface_default.xml"
noseCascadeFilePath = "haarcascade_mcs_nose.xml"
 
# Cargar los clasificadores
faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)
 
#-----------------------------------------------------------------------------
#       Cargar y configurar el BIGOTE (.png con transparencia alpha)
#-----------------------------------------------------------------------------
 
# Cargar la imagen que usaremos: mustache.png
imgMustache = cv2.imread('mustache.png',-1)
 
# Crear la mascara para el bigote
orig_mask = imgMustache[:,:,3]
 
# Crear la mascara invertida para el bigote
orig_mask_inv = cv2.bitwise_not(orig_mask)
 
# Convertir la imagen del bigote a BGR 
# y guardar el original de la imagen (usada cuando cambiamos la imagen)
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
 
#-----------------------------------------------------------------------------
#       Main
#-----------------------------------------------------------------------------
 
# Recibir video de la entrada
video_capture = cv2.VideoCapture(0)
 
while True:
    # Capturar la entrada de video
    ret, frame = video_capture.read()
 
    # Crear la imagen en escala de grises de la entrada de video. 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Detectar caras en la entrada de video
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
 
   # Iterar sobre cada imagen encontrada 
    for (x, y, w, h) in faces:
        # Descomentar la siguiente linea para debug (dibuja una caja en todas las caras )
        # face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
 
        # Detectar una nariz en la zona de cada cara
        nose = noseCascade.detectMultiScale(roi_gray)
 
        for (nx,ny,nw,nh) in nose:
            # Descomentar la siguiente linea para debug (dibuja una caja en todas las narices)
            #cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
 
            # El bigore debe ser 3 veces el ancho de la nariz
            mustacheWidth =  3 * nw
            mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth
 
            # Centrar el bigote en la base de la nariz
            x1 = nx - (mustacheWidth/4)
            x2 = nx + nw + (mustacheWidth/4)
            y1 = ny + nh - (mustacheHeight/2)
            y2 = ny + nh + (mustacheHeight/2)
 
            # Revisar para cortar bordes
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h
 
            # Re-calcular el ancho y altura de la imagen del bigote
            mustacheWidth = x2 - x1
            mustacheHeight = y2 - y1
 
            # Cambiar la imagen original y las mascaras de los bigotes 
            # calculados anteriormente
            mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
 
            # Tomar la zona de interes ROI para el bigote del fondo igual a la imagen del bigote
            roi = roi_color[y1:y2, x1:x2]
 
            # roi_bg contiene la imagen original solo donde el bigote 
            # en la zona que es igual a la del bigote
            roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
 
            # roi_fg contiene la imagen de el bigote solo donde el bigote 
            roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
 
            # unir roi_bg y roi_fg
            dst = cv2.add(roi_bg,roi_fg)
 
            # ubicar la imagen que unimos, y guardar en el destino sobre la imagen original
            roi_color[y1:y2, x1:x2] = dst
 
            break
 
    # Mostrar la imagen resultante
    cv2.imshow('Video', frame)
 
    # press any key to exit
    # NOTE;  x86 systems may need to remove: &quot;&amp; 0xFF == ord('q')&quot;
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()




