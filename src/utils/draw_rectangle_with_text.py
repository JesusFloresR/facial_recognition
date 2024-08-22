import cv2

def draw_rectangle_with_text(img, text, xmin, ymin, xmax, ymax):
    # Dibujar el rectángulo
    color_rect = (0, 255, 0)  # Color del rectángulo (verde en BGR)
    thickness_rect = 2  # Grosor del rectángulo
    cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax), int(ymax)), color_rect, thickness_rect)

    # Agregar el texto
    color_texto = (255, 0, 0)  # Color del texto (rojo en BGR)
    pos_texto = (int(xmin), int(ymin) - 10)  # Posición del texto (dibuja encima del rectángulo)
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    tam_fonte = 0.9
    grosor_texto = 2

    cv2.putText(img, text, pos_texto, fuente, tam_fonte, color_texto, grosor_texto)

    return img