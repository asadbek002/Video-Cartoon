import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

def color_quantization(frame, K=8):
    quantized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Z = quantized.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((quantized.shape))
    return res2

def cartoon_effect(frame, K=8):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 50, 150)
    res2 = color_quantization(frame, K)
    bilateral = cv2.bilateralFilter(res2, 9, 75, 75)
    cartoon = cv2.addWeighted(bilateral, 0.8, frame, 0.2, 0)
    return cartoon


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('NeW YORK.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        cartoon = cartoon_effect(frame, 8)
        out.write(cartoon)
        cv2.imshow('Cartoon Video',cartoon)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

