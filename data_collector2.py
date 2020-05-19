import cv2

camera = cv2.VideoCapture(0)
number = 50
count = 0
while True:
    count += 1
    return_value, image = camera.read()
    cv2.resize(image, (400,400))
    cv2.imwrite('C://Users//smc181002//Desktop//MlOpsTraining//mltask_transf_learn//test1//grant//opencv'+str(count)+'.png', image)
    cv2.putText(image,
        str(count),
        (50, 50),
        cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Face Cropper', image)
    if cv2.waitKey(1) == 13 or count == number: #13 is the Enter Key
        break

camera.release()
cv2.destroyAllWindows()
print("images collected")