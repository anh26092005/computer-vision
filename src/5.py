import cv2

# Đọc ảnh
img = cv2.imread("../input/lab_image.jpg")

# Vẽ đường thẳng
cv2.line (img,(100,100),(400,100),(255,0,0),3)

# Vẽ hình chữ nhật
cv2.rectangle(img,(200,200),(400,400),(0,255,0),3)

# Vẽ hình tròn
cv2.circle(img,(300,500),60,(0,0,255),3)

#Thêm chữ
cv2.putText(img,"Hello",(400,600),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

#Hiện thị 
cv2.imshow ("Image",img)

cv2.waitKey(0)
cv2.destroyAllWindows()