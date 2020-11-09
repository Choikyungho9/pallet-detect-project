import cv2
import numpy as np



def nothing(x):
    # any operation
    pass

cap = cv2.VideoCapture("pallet3.mp4")

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L-S", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U-S", "Trackbars", 225, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 50, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    _, frame = cap.read()
    dst = cv2.bitwise_not(frame)
    frame = cv2.resize(dst, (512, 512))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = ~hsv;
    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    # Contours detection
    if int(cv2.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Opencv 3.x.x
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.065*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area > 1000:
            cv2.drawContours(frame, [approx], 0, (1, 10, 100), 5)

            if len(approx) == 4:
                cv2.putText(frame, "Rectangle", (x, y), font, 0, (10, 10, 10))
                if cv2.contourArea(cnt) < 1500:
                    continue
            elif 10 < len(approx) < 1:
                cv2.putText(frame, "Circle", (x, y), font, 0, (0, 0, 0))


    def cropImage(frame):
        height, width, _ = frame.shape
        points = np.array([
            [(0, height), (127, 220), (265, 220), (width, 400), (width, height)]])

        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, np.int32([points]), True, 255, 2)
        cv2.fillPoly(mask, np.int32([points]), 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break



cap.release()
cv2.destroyAllWindows()