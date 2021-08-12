import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui as pg


##########################
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
FRAME_REDUCTION = 50 # Frame Reduction
SMOOTHING = 7
RIGHT_CLICK_FINGER = [1,0,0,0,0]
#########################

if __name__ == "__main__":
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)
    detector = htm.HandDetector(maxHands=1)
    wScr, hScr = pg.size()
    # print(wScr, hScr)

    while True:
        # 1. Find hand Landmarks
        success, img = cap.read()
        img = detector.find_hands(img)
        lmList, bbox = detector.find_position(img, draw=False)
        # 2. Get key landmarks location
        if len(lmList) != 0:
            x0, y0 = lmList[0][1:]
            x4, y4 = lmList[4][1:]
            x5 ,y5 = lmList[5][1:]
            # 3. Check which fingers are up
            fingers = detector.fingers_up()
            cv2.rectangle(img, (FRAME_REDUCTION, 100), (WINDOW_WIDTH - FRAME_REDUCTION, WINDOW_HEIGHT - FRAME_REDUCTION),
            (255, 0, 255), 2)
            # 4. Convert Coordinates
            x3 = np.interp(x0, (FRAME_REDUCTION, WINDOW_WIDTH - FRAME_REDUCTION), (0, wScr))
            y3 = np.interp(y0, (250, WINDOW_HEIGHT -FRAME_REDUCTION), (0, hScr))

            # 5. Smoothen Values
            clocX = plocX + (x3 - plocX) / SMOOTHING
            clocY = plocY + (y3 - plocY) / SMOOTHING

            # 6. Move Mouse
            pg.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x0, y0), 15, (255, 0, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY

            # 7. clicking mode
            if fingers[0] == 1 :
                # 8 . Find distance between landmarks [4] and [5]
                length, img, lineInfo = detector.find_distance(4, 5, img, draw=False)
                # 9. Left click mouse if distance short
                if length < 40:
                    pg.click()
                    print("Left Click")
                    time.sleep(0.1)
                # 10. Right click
                elif fingers == RIGHT_CLICK_FINGER:  # right_click_fingers = [1,0,0,0,0]
                    pg.rightClick()
                    print("Right Click")
                    time.sleep(0.1)


        # 11. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
        (255, 0, 0), 3)
        # 12. Display
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == 27 :
            break
    cap.release()
    cv2.destroyAllWindows()
