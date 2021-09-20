import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# img = cv.imread("images/fish.png")
# cv.imshow("OpenCV Demo", img)
# cv.waitKey(0)

capture = cv.VideoCapture("videos/HaloBT.mp4")
while True:
    isTure, frame = capture.read()
    resizedFrame = rescaleFrame(frame)
    cv.imshow("Halo BehaviourTree", resizedFrame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
