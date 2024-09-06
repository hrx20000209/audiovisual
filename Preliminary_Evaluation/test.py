import cv2

fps = 20
img_size = (1080, 1920)  # h, w

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video_save_path = './demo.avi'
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, fps)
cap.set(cv2.CAP_PROP_FOURCC, fourcc)  # 还需要这句话
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_size[1])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size[0])
videowriter = cv2.VideoWriter(video_save_path, fourcc, fps, (img_size[1], img_size[0]))

fps = int(cap.get(5))
print(f"fps:{fps}")
while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()

    # 显示帧
    cv2.imshow('frame', frame)
    videowriter.write(frame)

    # 按'q'键退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放VideoCapture对象和所有窗口
cap.release()
cv2.destroyAllWindows()
