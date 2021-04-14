import cv2
import sys
import pyprind

cap1 = cv2.VideoCapture(sys.argv[1])
cap2 = cv2.VideoCapture(sys.argv[2])

fps = cap1.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(sys.argv[1][:-4] + '_and_' + sys.argv[2], fourcc, fps, (w * 2, h))

total = cap1.get(cv2.CAP_PROP_FRAME_COUNT)
iter_bar = pyprind.ProgBar(total, title='Concat', stream=sys.stdout)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # now, you can do this either vertical (one over the other):
    # final = cv2.vconcat([img1, img2])

    # or horizontal (next to each other):
    final = cv2.hconcat([frame1, frame2])

    out.write(final)
    iter_bar.update()
out.release()
cap1.release()
cap2.release()
