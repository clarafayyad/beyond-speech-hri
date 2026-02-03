import cv2
from video.camera_stream import CameraStream
from video.facial_affect import FacialAffectService

camera = CameraStream(0)
affect = FacialAffectService(analyze_every_n_frames=12)

while True:
    frame = camera.read()
    if frame is None:
        break

    valence = affect.process_frame(frame)

    cv2.putText(
        frame,
        f"Affect: {valence}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Facial Affect (DeepFace)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
