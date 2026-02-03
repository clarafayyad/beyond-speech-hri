import cv2
from src.audio.audio_stream import AudioStream
from video.camera_stream import CameraStream
from video.facial_affect import FacialAffectService
from audio.prosody import ProsodyService

# -----------------------------
# Initialize video
# -----------------------------
camera = CameraStream(0)
affect = FacialAffectService(analyze_every_n_frames=12)

# -----------------------------
# Initialize audio
# -----------------------------
prosody_service = ProsodyService(smoothing_window=2)
audio = AudioStream(prosody_service)

# -----------------------------
# Main loop (video + display)
# -----------------------------
try:
    audio.start()
    while True:
        frame = camera.read()
        if frame is None:
            break

        # --- Video: Facial Affect ---
        valence = affect.process_frame(frame)

        # --- Overlay results ---
        cv2.putText(frame, f"Affect: {valence}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Fluency: {audio.prosody.fluency}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.putText(frame, f"Arousal: {audio.prosody.arousal}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Multimodal Perception", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Cleanup
    camera.release()
    audio.release()
    cv2.destroyAllWindows()
