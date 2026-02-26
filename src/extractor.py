import cv2
import os
import yt_dlp

VIDEO_URL = 'https://www.youtube.com/watch?v=1rRs1rRGhxI'
OUTPUT_DIR = '../raw_frames'
FRAME_INTERVAL = 300 # Saving frequency

def download_and_extract():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(VIDEO_URL, download=False)
        url = info['url']
        print(f"Video URL extracted: {info['title']}")

    cap = cv2.VideoCapture(url)
    count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_INTERVAL == 0:
            frame_resized = cv2.resize(frame, (400, 400))
            img_path = os.path.join(OUTPUT_DIR, f"frame_{saved_count}.jpg")
            cv2.imwrite(img_path, frame_resized)
            saved_count += 1
            if saved_count % 20 == 0:
                print(f"Saved {saved_count}", end='\r')
        
        count += 1

    cap.release()
    print(f"\nExtraction succesful. Total frames saved: {saved_count}")

if __name__ == "__main__":
    download_and_extract()