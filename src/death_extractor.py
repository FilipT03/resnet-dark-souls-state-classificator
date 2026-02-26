import cv2
import os
import yt_dlp

VIDEO_URL = 'https://www.youtube.com/watch?v=1rRs1rRGhxI'
OUTPUT_DIR = '../death_frames'

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

        x, y = 650, 575 # Coordinates of a red pixel in Y of YOU DIED screen
        x = int(x * frame.shape[1] / 1920)
        y = int(y * frame.shape[0] / 1080)
        pixel = frame[y, x]
        if pixel[2] > 100 and pixel[0] < 15 and pixel[1] < 15:
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