import urllib.request, os
url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
dest = "data/hand_landmarker.task"
print("Downloading hand_landmarker.task ...")
urllib.request.urlretrieve(url, dest)
print(f"Done: {os.path.getsize(dest)/1e6:.1f} MB")
