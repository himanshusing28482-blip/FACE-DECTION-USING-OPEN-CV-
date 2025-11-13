# pip install opencv-python
# haarcascade_frontalface_default.xml
# pip install opencv-python
import cv2 

# Load the Haar Cascade XML file
a = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open the webcam (0 for default camera)
b = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    c_rec, d_image = b.read()

    # Check if the frame is successfully captured
    if not c_rec:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame to grayscale
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    f = a.detectMultiScale(e, scaleFactor=1.3, minNeighbors=6)

    # Draw rectangles around detected faces
    for (x1, y1, w1, h1) in f:
        cv2.rectangle(d_image, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 5)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', d_image)

    # Exit the loop when the 'Esc' key is pressed
    h = cv2.waitKey(40) & 0xFF
    if h == 27:  # ASCII code for Esc key
        break

# Release the webcam and close the window
b.release()
cv2.destroyAllWindows()
