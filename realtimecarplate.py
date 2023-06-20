import cv2
import easyocr
import csv

harcascade = "model/haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Define the codec for video recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('recorded_video.avi', fourcc, 20.0, (640, 480))

min_area = 500
count = 0

reader = easyocr.Reader(['en'])

# Open the CSV file in write mode
csv_file = open('license_plates.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Plate Number'])  # Write the header row

while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y+h, x:x+w]
            cv2.imshow("ROI", img_roi)
            gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            plate_number = reader.readtext(gray_roi)[0][1]

            # Store the license plate number in the CSV file
            csv_writer.writerow([plate_number])

    # Write the current frame to the output video file
    output_video.write(img)
          
    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)
        count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the CSV file
csv_file.close()

# Release the video writer and webcam
output_video.release()
cap.release()

# Close the frames
cv2.destroyAllWindows()
