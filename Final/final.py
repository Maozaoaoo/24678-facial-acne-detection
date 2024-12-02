import cv2
import numpy as np
import dlib
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class FaceMasker:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def resize_image(self, image, target_width=500):
        h, w = image.shape[:2]
        aspect_ratio = float(h) / float(w)
        target_height = int(target_width * aspect_ratio)
        return cv2.resize(image, (target_width, target_height))


    def mask_out_mouth_and_eyes(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)

            # Mouth points
            mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            mouth_mask = np.zeros_like(gray)
            cv2.fillPoly(mouth_mask, [np.array(mouth_points, dtype=np.int32)], 255)

            # Fill the gap between the lips
            hull = cv2.convexHull(np.array(mouth_points, dtype=np.int32))
            cv2.fillConvexPoly(mouth_mask, hull, 255)

            # Dilate the mask to ensure the lips are fully covered
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=7)

            # Eye points
            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            eye_mask = np.zeros_like(gray)
            cv2.fillPoly(eye_mask, [np.array(left_eye_points, dtype=np.int32)], 255)
            cv2.fillPoly(eye_mask, [np.array(right_eye_points, dtype=np.int32)], 255)

            # Dilate the eye masks to ensure the eyes are fully covered
            eye_mask = cv2.dilate(eye_mask, kernel, iterations=3)

            # Combine mouth and eye masks
            combined_mask = cv2.bitwise_or(mouth_mask, eye_mask)

            # Invert the mask to mask out the mouth and eyes
            inverted_combined_mask = cv2.bitwise_not(combined_mask)
            masked_image = cv2.bitwise_and(image, image, mask=inverted_combined_mask)

        return masked_image

    def mask_face(self, image):
        """Mask out the background and keep only the face region, including the forehead."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        face_mask = np.zeros_like(gray)
        for face in faces:
            # Extend the face rectangle upwards to include the forehead
            forehead_extension = int((face.bottom() - face.top()) * 0.25)
            top = max(0, face.top() - forehead_extension)
            cv2.rectangle(face_mask, (face.left(), top), (face.right(), face.bottom()), 255, -1)

        # Apply the face mask to the image
        masked_face = cv2.bitwise_and(image, image, mask=face_mask)
        return masked_face

class AcneDetector:
    def __init__(self):
        self.red_params = {
            'top_percentage': 0.001,  # Increase to top 0.1% for redness
        }

    def find_red_regions(self, a_channel):
        """Find red regions and expand the mask using dilation."""
        threshold = np.percentile(a_channel, 100 - (self.red_params['top_percentage'] * 100))
        red_mask = (a_channel >= threshold).astype(np.uint8) * 255

        # Expand the red mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.dilate(red_mask, kernel, iterations=3)  # Expands redness regions
        return red_mask

    def visualize_a_channel(self, image):
        """Create and display a color map for the a-channel of the LAB color space."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)

        # Normalize the a-channel
        a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)

        return a_normalized

    def calculate_simi(self, image1, image2):
        # Convert both images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Calculate the histogram for both images
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        # Compare histograms using correlation
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity

    def class_acne(self,image,red_mask):
        """Compare the ance to the ground truth images to classify them"""
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ground_truth_images = [cv2.imread('papule.jpg'), cv2.imread('pustule.jpg'), cv2.imread('nodule.jpg')]
        acne_rectangle = []
        for i in contours:
            x, y, w, h = cv2.boundingRect(i)
            rect = image[y:y+h, x:x+w]
            acne_rectangle.append(rect)

        similarities = []
        num_papule = 0
        num_pustule = 0
        num_nodule = 0
        for test_image in acne_rectangle:
            for i, ground_truth_image in enumerate(ground_truth_images):
                similarity = self.calculate_simi(test_image, ground_truth_image)
                similarities.append((i + 1, similarity))
                # Find the ground truth image with the highest similarity
                best_match = max(similarities, key=lambda x: x[1])
                if best_match[0] == 1:
                    num_papule += 1
                elif best_match[0] == 2:
                    num_pustule += 1
                else:
                    num_nodule += 1
        print("Number of papule: "+str(num_papule))
        print("Number of pustule: "+str(num_pustule))
        print("Number of nodule: "+str(num_nodule))

    def plot_contours(self, image, red_mask):
        """Plot contours around the detected red regions."""
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)  # Draw contours with green color and thickness 2

    def eye_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        face = face_cascade.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in face:
            new_gray = gray[y:y + int(h / 2), x:x + w]
            new_color = image[y:y + int(h / 2), x:x + w]

            # After determine the face area, find eyes' position within it
            eyes = eye_cascade.detectMultiScale(new_gray)

            # draw rectangles
            for (eye_x, eye_y, eye_w, eye_h) in eyes:
                cv2.rectangle(new_color, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 255), 2)
                # cv2.rectangle(new_color,(ex-int(ew/2),ey-int(eh/2)),(ex+int(ew*1.5),ey+int(eh*1.5)),(0,255,255),2)

                new_area = gray[y + eye_y + int(eye_h / 2):y + eye_y + eye_h + 10, x + eye_x:x + eye_x + eye_w + 10]
                lower_bound = 120  # Lower intensity
                upper_bound = 150  # Upper intensity
                # Create a mask for the intensity range
                mask = cv2.inRange(new_area, lower_bound, upper_bound)
                filtered_image = cv2.bitwise_or(new_area, new_area, mask=mask)
                contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                adjusted_contours = []
                for contour in contours:
                    adjusted_contour = contour + [x + eye_x, y + eye_y + int(eye_h / 2)]  # Add offset to each point
                    adjusted_contours.append(adjusted_contour)
                sorted_contours = sorted(adjusted_contours, key=cv2.contourArea, reverse=True)
                cv2.drawContours(original_image, [sorted_contours[0]], -1, (0, 0, 255), 2)  # Green color, 2-pixel thickness

        # display the image with detected eyes
        cv2.imshow('Final Detection', original_image)
        cv2.waitKey(0)

    def score_skin_condition(self, red_mask, a_channel):
        """Score the face skin condition from 0-100 based on the detected contours and average redness."""
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        # Calculate the average redness of the top 1 percent redness
        top_redness_values = a_channel[red_mask > 0]
        average_redness = np.mean(top_redness_values)

        # Score calculation: more contours and higher average redness result in a lower score
        score = max(0, 100 - num_contours * 2 - int(average_redness / 500))  # Adjust the multipliers as needed
        return score

def select_image():
    """Open a file dialog to select an image."""
    Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
    image_path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not image_path:
        raise Exception("No image selected")
    return image_path

if __name__ == "__main__":
    try:
        image_path = select_image()
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to load image")

        face_masker = FaceMasker()
        image = face_masker.resize_image(image)
        original_image = image.copy()
        masked_face = face_masker.mask_face(image)
        masked_image = face_masker.mask_out_mouth_and_eyes(masked_face)

        acne_detector = AcneDetector()
        a_channel = acne_detector.visualize_a_channel(masked_image)
        red_mask = acne_detector.find_red_regions(a_channel)
        acne_detector.plot_contours(masked_image, red_mask)
        acne_detector.eye_detection(image)
        acne_detector.class_acne(image,red_mask)

        # Score the skin condition
        score = acne_detector.score_skin_condition(red_mask, a_channel)
        print(f"Skin Condition Score: {score}")
        output_path = "detection_results.png"
        cv2.imwrite(output_path, original_image)
        print(f"Saved detection results to {output_path}")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {str(e)}")