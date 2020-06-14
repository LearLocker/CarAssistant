import numpy as np
import cv2
import pickle
from math import sqrt
import math
import imutils
from kivy.uix.screenmanager import ScreenManager, Screen
import os
import RoadSignDetect.RoadSignModel as mod
#############################################

frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# Get path to the current working directory
CWD_PATH = os.getcwd()
##############################################
class RoadSignDetection(Screen):

    def __init__(self, **kw):
        super(RoadSignDetection, self).__init__(**kw)

    def on_enter(self):  # Будет вызвана в   момент открытия экрана
        ######## START - MAIN FUNCTION #################################################

        # IMPORT THE TRANNIED MODEL
        if os.path.exists('model_trained.p'):  # True
            pickle_in = open("model_trained.p", "rb")  ## rb = READ BYTE
        else:
            mod.main()
            pickle_in = open("model_trained.p", "rb")  ## rb = READ BYTE
        model = pickle.load(pickle_in)

        vidcap = self.readVideo()

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        width = vidcap.get(3)  # float
        height = vidcap.get(4)  # float

        # initialize the termination criteria for cam shift, indicating
        # a maximum of ten iterations or movement by a least one pixel
        # along with the bounding box of the ROI
        termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        roiBox = None
        roiHist = None

        success = True
        similitary_contour_with_circle = 0.65  # parameter
        count = 0
        current_sign = None
        current_text = ""
        current_probability = 0
        current_size = 0
        sign_count = 0
        coordinates = []
        position = []

        while True:
            success, frame = vidcap.read()
            if not success:
                print("FINISHED")
                break
            width = frame.shape[1]
            height = frame.shape[0]
            # frame = cv2.resize(frame, (640,int(height/(width/640))))
            frame = cv2.resize(frame, (640, 480))

            # print("Frame:{}".format(count))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            coordinate, image, sign_type, text, probabilityValue = self.localization(frame, 300,
                                                                                0.65, model, count,
                                                                                current_sign)
            if coordinate is not None:
                cv2.rectangle(image, coordinate[0], coordinate[1], (255, 255, 255), 1)
            # print("Sign:{}".format(sign_type))
            if sign_type > 0 and (not current_sign or sign_type != current_sign) and probabilityValue > threshold:
                current_sign = sign_type
                current_text = text
                current_probability = probabilityValue
                top = int(coordinate[0][1] * 1.05)
                left = int(coordinate[0][0] * 1.05)
                bottom = int(coordinate[1][1] * 0.95)
                right = int(coordinate[1][0] * 0.95)

                position = [count, sign_type, coordinate[0][0], coordinate[0][1], coordinate[1][0],
                            coordinate[1][1]]
                cv2.rectangle(image, coordinate[0], coordinate[1], (0, 255, 0), 1)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(image, text, (coordinate[0][0], coordinate[0][1] - 35), font, 1, (0, 0, 255), 2, cv2.LINE_4)
                cv2.putText(image, str(round(probabilityValue * 100, 2)) + "%",
                            (coordinate[0][0], coordinate[0][1] - 15), font, 1, (0, 0, 255), 2, cv2.LINE_4)

                tl = [left, top]
                br = [right, bottom]
                # print(tl, br)
                current_size = math.sqrt(math.pow((tl[0] - br[0]), 2) + math.pow((tl[1] - br[1]), 2))
                # grab the ROI for the bounding box and convert it
                # to the HSV color space
                roi = frame[tl[1]:br[1], tl[0]:br[0]]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

                # compute a HSV histogram for the ROI and store the
                # bounding box
                roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
                roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
                roiBox = (tl[0], tl[1], br[0], br[1])

            elif current_sign:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

                # apply cam shift to the back projection, convert the
                # points to a bounding box, and then draw them
                (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
                pts = np.int0(cv2.boxPoints(r))
                s = pts.sum(axis=1)
                tl = pts[np.argmin(s)]
                br = pts[np.argmax(s)]
                size = math.sqrt(pow((tl[0] - br[0]), 2) + pow((tl[1] - br[1]), 2))
                # print(size)

                if current_size < 1 or size < 1 or size / current_size > 30 or math.fabs(
                        (tl[0] - br[0]) / (tl[1] - br[1])) > 2 or math.fabs((tl[0] - br[0]) / (tl[1] - br[1])) < 0.5:
                    current_sign = None
                    print("Stop tracking")
                else:
                    current_size = size

                if sign_type > 0:
                    top = int(coordinate[0][1])
                    left = int(coordinate[0][0])
                    bottom = int(coordinate[1][1])
                    right = int(coordinate[1][0])

                    position = [count, sign_type if sign_type <= 8 else 8, left, top, right, bottom]
                    cv2.rectangle(image, coordinate[0], coordinate[1], (0, 255, 0), 1)
                    font = cv2.FONT_HERSHEY_PLAIN
                    cv2.putText(image, text, (coordinate[0][0], coordinate[0][1] - 35), font, 1, (0, 0, 255), 2,
                                cv2.LINE_4)
                    cv2.putText(image, str(round(probabilityValue * 100, 2)) + "%",
                                (coordinate[0][0], coordinate[0][1] - 15), font, 1, (0, 0, 255), 2, cv2.LINE_4)
                elif current_sign:
                    position = [count, sign_type if sign_type <= 8 else 8, tl[0], tl[1], br[0], br[1]]
                    cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (0, 255, 0), 1)
                    font = cv2.FONT_HERSHEY_PLAIN
                    cv2.putText(image, current_text, (tl[0], tl[1] - 35), font, 1, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.putText(image, str(round(current_probability * 100, 2)) + "%", (tl[0], tl[1] - 15), font, 1,
                                (0, 0, 255), 2, cv2.LINE_4)

            if current_sign:
                sign_count += 1
                coordinates.append(position)

            cv2.imshow('Result', image)
            count = count + 1
            # Write to video

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return

    def readVideo(self):
        # Read input video from current working directory
        inpImage = cv2.VideoCapture(os.path.join(CWD_PATH, 'RoadSignDetect\RoadSigns.mp4'))
        return inpImage

    def getCalssName(self, classNo):
        if classNo == 0: return 'Speed Limit 20 km/h'
        elif classNo == 1: return 'Speed Limit 30 km/h'
        elif classNo == 2: return 'Speed Limit 50 km/h'
        elif classNo == 3: return 'Speed Limit 60 km/h'
        elif classNo == 4: return 'Speed Limit 70 km/h'
        elif classNo == 5: return 'Speed Limit 80 km/h'
        elif classNo == 6: return 'End of Speed Limit 80 km/h'
        elif classNo == 7: return 'Speed Limit 100 km/h'
        elif classNo == 8: return 'Speed Limit 120 km/h'
        elif classNo == 9: return 'No passing'
        elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
        elif classNo == 11: return 'Right-of-way at the next intersection'
        elif classNo == 12: return 'Priority road'
        elif classNo == 13: return 'Yield'
        elif classNo == 14: return 'Stop'
        elif classNo == 15: return 'No vechiles'
        elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
        elif classNo == 17: return 'No entry'
        elif classNo == 18: return 'General caution'
        elif classNo == 19: return 'Dangerous curve to the left'
        elif classNo == 20: return 'Dangerous curve to the right'
        elif classNo == 21: return 'Double curve'
        elif classNo == 22: return 'Bumpy road'
        elif classNo == 23: return 'Slippery road'
        elif classNo == 24: return 'Road narrows on the right'
        elif classNo == 25: return 'Road work'
        elif classNo == 26: return 'Traffic signals'
        elif classNo == 27: return 'Pedestrians'
        elif classNo == 28: return 'Children crossing'
        elif classNo == 29: return 'Bicycles crossing'
        elif classNo == 30: return 'Beware of ice/snow'
        elif classNo == 31: return 'Wild animals crossing'
        elif classNo == 32: return 'End of all speed and passing limits'
        elif classNo == 33: return 'Turn right ahead'
        elif classNo == 34: return 'Turn left ahead'
        elif classNo == 35: return 'Ahead only'
        elif classNo == 36: return 'Go straight or right'
        elif classNo == 37: return 'Go straight or left'
        elif classNo == 38: return 'Keep right'
        elif classNo == 39: return 'Keep left'
        elif classNo == 40: return 'Roundabout mandatory'
        elif classNo == 41: return 'End of no passing'
        elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'

    ### Preprocess image
    def constrastLimit(self, image):
        img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(img_hist_equalized)
        channels[0] = cv2.equalizeHist(channels[0])
        img_hist_equalized = cv2.merge(channels)
        img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
        return img_hist_equalized

    def LaplacianOfGaussian(self, image):
        LoG_image = cv2.GaussianBlur(image, (3, 3), 0)  # paramter
        gray = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2GRAY)
        LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)  # parameter
        LoG_image = cv2.convertScaleAbs(LoG_image)
        return LoG_image

    def binarization(self, image):
        thresh = cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]
        # thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        return thresh

    def preprocess_image(self, image):
        image = self.constrastLimit(image)
        image = self.LaplacianOfGaussian(image)
        image = self.binarization(image)
        return image

    # Find Signs
    def removeSmallComponents(self, image, threshold):
        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        sizes = stats[1:, -1];
        nb_components = nb_components - 1

        img2 = np.zeros((output.shape), dtype=np.uint8)
        # for every component in the image, you keep it only if it's above threshold
        for i in range(0, nb_components):
            if sizes[i] >= threshold:
                img2[output == i + 1] = 255
        return img2

    def findContour(self, image):
        # find contours in the thresholded image
        cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        return cnts

    def contourIsSign(self, perimeter, centroid, threshold):
        #  perimeter, centroid, threshold
        # # Compute signature of contour
        result = []
        for p in perimeter:
            p = p[0]
            distance = sqrt((p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
            result.append(distance)
        max_value = max(result)
        signature = [float(dist) / max_value for dist in result]
        # Check signature of contour.
        temp = sum((1 - s) for s in signature)
        temp = temp / len(signature)
        if temp < threshold:  # is  the sign
            return True, max_value + 2
        else:  # is not the sign
            return False, max_value + 2

    # crop sign
    def cropContour(self, image, center, max_distance):
        width = image.shape[1]
        height = image.shape[0]
        top = max([int(center[0] - max_distance), 0])
        bottom = min([int(center[0] + max_distance + 1), height - 1])
        left = max([int(center[1] - max_distance), 0])
        right = min([int(center[1] + max_distance + 1), width - 1])
        print(left, right, top, bottom)
        return image[left:right, top:bottom]

    def cropSign(self, image, coordinate):
        width = image.shape[1]
        height = image.shape[0]
        top = max([int(coordinate[0][1]), 0])
        bottom = min([int(coordinate[1][1]), height - 1])
        left = max([int(coordinate[0][0]), 0])
        right = min([int(coordinate[1][0]), width - 1])
        # print(top,left,bottom,right)
        return image[top:bottom, left:right]

    def findLargestSign(self, image, contours, threshold, distance_theshold):
        max_distance = 0
        coordinate = None
        sign = None
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            is_sign, distance = self.contourIsSign(c, [cX, cY], 1 - threshold)
            if is_sign and distance > max_distance and distance > distance_theshold:
                max_distance = distance
                coordinate = np.reshape(c, [-1, 2])
                left, top = np.amin(coordinate, axis=0)
                right, bottom = np.amax(coordinate, axis=0)
                coordinate = [(left - 2, top - 2), (right + 3, bottom + 1)]
                sign = self.cropSign(image, coordinate)
        return sign, coordinate

    def findSigns(self, image, contours, threshold, distance_theshold):
        signs = []
        coordinates = []
        for c in contours:
            # compute the center of the contour
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            is_sign, max_distance = self.contourIsSign(c, [cX, cY], 1 - threshold)
            if is_sign and max_distance > distance_theshold:
                sign = self.cropContour(image, [cX, cY], max_distance)
                signs.append(sign)
                coordinate = np.reshape(c, [-1, 2])
                top, left = np.amin(coordinate, axis=0)
                right, bottom = np.amax(coordinate, axis=0)
                coordinates.append([(top - 2, left - 2), (right + 1, bottom + 1)])
        return signs, coordinates

    def localization(self, image, min_size_components, similitary_contour_with_circle, model, count, current_sign_type):
        original_image = image.copy()
        # cv2.imshow('Original', original_image)
        binary_image = self.preprocess_image(image)
        # cv2.imshow('Binary', binary_image)
        binary_image = self.removeSmallComponents(binary_image, min_size_components)

        binary_image = cv2.bitwise_and(binary_image, binary_image, mask=self.remove_other_color(image))

        # binary_image = remove_line(binary_image)

        # cv2.imshow('BINARY IMAGE', binary_image)
        contours = self.findContour(binary_image)
        # signs, coordinates = findSigns(image, contours, similitary_contour_with_circle, 15)
        sign, coordinate = self.findLargestSign(original_image, contours, similitary_contour_with_circle, 15)

        text = ""
        sign_type = -1
        i = 0
        probabilityValue = 0

        if sign is not None:
            # sign_type = getLabel(model, sign)
            img = np.asarray(sign)
            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = img / 255

            cv2.imshow('SIGN', img)
            img = img.reshape(1, 32, 32, 1)

            predictions = model.predict(img)
            sign_type = model.predict_classes(img)
            probabilityValue = np.amax(predictions)
            # sign_type = sign_type if sign_type <= 8 else 8
            # print(current_sign_type)
            text = str(self.getCalssName(sign_type))
            # cv2.imwrite(str(count) + '_' + text + '.png', sign)

        if sign_type > 0 and sign_type != current_sign_type and probabilityValue > threshold:
            cv2.rectangle(original_image, coordinate[0], coordinate[1], (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_PLAIN
            # cv2.putText(original_image, text, (coordinate[0][0], coordinate[0][1] - 35), font, 1, (0, 0, 255), 2, cv2.LINE_4)
            # cv2.putText(original_image, str(round(probabilityValue * 100, 2)) + "%", (coordinate[0][0], coordinate[0][1] - 15), font, 1, (0, 0, 255), 2, cv2.LINE_4)

        return coordinate, original_image, sign_type, text, probabilityValue

    def remove_line(self, img):
        gray = img.copy()
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        minLineLength = 5
        maxLineGap = 3
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength, maxLineGap)
        mask = np.ones(img.shape[:2], dtype="uint8") * 255
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 0), 2)
        return cv2.bitwise_and(img, img, mask=mask)

    def remove_other_color(self, img):
        frame = cv2.GaussianBlur(img, (3, 3), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_blue = np.array([100, 128, 0])
        upper_blue = np.array([215, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        lower_white = np.array([0, 0, 128], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        # Threshold the HSV image to get only blue colors
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([170, 150, 50], dtype=np.uint8)

        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        mask_1 = cv2.bitwise_or(mask_blue, mask_white)
        mask = cv2.bitwise_or(mask_1, mask_black)
        # Bitwise-AND mask and original image
        # res = cv2.bitwise_and(frame,frame, mask= mask)
        return mask