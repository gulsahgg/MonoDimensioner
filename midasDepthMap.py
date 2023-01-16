import cv2
import imutils
import torch
import time
import numpy as np
from scipy.spatial import distance as dist
import random as rng
from mask_rcnn import *

class BoxSize:
    def __init__(self):
        # Load a MiDas model for depth estimation
        # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        self.model_type = "DPT_Hybrid"  # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #self.model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        # Move model to GPU if available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        # Load transforms to resize and normalize the image
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def find_dim(self, img, box, ppm, show=False):
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = self.midpoint(tl, tr)
        (blbrX, blbrY) = self.midpoint(bl, br)

        (tlblX, tlblY) = self.midpoint(tl, bl)
        (trbrX, trbrY) = self.midpoint(tr, br)
        if show:
            # draw the midpoints on the image
            cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            # draw lines between the midpoints
            cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 1)
            cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 1)
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        dimA = dA / ppm
        dimB = dB / ppm
        return dimA,dimB, self.midpoint(tl, tr), self.midpoint(tr, br)

    def safe_div(self, x, y):  # so we don't crash so often
        if y == 0: return 0
        return x / y

    def midpoint(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    def midOfBox(self, img, box, show=True):
        M = cv2.moments(box)
        cX = int(self.safe_div(M["m10"], M["m00"]))
        cY = int(self.safe_div(M["m01"], M["m00"]))
        if (show):  # draw the contour and center of the shape on the image
            cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
            cv2.putText(img, "center", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
        return cX, cY

    def depth_to_distance(self, depth):
        return -1.8 * depth + 2 #P = D * scale + shift

    def findDepthMap(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Apply input transforms
        input_batch = self.transform(img).to(self.device)
        # Prediction and resize to original resolution
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        depth_map = (depth_map * 255).astype(np.uint8)
        return depth_map

    def findBox(self, img):
        edged = cv2.Canny(img, 0, 100)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        minRect = [None] * len(cnts)
        areaArray = []
        for i, c in enumerate(cnts):
            minRect[i] = cv2.minAreaRect(c)
            ar = cv2.contourArea(c)
            areaArray.append(ar)
        area=max(areaArray)
        area_index=areaArray.index(area)
        box = cv2.boxPoints(minRect[area_index])
        box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        return box, area

    def map(self, depth_map, point):
        return depth_map[point[1], point[0]]

    # def show_distance(self, event, x, y, args, params):
    #     depth = map(, (x,y))
    #     print(self.depth_to_distance(depth))


def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    box_size = BoxSize()
    while cap.isOpened():
        success, img = cap.read()
        #--------------------------
        depth_map=box_size.findDepthMap(img)
        box, area = box_size.findBox(depth_map)
        #--------------------------
        for (x, y) in box:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        # compute the size of the object
        pixelsPerMetric = 3.4
        dimA, dimB, placeA, placeB = box_size.find_dim(img, box, pixelsPerMetric)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        # draw the object sizes on the image
        cv2.putText(img, "{:.1f}mm".format(dimA), (int(placeA[0]), int(placeA[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)
        cv2.putText(img, "{:.1f}mm".format(dimB), (int(placeB[0])+10, int(placeB[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)
        # compute the center of the contour
        cX, cY = box_size.midOfBox(img, box)
        depth_mid = depth_map[cY, cX]
        depth_mid = box_size.depth_to_distance(depth_mid)
        # display depth on image
        cv2.putText(img, "Depth in cm: " + str(round(depth_mid/10, 2)), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3)
        #draw flat area
        cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
        #dispaly FPS
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Image', img)
        cv2.imshow('Depth', depth_map)
        counter=0
        counter= counter+1
        print(counter)
        if cv2.waitKey(2) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

print(__name__)
if __name__ == '__main__':
    main()
