# nacin na koji je cijeli projekat implementiran je opisan u glavnom README fajlu: https://github.com/miilank/segmentation
import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd

def program(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    # AI (Claude): pomoc pri odredjivanju HSV praga zute boje (prilagodjavanje klizaca za rad sa
    # slikom na sajtu https://medium.com/@sardorabdirayimov/lower-and-upper-mask-range-for-color-detection-opencv-7f64a5df568b
    lower_yellow = np.array([20, 190, 207])
    upper_yellow = np.array([35, 255, 255])
    mask = cv.inRange(img_hsv, lower_yellow, upper_yellow)
    mask[0:110, 0:110] = 0
    mask[200:270, 60:170] = 0
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask_open = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)
    mask_close = cv.morphologyEx(mask_open, cv.MORPH_CLOSE, kernel_close) 
    contours, _ = cv.findContours(mask_close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    noises = []
    bananas = []
    bunches = []
    coins = []
    # AI (Claude) je izmjerio area, aspect_ratio i solidity svih kontura na svih 11 slika i odredio vrijednosti ovih konstanti
    MIN_OBJECT_AREA = 14
    COIN_MIN_AREA = 500
    COIN_MIN_SOLIDITY = 0.9
    BANANA_MIN_RATIO = 1.9
    BANANA_MAX_AREA = 250
    BUNCH_MIN_AREA = 100
    BUNCH_MAX_AREA = 345
    sorted_contours = sorted(contours, key = cv.contourArea, reverse=True)
    for _, cnt in enumerate(sorted_contours):
        area = cv.contourArea(cnt)
        _, (w, h), _ = cv.minAreaRect(cnt)
        duza = max(w, h)
        kraca = min(w, h)
        if kraca > 0:
            aspect_ratio = duza / kraca
        else:
            aspect_ratio = 0
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        if hull_area > 0:
            solidity = float(area) / hull_area
        else:
            solidity = 0
        # AI(CLAUDE): pomogao oko redoslijeda provjera: sum, novcic, banana, grozd
        if area < MIN_OBJECT_AREA:
            noises.append(cnt)
        elif area >= COIN_MIN_AREA and solidity >= COIN_MIN_SOLIDITY:
            coins.append(cnt)
        elif aspect_ratio >= BANANA_MIN_RATIO and area <= BANANA_MAX_AREA:
            bananas.append(cnt)
        elif aspect_ratio < BANANA_MIN_RATIO and BUNCH_MIN_AREA <= area <= BUNCH_MAX_AREA:
            bunches.append(cnt)
        else:
            noises.append(cnt)
    value = len(bananas) * 1 + len(bunches) * 5 + len(coins) * 10
    return value

def main():
    folder = sys.argv[1]
    csv = pd.read_csv(os.path.join(folder, 'count.csv'))
    err = []
    for name, correct in zip(csv['image_name'], csv['bananas_value']):
        predicted = program(os.path.join(folder, 'images', name))
        err.append(abs(predicted - correct))
    mae = np.mean(err)
    print(mae)
if __name__ == '__main__':
    main()
