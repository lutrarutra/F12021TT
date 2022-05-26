import cv2
import numpy as np
import pandas as pd
import pytesseract as pts
import tqdm
import argparse

from PIL import Image

from matplotlib import pyplot as plt
import random
import string
import os

time_whitelist = ".:0123456789"
name_whitelist = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
date_whitelist = "/0123456789"
number_whitelist = "0123456789"
custom_whitelist = "YesNo"
full_whitelist = "/.:0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
TEAM_COLORS = {
    "Mercedes":(36.5, 193.8, 159.4),
    "Custom":(22.6, 21.1, 29.0),
    "Mc Laren":(200.3, 113.5,  26.3),
    "Ferrari":(167.3,  13.1,  32.5),
    "Red Bull":(28.2,  14.7, 191.7),
    "Alpine":(35.4, 127.8, 205.5),
    "Aston Martin":(28.6, 105.0,  91.3),
    "Haas":(178.0, 176.8, 182.1),
    "Williams":(29.2,  76.7, 202.8),
    "Alfa Romeo":(118.1,  15.0,  23.1),
    "AlphaTauri":(50.5, 64.4, 83.1),
}

def process_text(thresh, whitelist, psm, i):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = opening

    outname = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    if not os.path.exists("temp"):
        os.mkdir("temp")
    # plt.imsave(f"temp/{i}.png", invert, cmap="gray")

    return pts.image_to_string(invert,  lang='eng', config=f'--psm {psm} --oem 1 -c tessedit_char_whitelist={whitelist}').strip()


def crop(frame):
    number = frame[:, 0:256]
    name = frame[:, 500:1500]
    time = frame[:, 2100:2900]
    date = frame[:, 3000:3900]
    custom = frame[:, 5600:6000]
    return (number, name, time, date, custom)

def process_team_color(clr_avg):
    closest_key = "Mercedes"
    closest_value = np.abs(TEAM_COLORS[closest_key] - clr_avg).mean()
    for key in TEAM_COLORS.keys():
        value = np.abs(TEAM_COLORS[key] - clr_avg).mean()
        if value < closest_value:
            closest_key = key
            closest_value = value

    return closest_key

def process_colors(frame):
    team = frame[10:20,972:975,:]
    tc = frame[:,1617:1642,1]
    gear = frame[:,1650:1675,1]
    brakes = frame[:,1683:1708,1]

    # print(f"{tc[:,:,1].mean():.2f} {gear[:,:,1].mean():.2f} {brakes[:,:,1].mean():.2f}")
    team = process_team_color(team.mean(axis=(0,1)))

    tc_on = 1 if tc.mean() > 85 else 0
    gear_on = 1 if gear.mean() > 85 else 0
    brakes_on = 1 if brakes.mean() > 85 else 0
    return (team, tc_on, gear_on, brakes_on)


def process_video(filename, output):
    cap = cv2.VideoCapture(filename)
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_frame = None
    pbar = tqdm.tqdm(range(frame_count))
    diff = 0.0
    with open(output, "w") as f:
        for i in pbar:
            flag, frame = cap.read()
            if prev_frame is not None:
                diff = cv2.absdiff(frame, prev_frame).mean()
                if diff < 0.1:
                    continue
            if flag:
                # Frame is ready and already captured
                cropped_frame = frame[5:35,:,:]
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                team, tc_on, gear_on, brakes_on = process_colors(cropped_frame)
                # plt.imsave(f"temp/{team}_{i}_frame.png", cropped_frame)
                img = Image.fromarray(cropped_frame)
                BASE_HEIGHT = 128
                hpercent = (BASE_HEIGHT/float(img.size[1]))
                hsize = int((float(img.size[0])*float(hpercent)))
                img = img.resize((hsize,BASE_HEIGHT), Image.Resampling.LANCZOS)
                cropped_frame = np.array(img)

                gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, (13,13), 0)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                number_img, name_img, time_img, date_img, custom_img = crop(thresh)

                number_text = process_text(number_img, number_whitelist, 10, f"{i}_number")
                name_text = process_text(name_img, name_whitelist, 10, f"{i}_name")
                time_text = process_text(time_img, time_whitelist, 10, f"{i}_time")
                date_text = process_text(date_img, date_whitelist, 10, f"{i}_date")
                custom_text = process_text(custom_img, custom_whitelist, 10, f"{i}_custom")

                pbar.set_postfix({"number_text": number_text, "name_text":name_text, "time":time_text, "date_text":date_text, "custom_text":custom_text})
                f.write(f"{i},{number_text},{name_text},{time_text},{date_text},{custom_text},{tc_on},{gear_on},{brakes_on},{team}\n")
                f.flush()

            prev_frame = frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to video recording")
    parser.add_argument("output", type=str, help="Output path")
    process_video(parser.input, parser.output)
