import cv2
import numpy as np
import pandas as pd
import pytesseract as pts
import tqdm
import argparse

from PIL import Image
from matplotlib import pyplot as plt

import os
import concurrent.futures

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

    return pts.image_to_string(opening,  lang='eng', config=f'--psm {psm} --oem 1 -c tessedit_char_whitelist={whitelist}').strip()


def crop(frame):
    #number = frame[:, 0:256]
    name = frame[:, 500:1500]
    time = frame[:, 2100:2900]
    date = frame[:, 3000:3900]
    custom = frame[:, 5600:6000]
    return (name, time, date, custom)

def process_team_color(clr_avg):
    closest_key = "Mercedes"
    closest_value = np.abs(TEAM_COLORS[closest_key] - clr_avg).mean()
    for key in TEAM_COLORS.keys():
        value = np.abs(TEAM_COLORS[key] - clr_avg).mean()
        if value < closest_value:
            closest_key = key
            closest_value = value

    return closest_key

def crop_colors(frame):
    team = frame[10:20,964:967,:]
    tc = frame[:,1607:1632,:]
    gear = frame[:,1640:1665,:]
    brakes = frame[:,1672:1699,:]

    return (team, tc, gear, brakes)

def process_colors(frame):
    team, tc, gear, brakes = crop_colors(frame)

    team = process_team_color(team.mean(axis=(0,1)))

    tc_on = 1 if tc[:,:,1].mean() > 85 else 0
    gear_on = 1 if gear[:,:,1].mean() > 85 else 0
    brakes_on = 1 if brakes[:,:,1].mean() > 85 else 0
    return (team, tc_on, gear_on, brakes_on)

def resize(frame, base_height=128):
    img = Image.fromarray(frame)
    hpercent = (base_height/float(img.size[1]))
    hsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((hsize, base_height), Image.Resampling.LANCZOS)

    return np.array(img)

def deconvolute(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (13,13), 0)
    return cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

def test_crop(filename):
    cap = cv2.VideoCapture(filename)
    prev_frame = None
    pbar = tqdm.tqdm(range(200))
    diff = 0.0

    if not os.path.exists("temp"):
        os.mkdir("temp")
    
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
            plt.imsave(f"temp/{i}_line.png", cropped_frame)

            team, tc, gear, brakes = crop_colors(cropped_frame)
            plt.imsave(f"temp/{i}_team.png", team)
            plt.imsave(f"temp/{i}_tc.png", tc)
            plt.imsave(f"temp/{i}_gear.png", gear)
            plt.imsave(f"temp/{i}_brakes.png", brakes)

            cropped_frame = resize(cropped_frame)
            cropped_frame = deconvolute(cropped_frame)
            name_img, time_img, date_img, custom_img = crop(cropped_frame)

            #plt.imsave(f"temp/{i}_number.png", number)
            plt.imsave(f"temp/{i}_name.png", name_img)
            plt.imsave(f"temp/{i}_time.png", time_img)
            plt.imsave(f"temp/{i}_date.png", date_img)
            plt.imsave(f"temp/{i}_custom.png", custom_img)

        prev_frame = frame

def process_video(filename, output, cores):
    cap = cv2.VideoCapture(filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_frame = None
    pbar = tqdm.tqdm(range(frame_count))
    diff = 0.0
    with open(output, "w") as f:
        f.write("frame,name,time,date,custom,tc,gear,brakes,team\n")
        for i in pbar:
            flag, frame = cap.read()

            # Check if frame changed from previous
            if prev_frame is not None:
                diff = cv2.absdiff(frame, prev_frame).mean()
                if diff < 0.1:
                    continue

            if flag:
                cropped_frame = frame[5:35,:,:]
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                team, tc_on, gear_on, brakes_on = process_colors(cropped_frame)
                cropped_frame = resize(cropped_frame)
                cropped_frame = deconvolute(cropped_frame)
                name_img, time_img, date_img, custom_img = crop(cropped_frame)

                imgs = [
                    (name_img, name_whitelist, 10, f"{i}_name"),
                    (time_img, time_whitelist, 10, f"{i}_time"),
                    (date_img, date_whitelist, 10, f"{i}_date"),
                    (custom_img, custom_whitelist, 10, f"{i}_custom")
                ]

                name_text = ""
                time_text = ""
                date_text = ""
                custom_text = ""

                # Parallel text recognition
                with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as exec:
                    future_to_img = {exec.submit(process_text, args[0], args[1], args[2], args[3]): args for args in imgs}
                    for future in concurrent.futures.as_completed(future_to_img):
                        img = future_to_img[future]
                        try:
                            text = future.result()
                            if "name" in img[3]:
                                name_text = text
                            elif "time" in img[3]:
                                time_text = text
                            elif "date" in img[3]:
                                date_text = text
                            elif "custom" in img[3]:
                                custom_text = text
                        except Exception as e:
                            print(e)

                pbar.set_postfix({"time":time_text, "date_text":date_text})
                f.write(f"{i},{name_text},{time_text},{date_text},{custom_text},{tc_on},{gear_on},{brakes_on},{team}\n")
                f.flush()

            prev_frame = frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to video recording")
    parser.add_argument("output", type=str, help="Output path")
    parser.add_argument("cores", type=int, help="Number of parallel cores")
    args = parser.parse_args()
    process_video(args.input, args.output, args.cores)
