import glob
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import re
import os
import itertools
import pickle
import PIL
import numpy as np
import pandas as pd
from PIL import Image
import cv2  # Aggiunto import per opencv


def get_image_color(image, x, y):
    return image[y, x, :].tolist()


def get_sorted_pixel_histogram(pixel_c):
    if pixel_c:
        count_element = Counter(map(tuple, pixel_c))
        df = pd.DataFrame.from_dict(count_element, orient='index').reset_index()
        df.columns = ['R-G-B', 'N_occurrence']
        sorted_df = df.sort_values(by=['N_occurrence'], ascending=False)
        color_frequence = sorted_df['R-G-B'].values.tolist()
        return color_frequence


def t_level(level):
    if level / 255 <= 0.03928:
        return level / 12.92
    else:
        value = (level + 0.055) / (1.055)
        return pow(value, 2.4)


def get_luminance_by_rgb(color):
    return 0.216 * t_level(color[0]) + 0.7152 * t_level(color[1]) + 0.0722 * t_level(color[2])


def luminance_contrast(color, medoids):
    L_a = get_luminance_by_rgb(color)
    L_b = get_luminance_by_rgb(medoids)
    return (L_a + 0.05) / (L_b + 0.05) if L_a > L_b else (L_b + 0.05) / (L_a + 0.05)


def get_medoids(hist, k, r):
    return hist[:k]


def euclidean_distance(color1, color2):
    a = np.array(color1)
    b = np.array(color2)
    dist = np.sqrt(np.sum(np.square(a - b)))
    return dist


def get_closest(color, medoids):
    if not medoids:
        return None
    closest = medoids[0]
    min_distance = euclidean_distance(color, closest)
    for med in medoids:
        dist = euclidean_distance(color, med)
        if dist < min_distance:
            min_distance = dist
            closest = med
    return closest


def check_pixel(pixel_x, pixel_y, visited_pixels):
    pixel = (pixel_x, pixel_y)
    if pixel not in visited_pixels:
        visited_pixels.add(pixel)
        return True
    return False


def getPixels(image, component, tipo, visited_pixels):
    values = re.findall(r'\d+', component[16][1])
    comp = component[3][1]

    if tipo == 1:
        values = re.findall(r'\d+', component[17][1])
        comp = component[4][1]

    start_x, start_y, end_x, end_y = map(int, values)

    result, list_colors, list_x, list_y, list_components = [], [], [], [], []

    for x in range(start_x, end_x):
        list_x.append(x)

    for y in range(start_y, end_y):
        list_y.append(y)

    for element in itertools.product(list_x, list_y):
        if check_pixel(element[0], element[1], visited_pixels):
            result.append(element)
            list_colors.append(get_image_color(image, element[0], element[1]))
            list_components.append(comp)

    if result:
        pixel_df = pd.DataFrame(data=result, columns=['Axis-X', 'Axis-Y'])
        pixel_df['R-G-B'] = list_colors
        pixel_df["Components"] = list_components

        return pixel_df

    return pd.DataFrame()


def get_guis(path):
    xml_files = [filename for filename in os.listdir(path) if filename.endswith("xml")]
    guis = []

    for xml_file in xml_files:
        base_name = os.path.splitext(xml_file)[0]
        png_file = base_name + ".png"
        if png_file in os.listdir(path):
            guis.append(base_name)

    return guis


def bocp_bocc_algorithm(k, r, xml_file, png_file):
    bocp, bocc = defaultdict(list), defaultdict(set)

    print(f"Processing GUI: {xml_file}")

    image = cv2.imread(png_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tree = ET.parse(os.path.join(xml_file))
    root = tree.getroot()
    components = list(root.iter())

    for component in reversed(components):
        c_item = component.items()
        pixel_df = pd.DataFrame()

        if len(c_item) == 17 and "ImageView" not in c_item[3][1] and "ImageButton" not in c_item[3][1]:
            pixel_df = getPixels(image, c_item, 0, set())
        elif len(c_item) == 18 and "ImageView" not in c_item[4][1] and "ImageButton" not in c_item[4][1]:
            pixel_df = getPixels(image, c_item, 1, set())

        if not pixel_df.empty:
            list_colors = list(pixel_df["R-G-B"])
            hist_df = get_sorted_pixel_histogram(list_colors)
            medoids = get_medoids(hist_df, k, r)
            list_x = list(pixel_df["Axis-X"])
            list_y = list(pixel_df["Axis-Y"])

            for i, pixel_color in enumerate(list_colors):
                color_quant = get_closest(pixel_color, medoids)
                pixel_x_y = (list_x[i], list_y[i])
                c_n = component

                bocp[color_quant].append(pixel_x_y)
                bocc[color_quant].add(c_n)

    return bocp, bocc


def main():
    path = "screenshotsAndSnapshots"
    k, r = 3, 1.6

    # Crea la cartella "Data" se non esiste
    data_folder = "Data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    guis = get_guis(path)

    for gui in guis:
        xml_file = os.path.join(path, gui + ".xml")
        png_file = os.path.join(path, gui + ".png")
        bocp, bocc = bocp_bocc_algorithm(k, r, xml_file, png_file)

        with open(f"{data_folder}/{gui}_BOCP.pickle", "wb") as handle:
            pickle.dump(bocp, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{data_folder}/{gui}_BOCC.pickle", "wb") as handle:
            pickle.dump(bocc, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
