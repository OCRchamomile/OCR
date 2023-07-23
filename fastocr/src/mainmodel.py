import pandas as pd
import numpy as np
import cv2 as cv

import easyocr

def scaleimg(pic):
    img = cv.imread(pic, cv.IMREAD_GRAYSCALE)
    #assert img is not None, "file could not be read, check with os.path.exists()"
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,31,36)
    morph_kernel = np.ones([1,1])
    inverted_image = cv.bitwise_not(th3)
    image = inverted_image
    erode_img = cv.erode(image, kernel= morph_kernel, iterations=5)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    hor = np.array([[1,1,1,1,1,1]])
    vertical_lines_eroded_image = cv.erode(erode_img, hor, iterations=1)
    vertical_lines_eroded_image = cv.dilate(vertical_lines_eroded_image, hor, iterations=1)

    ver = np.array([[1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1]])
    horizontal_lines_eroded_image = cv.erode(erode_img, ver, iterations=1)
    horizontal_lines_eroded_image = cv.dilate(horizontal_lines_eroded_image, ver, iterations=1)

    combined_image = cv.add(vertical_lines_eroded_image, horizontal_lines_eroded_image)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    combined_image_dilated = cv.dilate(combined_image, kernel, iterations=1)

    ret, thresh = cv.threshold(combined_image_dilated, 140, 170, 0)
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cc = []
    for c in cnts:
        area = cv.contourArea(c)
        if area > 5000:
            rect = cv.minAreaRect(c)
            corners = cv.boxPoints(rect)
            corners = np.int0(corners)
            cv.fillPoly(mask, [corners], (255,255,255))

            corners = corners.tolist()
            cc.append(corners)
            for i in range(len(cc)):
                xx = []
                yy = []
                if cc[i][0][0] < (img.shape[1]*3/4):
                    for corner in cc[i]:
                        x, y = corner
                        xx.append(x)
                        yy.append(y)
                        cv.circle(image, (x, y), 5, 255, -1)
                else:
                    pass

                img = cv.imread(pic, cv.IMREAD_GRAYSCALE)

                pt_A = [xx[0], yy[0]]
                pt_B = [xx[1], yy[1]]
                pt_C = [xx[2], yy[2]]
                pt_D = [xx[3], yy[3]]

                width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
                width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
                maxWidth = max(int(width_AD), int(width_BC))


                height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
                height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
                maxHeight = max(int(height_AB), int(height_CD))

                input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
                output_pts = np.float32([[0, 0],
                                        [0, maxHeight - 1],
                                        [maxWidth - 1, maxHeight - 1],
                                        [maxWidth - 1, 0]])

                M = cv.getPerspectiveTransform(input_pts,output_pts)
                out = cv.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv.INTER_LINEAR)

                if out.shape[0] < out.shape[1]:
                    h = cv.flip(out,1)
                    image = cv.rotate(h, cv.ROTATE_180)
                else:
                    h = cv.flip(out,0)
                    image = cv.rotate(h, cv.ROTATE_90_CLOCKWISE)

        return image

def create_df_ocr(df_result):
    #reader = easyocr.Reader(['ru'], gpu=True)
    #result = reader.readtext(img)
    #df_result = pd.DataFrame(result, columns=['box', 'text', 'coef'])

    #df_new = pd.DataFrame(columns=['Класс крови', 'Дата донации', 'Дата добавления донации', 'Тип донации'])

    for i in range(len(df_result)):
        if df_result['text'][i].lower() in ['пл/д (бв)', 'плад (бв)', 'паф(бв)', 'п/ф(бв)', 'пф(бв)']:
            df_new.loc[len(df_new), ['Дата донации']] = df_result['text'][i-1]
            df_new.loc[len(df_new)-1, ['Класс крови']] = 'Плазма'
            df_new.loc[len(df_new)-1, ['Тип донации']] = 'Безвозмездно'
            df_new.loc[len(df_new)-1, ['Дата добавления донации']] = pd.to_datetime('today', dayfirst=True).normalize()
        elif df_result['text'][i].lower() in ['крад (бв)', 'крод (бв)', 'круд (бв)', 'кр/д (бв)', 'крд (бв)', 'кр/д(бв)',
                                            'с/д бв)" |', 'кред (бв)', 'к/д(бв)', 'хр/д (бв) ф', 'жр/д (бв)']:
            df_new.loc[len(df_new), ['Дата донации']] = df_result['text'][i-1]
            df_new.loc[len(df_new)-1, ['Класс крови']] = 'Цельная кровь'
            df_new.loc[len(df_new)-1, ['Тип донации']] = 'Безвозмездно'
            df_new.loc[len(df_new)-1, ['Дата добавления донации']] = pd.to_datetime('today', dayfirst=True).normalize()
        elif df_result['text'][i].lower() in ['цд (бв)', 'ц/д (бв)', 'таф(бв)', 'тф(бв)', 'т/ф(бв)']:
            df_new.loc[len(df_new), ['Дата донации']] = df_result['text'][i-1]
            df_new.loc[len(df_new)-1, ['Класс крови']] = 'Тромбоциты'
            df_new.loc[len(df_new)-1, ['Тип донации']] = 'Безвозмездно'
            df_new.loc[len(df_new)-1, ['Дата добавления донации']] = pd.to_datetime('today', dayfirst=True).normalize()
        else:
            pass

    try:
        df_new['Дата донации'] = pd.to_datetime(df_new['Дата донации'], format='%d.%m.%Y', dayfirst=True)
    except:
        pass
    df_new = df_new.sort_values(by='Дата донации').reset_index(drop=True)
    return df_new