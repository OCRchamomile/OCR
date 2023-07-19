# OCR Project

## Вашему вниманию представлен проект по распознаванию текста со справки о донации крови.

Компании DonorSearch нужна была помощь в автоматизации процесса перевода текста с изображения в xlsx или csv файл.

После анализа разных подходов и проблем, наша команда сформулировала начальные задачи:
  1. Нужно чтобы изображения были в хорошем качестве (как-то улучшить фото)
  2. Найти готовое решение по обнаружению таблиц
  3. Найти готовое решение по распознаванию русского текста с изображения

- Для решения 1 задачи были использованы стандартные инструменты cv2, но ощутимого результата это не дало.

- При решении 2 задачи были найдены преобученные нейросети, но они не могут нормально распознать кривую таблицу с фотографии, поэтому было принято решение сделать функцию самим.

Пример функции:

    def scaleimg(pic):
        img = cv.imread(pic, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv.THRESH_BINARY,31,36)
        morph_kernel = np.ones([1,1])
        inverted_image = cv.bitwise_not(th3)
        image = inverted_image
        erode_img = cv.erode(image, kernel= morph_kernel, iterations=5)
        #plt.figure(figsize=(10,15))
        #plt.imshow(erode_img,'gray')
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
    
                    h = cv.flip(out,1)
    
                    #img = Image.fromarray(h)
                    #img = img.filter(ImageFilter.SHARPEN)
                    if out.shape[0] < out.shape[1]:
                      h = cv.flip(out,1)
                      image = cv.rotate(h, cv.ROTATE_180)
                    else:
                      h = cv.flip(out,0)
                      image = cv.rotate(h, cv.ROTATE_90_CLOCKWISE)
    
                return image

Эта функция принимает путь к изображению (файлу) в качестве аргумента и выполняет следующие операции:
- Загружает изображение в оттенках серого с помощью функции cv.imread и флага cv.IMREAD_GRAYSCALE.
- Проверяет, что изображение было успешно прочитано с помощью утверждения assert.
- Применяет адаптивный пороговый фильтр Gaussian к изображению с помощью функции cv.adaptiveThreshold, чтобы получить двоичное изображение.
- Производит эрозию двоичного изображения с использованием структурного элемента morph_kernel с помощью функции cv.erode.
- Создает маску нулей того же размера, что и исходное изображение.
- Выполняет эрозию и дилатацию двоичного изображения, чтобы выделить вертикальные и горизонтальные линии.
- Объединяет вертикальные и горизонтальные линии с помощью функции cv.add.
- Применяет дилатацию к объединенному изображению с помощью структурного элемента kernel.
- Производит бинаризацию объединенного изображения с помощью функции cv.threshold.
- Ищет контуры на бинаризованном изображении с помощью функции cv.findContours.
- Фильтрует контуры по площади и минимальному значению, чтобы оставить только интересующие области.
- Вычисляет координаты углов прямоугольника, описывающего каждую интересующую область.
- Заполняет маску белым цветом внутри каждого прямоугольника.
- Выполняет перспективное преобразование каждой интересующей области с помощью функции cv.warpPerspective.
- Отражает полученное изображение горизонтально или вертикально в зависимости от его размера.
- Возвращает преобразованное изображение.

Пример работы функции:

Было

![image](https://github.com/OCRchamomile/OCR/assets/126798126/909b9dba-ecd4-46f0-bc04-6c14dde25bfe)

Стало

![image](https://github.com/OCRchamomile/OCR/assets/126798126/97780bd8-1b98-407d-8457-08dcca577024)



- Для решения 3 задачи были проверены предобученне сети, такие как tesseract, easyOCR, paddleOCR.
