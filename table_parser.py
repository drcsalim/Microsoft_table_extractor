import argparse
import os
import string
import torch
import asyncio
import pytesseract
import json
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import fitz  # PyMuPDF
import markdown2
# import TDTSR

from bs4 import BeautifulSoup
from typing import Any, List
from PIL import Image, ImageEnhance
from collections import Counter
from itertools import tee, count
from pytesseract import Output
from cv2 import dnn_superres
from transformers import DetrImageProcessor
from transformers import TableTransformerForObjectDetection

# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from transformers import DetrForObjectDetection
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/pytesseract'
# pytesseract.pytesseract.tesseract_cmd = r"/home/bp39374/anaconda3/envs/test/bin/pytesseract"

class TableExtractionPipeline():
    def __init__(self) -> None:
        self.colors = ["red", "blue", "green", "yellow", "orange", "violet"]
        self.sr_model_path = "./LapSRN_x8.pb"
        self.table_detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        self.table_structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

    def PIL_to_CV(self, pil_img):
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def CV_to_PIL(self, cv_img):
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

    def sharpen_image(self, pil_img):
        img = self.PIL_to_CV(pil_img)
        sharpen_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpen = cv2.filter2D(img, -1, sharpen_kernel)
        pil_img = self.CV_to_PIL(sharpen)
        return pil_img

    def uniquify(self, seq, suffs):
        """
        Make all the items unique by adding a suffix i.e. 1, 2, 3, ...
            `seq` is mutable sequence of strings.
            `suffs` is an optional alternative suffix iterable.
        """
        not_unique = [k for k,v in Counter(seq).items() if v>1]
        suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
        for idx, seqitem in enumerate(seq):
            try:
                suffix = str(next(suff_gens[seqitem]))
            except KeyError:
                continue
            else:
                seq[idx] += suffix
        return seq

    def binarize_and_blur_image(self, pil_img):
        image = self.PIL_to_CV(pil_img)
        thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]
        result = cv2.GaussianBlur(thresh, (5,5), 0)
        result = 255 - result
        result = self.CV_to_PIL(result)
        return result
    
    def greybg_postprocessing(self, pil_img):
        '''
        Removes gray background from tables
        '''
        img = self.PIL_to_CV(pil_img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 100), (255, 5, 255)) 
        nzmask = cv2.inRange(hsv, (0, 0, 5), (255, 255, 255)) 
        nzmask = cv2.erode(nzmask, np.ones((3,3))) 
        mask = mask & nzmask
        new_img = img.copy()
        new_img[np.where(mask)] = 255
        return self.CV_to_PIL(new_img)
    
    def super_resolution(self, pil_img):
        '''
        requires opencv-contrib-python installed without the opencv-python
        '''
        sr = dnn_superres.DnnSuperResImpl_create()
        image = self.PIL_to_CV(pil_img)
        model_path = str(self.sr_model_path)
        model_name = model_path.split('/')[1].split('_')[0].lower()
        model_scale = int(model_path.split('/')[1].split('_')[1].split('.')[0][1])

        sr.readModel(model_path)
        sr.setModel(model_name, model_scale)
        final_img = sr.upsample(image)
        final_img = self.CV_to_PIL(final_img)
        return final_img

    async def pytess(self, cell_pil_image):
        extracted_text = ' '.join(
            pytesseract.image_to_data(
                cell_pil_image, 
                output_type=Output.DICT, 
                config='-c tessedit_char_blacklist=œ˜â€œï¬â™Ã©œ¢!|”?«“¥ --psm 6 preserve_interword_spaces'
            )['text']
        ).strip()
        return extracted_text

    def add_padding(self, pil_img, top, right, bottom, left, color=(255,255,255)):
        '''
        image padding as part of TSR preprocessing to prevent missing table edges
        '''
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def plot_results_detection(self, savepath, model_label_mapper, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        '''
        crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates
        '''
        plt.imshow(pil_img)
        ax = plt.gca()
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax()
            xmin, ymin, xmax, ymax = xmin-delta_xmin, ymin-delta_ymin, xmax+delta_xmax, ymax+delta_ymax
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='red', linewidth=3))
            text = f'{model_label_mapper[cl.item()]}: {p[cl]:0.2f}'
            ax.text(xmin-20, ymin-50, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.savefig(savepath)

    def crop_tables(self, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        '''
        crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates
        '''
        cropped_img_list = []
        for pval, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            xmin, ymin, xmax, ymax = xmin-delta_xmin, ymin-delta_ymin, xmax+delta_xmax, ymax+delta_ymax
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cropped_img_list.append(cropped_img)
        return cropped_img_list

    def generate_structure(self, model_label_mapper, pil_img, prob, boxes, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        '''
        Co-ordinates are adjusted here by 3 'pixels'. To plot table pillow image and the TSR bounding boxes on the table
        '''
        plt.figure(figsize=(32,20))
        plt.imshow(pil_img)
        ax = plt.gca()
        rows = {}
        cols = {}
        idx = 0

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax
            cl = p.argmax()
            class_text = model_label_mapper[cl.item()]
            text = f'{class_text}: {p[cl]:0.2f}'
            if (class_text == 'table row')  or (class_text =='table projected row header') or (class_text == 'table column'):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,fill=False, color=self.colors[cl.item()], linewidth=2))
                ax.text(xmin-10, ymin-10, text, fontsize=5, bbox=dict(facecolor='yellow', alpha=0.5))
            if class_text == 'table row':
                rows['table row.'+str(idx)] = (xmin, ymin-expand_rowcol_bbox_top, xmax, ymax+expand_rowcol_bbox_bottom)
            if class_text == 'table column':
                cols['table column.'+str(idx)] = (xmin, ymin-expand_rowcol_bbox_top, xmax, ymax+expand_rowcol_bbox_bottom)
            idx += 1
        plt.axis('on')
        return rows, cols

    def sort_table_features(self, rows:dict, cols:dict):
        '''
        Sometimes the header and first row overlap, and we need the header bbox not to have first row's bbox inside the headers bbox
        '''
        rows_ = {
            table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(rows.items(), key=lambda tup: tup[1][1])
        }
        cols_ = {
            table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])
        }
        return rows_, cols_

    def individual_table_features(self, pil_img, rows:dict, cols:dict):
        for k, v in rows.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            rows[k] = xmin, ymin, xmax, ymax, cropped_img
        for k, v in cols.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cols[k] = xmin, ymin, xmax, ymax, cropped_img
        return rows, cols

    def object_to_cells(self, rows:dict, cols:dict, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left):
        '''
        Removes redundant bbox for rows&columns and divides each row into cells from columns
        '''
        cells_img = {}
        header_idx = 0
        row_idx = 0
        previous_xmax_col = 0
        new_cols = {}
        new_master_row = {}
        previous_ymin_row = 0
        new_cols = cols
        new_master_row = rows

        for k_row, v_row in new_master_row.items():
            _, _, _, _, row_img = v_row
            xmax, ymax = row_img.size
            xa, ya, xb, yb = 0, 0, 0, ymax
            row_img_list = []
            # plt.imshow(row_img)

            for idx, kv in enumerate(new_cols.items()):
                k_col, v_col = kv
                xmin_col, _, xmax_col, _, col_img = v_col
                xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
                # plt.imshow(col_img)

                # xa + 3 : to remove borders on the left side of the cropped cell
                # yb = 3: to remove row information from the above row of the cropped cell
                # xb - 3: to remove borders on the right side of the cropped cell

                xa = xmin_col
                xb = xmax_col
                if idx == 0:
                    xa = 0
                if idx == len(new_cols)-1:
                    xb = xmax
                xa, ya, xb, yb = xa, ya, xb, yb
                row_img_cropped = row_img.crop((xa, ya, xb, yb))
                row_img_list.append(row_img_cropped)

            cells_img[k_row+'.'+str(row_idx)] = row_img_list
            row_idx += 1
        return cells_img, len(new_cols), len(new_master_row)-1

    def clean_dataframe(self, df):
        '''
        Remove irrelevant symbols that appear with tesseractOCR
        '''
        for col in df.columns:
            df[col]=df[col].str.replace("'", '', regex=True)
            df[col]=df[col].str.replace('"', '', regex=True)
            # df[col]=df[col].str.replace('|', '', regex=True)
            # df[col]=df[col].str.replace(']', '', regex=True)
            # df[col]=df[col].str.replace('[', '', regex=True)
            # df[col]=df[col].str.replace('{', '', regex=True)
            # df[col]=df[col].str.replace('}', '', regex=True)
        return df

    def convert_dataframe_into_csv(self, df:pd.DataFrame):
        return df.to_csv(index=False)

    def convert_dataframe_into_markdown(self, df:pd.DataFrame):
        return df.to_markdown(index=False)

    def convert_dataframe_into_json(self, df:pd.DataFrame):
        return df.to_json(orient='records')
    
    def convert_ocr_extraction_into_dataframe(self, cells_pytess_result:list, max_cols:int, max_rows:int):
        '''
        Create dataframe using list of cell values of the table, also checks for valid header of dataframe
        Args:
            cells_pytess_result: list of strings, each element representing a cell in a table
            max_cols, max_rows: number of columns and rows
        Returns:
            dataframe : final dataframe after all pre-processing
        '''

        headers = cells_pytess_result[:max_cols]
        new_headers = self.uniquify(headers, (f' {x!s}' for x in string.ascii_lowercase))
        counter = 0

        cells_list = cells_pytess_result[max_cols:]
        df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)

        cell_idx = 0
        for nrows in range(max_rows):
            for ncols in range(max_cols):
                df.iat[nrows, ncols] = str(cells_list[cell_idx])
                cell_idx += 1

        ## To check if there are duplicate headers if result of uniquify+col == col
        ## This check removes headers when all headers are empty or if median of header word count is less than 6
        for x, col in zip(string.ascii_lowercase, new_headers):
            if f' {x!s}' == col:
                counter += 1
        header_char_count = [len(col) for col in new_headers]

        # if (counter == len(new_headers)) or (statistics.median(header_char_count) < 6):
        #     st.write('woooot')
        #     df.columns = uniquify(df.iloc[0], (f' {x!s}' for x in string.ascii_lowercase))
        #     df = df.iloc[1:,:]
        df = self.clean_dataframe(df)
        return df
    
    def table_detector(self, image, THRESHOLD_PROBA):
        '''
        Table detection using DEtect-object TRansformer pre-trained on 1 million tables
        '''

        feature_extractor = DetrImageProcessor(do_resize=True, size=800, max_size=800)
        encoding = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.table_detection_model(**encoding)

        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > THRESHOLD_PROBA

        target_sizes = image.size[::-1]
        target_sizes = torch.tensor(target_sizes).unsqueeze(0)
        postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
        bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
        return (probas[keep], bboxes_scaled)

    def table_structure_recognizer(self, image, THRESHOLD_PROBA):
        '''
        Table structure recognition using DEtect-object TRansformer pre-trained on 1 million tables
        '''

        feature_extractor = DetrImageProcessor(do_resize=True, size=1000, max_size=1000)        
        encoding = feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.table_structure_model(**encoding)

        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > THRESHOLD_PROBA

        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
        postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
        bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
        return (probas[keep], bboxes_scaled)

    def read_PDF_and_convert_pages_into_images(self, pdf_filepath) -> List[Any]:        
        pdf_document = fitz.open(pdf_filepath)
        pdf_page_images = []
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # Adjust resolution as needed
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pdf_page_images.append(image) 
            # image.save(f"{pdf_filepath.rstrip(".pdf")}_{page_number}.png")
        pdf_document.close()
        return pdf_page_images

    async def start_process(self, 
        pdf_path:str, 
        output_folder_path:str,
        TD_THRESHOLD, 
        TSR_THRESHOLD, 
        padd_top, 
        padd_left, 
        padd_bottom, 
        padd_right, 
        delta_xmin, 
        delta_ymin, 
        delta_xmax, 
        delta_ymax, 
        expand_rowcol_bbox_top, 
        expand_rowcol_bbox_bottom
    ):
        '''
        Initiates process of generating pandas dataframes from raw pdf-page images
        '''
        page_images = self.read_PDF_and_convert_pages_into_images(pdf_path)
        tables_book = []
        for page_num, page_image in enumerate(page_images):
            probas, bboxes_scaled = self.table_detector(page_image, THRESHOLD_PROBA=TD_THRESHOLD)
            model_label_mapper = self.table_detection_model.config.id2label

            if bboxes_scaled.nelement() == 0:
                print(f'No table found in the pdf-page - {page_num+1}')

            figpath = os.path.join(output_folder_path, os.path.basename(pdf_filepath.rstrip(".pdf")+f"_pg{page_num}_.png"))
            self.plot_results_detection(figpath, model_label_mapper, page_image, probas, bboxes_scaled,  delta_xmin, delta_ymin, delta_xmax, delta_ymax)
            list_of_cropped_imges = self.crop_tables(page_image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax)

            for table_num, unpadded_table_image in enumerate(list_of_cropped_imges):
                table = self.add_padding(unpadded_table_image, padd_top, padd_right, padd_bottom, padd_left)
                # table = self.super_resolution(table)
                # table = self.binarize_and_blur_image(table)
                # table = self.sharpen_image(table)
                # table = self.greybg_postprocessing(table)

                probas, bboxes_scaled = self.table_structure_recognizer(table, THRESHOLD_PROBA=TSR_THRESHOLD)
                model_label_mapper = self.table_structure_model.config.id2label
                rows, cols = self.generate_structure(
                    model_label_mapper, table, probas, bboxes_scaled, 
                    expand_rowcol_bbox_top, expand_rowcol_bbox_bottom
                )
                rows, cols = self.sort_table_features(rows, cols)
                rows, cols = self.individual_table_features(table, rows, cols)
                cells_img, max_cols, max_rows = self.object_to_cells(
                    rows, cols, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left
                )

                sequential_list_of_cell_images = []
                for _, cell_img_list in cells_img.items():
                    for cell_image in cell_img_list:
                        # cell_image = self.super_resolution(cell_image)
                        # cell_image = self.sharpen_image(cell_image) 
                        # cell_image = self.binarize_and_blur_image(cell_image)
                        cell_image = self.add_padding(cell_image, 10, 10, 10, 10)
                        sequential_list_of_cell_images.append(self.pytess(cell_image))
                cells_pytess_result = await asyncio.gather(*sequential_list_of_cell_images)

                dataframe = self.convert_ocr_extraction_into_dataframe(cells_pytess_result, max_cols, max_rows)
                
                markdown_dataframe = str(self.convert_dataframe_into_markdown(dataframe))
                json_dataframe = str(self.convert_dataframe_into_json(dataframe))
                tables_book.append({
                    "page_number": page_num,
                    "table_number": table_num,
                    "markdown_table": markdown_dataframe,
                    "json_table": json_dataframe
                })
        
        tables_book = pd.DataFrame(tables_book)
        tables_filename = os.path.basename(pdf_path.rstrip(".pdf") + ".json")
        tables_book.to_json(os.path.join(output_folder_path, tables_filename), orient='records', indent=1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--input_pdf_filepath", required=True, help="path to input PDF file")  
    parser.add_argument("--output_folder_path", required=True, help="path to output folder for saving parsed data")  
    args = parser.parse_args() 
    
    pdf_filepath = str(args.input_pdf_filepath)
    output_folder_path = str(args.output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    TD_th = 0.6
    TSR_th = 0.8
    padd_top = 2
    padd_left = 2
    padd_right = 2
    padd_bottom = 2

    tep = TableExtractionPipeline()
    asyncio.run(
        tep.start_process(
            pdf_filepath, 
            output_folder_path,
            TD_THRESHOLD=TD_th , 
            TSR_THRESHOLD=TSR_th , 
            padd_top=padd_top, 
            padd_left=padd_left, 
            padd_bottom=padd_bottom, 
            padd_right=padd_right, 
            delta_xmin=0, 
            delta_ymin=0, 
            delta_xmax=0, 
            delta_ymax=0, 
            expand_rowcol_bbox_top=0, 
            expand_rowcol_bbox_bottom=0
        )
    )