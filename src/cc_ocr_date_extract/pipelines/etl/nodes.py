from typing import Dict, Any
import pandas as pd
import os


def load_pdf_file_names(parameters: Dict[str, Any]) -> pd.DataFrame:
    """Load pdf file names from path.
        Args:
            parameters: parameters from parameters.yml.
        Returns:
            DataFrame with parsed file names.
    """
    import glob
    files = glob.glob(parameters['path_pdfs'])
    df = pd.DataFrame({'file_path': files})
    df['file_name'] = df['file_path'].apply(lambda x: os.path.split(x)[1])
    df['file_number'] = df['file_name'].apply(lambda x: int(os.path.splitext(x)[0]))

    return df


def perform_ocr_on_files(df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Perform OCR on files loaded in DataFrame.
        Args:
            df: Loaded files.
            parameters: parameters from parameters.yml.
        Returns:
            DataFrame with file along with recognized text.
    """
    from pdf2image import convert_from_path
    import pytesseract

    def do_ocr(file_path):
        doc = convert_from_path(file_path)
        path, file_name = os.path.split(file_path)
        file_base_name, _ = os.path.splitext(file_name)

        page_number = 0
        text = ''
        for page_data in doc:
            page_number += 1
            if page_number <= parameters['max_page_numbers']:
                page_text = pytesseract.image_to_string(page_data, config=parameters['ocr_config'])
                text += f"\n\n{page_text}"

        return page_number, text

    df[['pdf_number_pages', 'pdf_text']] = df.apply(lambda r: do_ocr(r['file_path']), axis=1, result_type='expand')

    return df


def create_master_table(labels: pd.DataFrame, pdfs: pd.DataFrame) -> pd.DataFrame:
    """Combines all data to create a master table.
        Args:
            labels: auswertung labels.
            pdfs: Preprocessed pdf files.
        Returns:
            Master table.
    """
    master_table = (pd.merge(labels, pdfs, left_on='Dokumentnummer', right_on='file_number')
                    .drop(['Dokumentnummer', 'Dokumentenbezeichnung'], axis=1))

    return master_table