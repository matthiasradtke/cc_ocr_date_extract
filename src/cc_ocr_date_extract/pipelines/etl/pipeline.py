from kedro.pipeline import Pipeline, node

from .nodes import load_pdf_file_names, perform_ocr_on_files, create_master_table


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=load_pdf_file_names,
                inputs='parameters',
                outputs='pdf_file_names',
                name='load pdf file names',
            ),
            node(
                func=perform_ocr_on_files,
                inputs=['pdf_file_names', 'parameters'],
                outputs='pdf_file_texts',
                name='perform ocr on files',
            ),
            node(
                func=create_master_table,
                inputs=['labels', 'pdf_file_texts'],
                outputs='data_raw',
                name='create master table',
            ),
        ]
    )
