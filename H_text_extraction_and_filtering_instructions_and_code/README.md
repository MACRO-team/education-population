We have 2 scripts here:

1. *text_extraction_code.py*: This is for extracting the text from a pdf file. It has two named parameters. The first one is **pdf_path** which is a path to a folder that contains pdf files. The other one  is **output_path** that is a path to a folder which will contain the extracted txt files.
2. *header_extract.py*: This is for filtering the txt file to only have necassary sections of the article. It has two named parameters as well. The first one is **pdf_extract_dir** is a path to a fodler that contains extracted pdf files. The other one is **output_path** which is a path to a folder which will contain the filest with relevant sections of the articles.

Before running these scripts, it is needed to install pdfminer using ```pip install pdfminer``` command.
