import argparse
import os
from pdfminer.high_level import extract_text

parser = argparse.ArgumentParser()

parser.add_argument('--pdf_path', help='path of the folder containing pdfs')
parser.add_argument('--output_path', help='path of output folder')

args = parser.parse_args()

i=0

if not os.path.exists(args.output_path) :
    os.mkdir(args.output_path)

## Getting list of all pdfs
pdfs = os.listdir(args.pdf_path)

## Iterating over each pdf and saving extracted text in a txt file
for pdf in pdfs :
    print ( str(i) + ' /  ' + str(len(pdfs)))
    i += 1
    filename = os.path.join(args.pdf_path, pdf)
    file_name = filename.split('/')[-1]
    try : 
        text = extract_text(filename)

        temp_txt = open(os.path.join(args.output_path, file_name + '.txt'), "w+")
        n = temp_txt.write(text)
        temp_txt.close()
    except :
        print('pdf extraction failed for ' + str(pdf))
        print('Possible issue with path name or input file is corrupt')
        continue
