
import argparse
import json
import os

def save_json(out_path, save_dict, indent=4, encoding='utf-8') :
    with open(out_path, 'w+') as f :
        json.dump(save_dict, f, indent=indent, ensure_ascii=False)


parser = argparse.ArgumentParser()

parser.add_argument('--pdf_extract_dir', help = 'path of the folder containing text for extracted pdfs')
parser.add_argument('--output_path', help = 'path of output folder')

args = parser.parse_args()


if not os.path.exists(args.output_path) :
    os.mkdir(args.output_path)

## Getting list of all pdfs
extracted_pdfs = os.listdir(args.pdf_extract_dir)

## Iterating over each extracted pdf text and getting Abstract and Methods section

j=0
for pdf_txt in extracted_pdfs :
    print (str(j) + '  /  ' + str(len(extracted_pdfs)))
    j += 1

    extracted_sections = {}

    lines = open(os.path.join(args.pdf_extract_dir,pdf_txt),'r').readlines()
    

    ## Abstract extraction
    abstract_lines = ''
    add_line = False
    end_abstract = False
    t=1
    for line in lines :
        line = line.strip('\n').strip()

        #To ignore lines where pdfminer reads random text
        if len(line) > 500 :
            continue

        #if add_line :
        #    pdb.set_trace()
        if '©' in line :
            line = line.split('©')[0].strip()

        if 'abstract' in line.lower()[:8] and (t==1):
            add_line = True
            t=0

        if 'keywords' in line.lower()[:8] or 'introduction' in line.lower()[:12] or 'table' in line.lower()[:8]:
            break

        if add_line :
            if line == '' and end_abstract :
                add_line = False
                break
            if line != '' :
                if line[-1] == '.' :
                    end_abstract = True
                abstract_lines += ' ' + line.strip()

    extracted_sections['abstract'] = abstract_lines
    extracted_sections['abstract_num_words'] = len(abstract_lines.split())


    ## Participants Extraction
    participants_lines = ''
    add_line = False
    end_participants = False
    t=1
    i=0
    for i in range(len(lines)) :
        line = lines[i]
        line = line.strip('\n').strip()

        if len(line) > 500 :
            continue
        #if add_line :
        #    pdb.set_trace()
        #if '©' in line :
        #    line = line.split('©')[0].strip()

        if ('participants' in line.lower()[:17]  or 'sample' in line.lower()[:10] or 'setting and participants' in line.lower()[:30]) and (t==1):
            ## to bypass cases where Participants is present in Tables of content
            if '.........' in line.replace(' ','') or '.........' in lines[i+1].replace(' ','') :
                continue
            add_line = True
            t=0

        if add_line :
            if 'procedure' in line.lower()[:12]  or 'fig' in line.lower()[:5] or 'independent variables' in line.lower()[:23] or 'research design' in line.lower()[:18]:
                break

            if line == '' and end_participants and (len(lines[i+1]) < 30):
                add_line = False
                break
            if line != '' :
                if line[-1] == '.' :
                    end_participants = True
                participants_lines += ' ' + line.strip()

    extracted_sections['participants'] = participants_lines
    extracted_sections['participants_num_words'] = len(participants_lines.split())

    save_json(os.path.join(args.output_path, pdf_txt + '.json'), extracted_sections)

