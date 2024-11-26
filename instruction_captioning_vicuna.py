from transformers import pipeline
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
import sys, os, json
import torch

#img = f'data/detection/L/image_996.jpg'

'''
imgs = os.listdir(impath)

#captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
imcaps = {}
for img in imgs:
    imname = os.path.join(impath, img)
    txt = captioner(imname)
    imcaps[img] = txt[0][f'generated_text']
    print(txt)

with open('captions.json', 'w') as of:
    of.write(json.dumps(imcaps))
'''


#url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
#image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#prompt = "Is there a text present in this image?If yes is it hindi? Give me the hindi text in a one line caption. "
#prompt = "What hindi text is present in this Image? Give one line caption with hindi word."
#prompt = "What hindi text is present in this Image?"
#prompt = "Perform OCR on this image"
prompt = "Which Hindi text is present in the Image?"

#input_csv = f'data/BSTD/recognition/test_hindi.csv'
#base_impath = f'data/BSTD/detection'

input_csv = ''
base_impath = f'data/ILST/IIIT-ILST/Devanagari'

imgs = []
if 'BSTD' in base_impath:
    infix = f'bstd'
    with open(input_csv, 'r') as inp:
        images_paths = inp.readlines()

    for img_path in images_paths:
        tmp = img_path.split('_')[0].split('/')[-1]
        im_base = os.path.join(base_impath, tmp)

        imname = img_path.split('_')[1:3]
        imname = imname[0] + '_' + imname[1] + '.jpg'
        img_path = os.path.join(im_base, imname)
        if os.path.exists(img_path) is False:
            continue
        imgs.append(img_path)
    imgs = list(set(imgs))
elif 'ILST' in base_impath:
    infix = f'ilst'
    imglist = os.listdir(base_impath)
    imgs, xmls = [], []
    for imp in imglist:
        imp = os.path.join(base_impath, imp)
        if imp.endswith('.jpg'):
            imgs.append(imp)
        if imp.endswith('.xml'):
            xmls.append(imp)
        if imp.endswith('.py') or imp == 'cropped':
            continue

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

desc_caption = {}
for img in imgs:
    imname = img #os.path.join(impath, img)
    image = Image.open(imname).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, do_sample=False, num_beams=5,
                             max_length=256, min_length=1, top_k=0.9,
                             repetition_penalty=1.5, length_penalty=1.0,
                             temperature=1,)
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(f"{img}: {generated_text}")

    gen_text = captioner(imname)
    print(f'{img}: {gen_text}')

    desc_caption[img] = {
                            'imageid': img.split('/')[-1],
                            'caption': gen_text[0]['generated_text'],
                            'description': generated_text
                        }

with open(f'captions_description_{infix}.json', 'w') as of:
    of.write(json.dumps(desc_caption))
