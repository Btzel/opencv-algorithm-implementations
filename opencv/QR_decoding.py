from pyzbar.pyzbar import decode
from PIL import Image
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
img = Image.open(os.path.join(script_dir, '../external/datasets/random/qrcode.png'))
result = decode(img)

for i in result:
    print(i.data.decode("utf-8"))
