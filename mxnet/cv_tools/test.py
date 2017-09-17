

from PIL import Image
import image_tool
from PIL import ImageFilter


img = Image.open('/home/workspace/DATASET/JPEGImages/911/92_0')
img = img.convert('L')
img.show()
enhance = image_tool.Enhance(img=img)
enhance.brightness_enhance(factor=0.3, is_show=False)
enhance.set_img(img)
img = enhance.color_enhance(factor=0.8, is_show=True)
enhance.set_img(img)
img = enhance.sharpness_enhance(factor=2.4, is_show=True)
img.filter(ImageFilter.SMOOTH).show(title='smooth')
