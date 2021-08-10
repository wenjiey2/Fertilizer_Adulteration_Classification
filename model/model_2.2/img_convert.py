from PIL import Image
import glob

for filename in glob.glob('fert_dataset\\*\\*\\*.JPEG') or  glob.glob('fert_dataset\\*\\*\\*.jpeg') or glob.glob('fert_dataset\\*\\*\\*.PNG') or glob.glob('fert_dataset\\*\\*\\*.png'):
  # print(filename[:-5])
  img1 = Image.open(filename)
  img1.save(filename[:-4]+'JPG')
