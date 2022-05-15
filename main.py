from src.intelligent_placer_lib import check_image, get_image_files

for image in get_image_files("data/dataset/"):
    print(check_image(image, "output/reports/"))
