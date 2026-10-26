BACKBONE_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
IMAGE_SIZE = 224
D_MODEL = 192
D_STATE = 64
CHUNK_SIZE = 64
NUM_CLASSES = 9

# Exact class ordering used by the released 9-class checkpoint.
CLASS_NAMES = [
    "1Gallstones",
    "2Abdomen and retroperitoneum",
    "3cholecystitis",
    "4Membranous and gangrenous cholecystitis",
    "5Perforation",
    "6Polyps and cholesterol crystals",
    "7Adenomyomatosis",
    "8Carcinoma",
    "9Various causes of gallbladder wall thickening",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
