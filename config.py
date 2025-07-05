# Constants
DEVICE = "cpu"
STEP_SIZE = 1/255
EPS = 8/255
STEPS = 200

CLIP_TYPE = "e14plus"  # options: "e14", "e14plus"

TARGET_TEXT = "target_text" # a dog wearing sunglasses
CLEAN_TEXT = "description_of_the_original_screenshot" # a google search page

IMAGE_PATH = "path_to_original_screenshot"
REF_IMAGE_PATH = "path_to_reference_image"

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
