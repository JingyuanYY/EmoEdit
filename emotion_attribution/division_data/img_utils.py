from PIL import Image, ImageOps


def crop_img(img, output_size=(512, 512)):
    width, height = img.size
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2
    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize(output_size)
    return img_resized


def load_img(image_path):
    image = Image.open(image_path)
    image = crop_img(image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def vae2pil(image):
    """

    :param image:
    :return:
    """
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images
