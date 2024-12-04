def resize_image(img, nw):
    w, h = img.width, img.height
    ratio = nw / w
    nw = int(nw)
    nh = int(ratio * h)
    return img.resize((nw, nh))