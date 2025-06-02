from PIL import Image

# Load base and overlay images
base = Image.open('rgb_image.png').convert('RGBA').rotate(180, expand=True)
overlay = Image.open('timesurface.png').convert('RGBA').rotate(180, expand=True)

# Adjust overlay opacity (0.0 to 1.0)
opacity = 0.50
overlay = Image.blend(Image.new('RGBA', overlay.size), overlay, opacity)

# Composite the images
combined = Image.alpha_composite(base, overlay)

# Save or show result
combined.save('combined.png')
# combined.show()
