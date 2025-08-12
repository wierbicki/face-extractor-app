import base64
import io
from PIL import Image
import streamlit as st

def convert_image_to_bytes(pil_image, format='PNG'):
    """
    Convert a PIL image to bytes
    
    Args:
        pil_image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Image as bytes
    """
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format=format)
    img_buffer.seek(0)
    return img_buffer.getvalue()

def create_download_link(img_bytes, filename, text="Download"):
    """
    Create a download link for image bytes
    
    Args:
        img_bytes: Image as bytes
        filename: Name for the downloaded file
        text: Link text
        
    Returns:
        HTML download link
    """
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

def validate_image(uploaded_file):
    """
    Validate uploaded image file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple (is_valid, error_message)
    """
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size (limit to 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        return False, "File size too large. Please upload an image smaller than 10MB."
    
    # Check file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
    if uploaded_file.type not in allowed_types:
        return False, "Invalid file type. Please upload a JPG, JPEG, or PNG image."
    
    try:
        # Try to open the image
        image = Image.open(uploaded_file)
        image.verify()  # Verify it's a valid image
        uploaded_file.seek(0)  # Reset file pointer
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def resize_image_proportional(image, max_width=800, max_height=600):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: PIL Image object
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def get_image_info(image):
    """
    Get basic information about an image
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image information
    """
    return {
        'format': image.format,
        'mode': image.mode,
        'size': image.size,
        'width': image.size[0],
        'height': image.size[1]
    }

def create_thumbnail(image, size=(150, 150)):
    """
    Create a thumbnail of the image
    
    Args:
        image: PIL Image object
        size: Thumbnail size tuple (width, height)
        
    Returns:
        Thumbnail PIL Image
    """
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    return thumbnail

@st.cache_data
def load_demo_instructions():
    """
    Load demo instructions and tips
    
    Returns:
        Dictionary with instructions
    """
    return {
        'upload_tips': [
            "Choose clear, well-lit images for best results",
            "Front-facing photos work better than profile shots",
            "Ensure faces are not too small in the image",
            "Good contrast between face and background helps"
        ],
        'supported_formats': ['JPG', 'JPEG', 'PNG'],
        'max_file_size': '10MB',
        'output_format': '200x200 pixels PNG files'
    }
