import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import io
import zipfile
import os
from face_detector import FaceDetector
from utils import convert_image_to_bytes, create_download_link

# Initialize the face detector
@st.cache_resource
def load_face_detector():
    return FaceDetector()

def main():
    st.set_page_config(
        page_title="Face Extractor",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("🔍 Face Extractor")
    st.markdown("Upload an image and automatically extract detected faces as 200x200 pixel images.")
    
    # Initialize face detector
    detector = load_face_detector()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Image info
            st.info(f"Image size: {image.size[0]} x {image.size[1]} pixels")
        
        with col2:
            st.subheader("🎯 Processing")
            
            # Convert PIL image to OpenCV format
            image_array = np.array(image)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # RGB to BGR for OpenCV
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_cv = image_array
            
            # Process button
            if st.button("🚀 Extract Faces", type="primary"):
                with st.spinner("Detecting faces..."):
                    # Detect faces
                    faces = detector.detect_faces(image_cv)
                    
                    if len(faces) == 0:
                        st.warning("❌ No faces detected in the image.")
                        st.info("💡 Tips for better detection:\n"
                               "- Ensure faces are clearly visible\n"
                               "- Good lighting conditions\n"
                               "- Faces should be front-facing\n"
                               "- Try a different image")
                    else:
                        st.success(f"✅ Found {len(faces)} face(s)!")
                        
                        # Extract and resize faces
                        extracted_faces = []
                        face_images = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, (x, y, w, h) in enumerate(faces):
                            status_text.text(f"Processing face {i+1}/{len(faces)}...")
                            
                            # Expand the face region to include more context (hair, etc.)
                            # Add 40% padding on all sides
                            padding_factor = 0.4
                            padding_x = int(w * padding_factor)
                            padding_y = int(h * padding_factor)
                            
                            # Calculate expanded coordinates
                            x_start = max(0, x - padding_x)
                            y_start = max(0, y - padding_y)
                            x_end = min(image_cv.shape[1], x + w + padding_x)
                            y_end = min(image_cv.shape[0], y + h + padding_y)
                            
                            # Extract expanded face region
                            face_region = image_cv[y_start:y_end, x_start:x_end]
                            
                            # Convert back to RGB for PIL
                            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                            face_pil = Image.fromarray(face_rgb)
                            
                            # Resize to 200x200
                            face_resized = face_pil.resize((200, 200), Image.Resampling.LANCZOS)
                            
                            extracted_faces.append(face_resized)
                            
                            # Convert to bytes for download
                            img_bytes = convert_image_to_bytes(face_resized)
                            face_images.append(img_bytes)
                            
                            progress_bar.progress((i + 1) / len(faces))
                        
                        status_text.text("✅ Processing complete!")
                        
                        # Display extracted faces
                        st.subheader("👥 Extracted Faces")
                        
                        # Show faces in a grid
                        cols = st.columns(min(3, len(extracted_faces)))
                        for i, face in enumerate(extracted_faces):
                            with cols[i % 3]:
                                st.image(face, caption=f"Face {i+1}", width=150)
                                
                                # Individual download button
                                img_bytes = face_images[i]
                                st.download_button(
                                    label=f"📥 Download Face {i+1}",
                                    data=img_bytes,
                                    file_name=f"face_{i+1}.png",
                                    mime="image/png",
                                    key=f"download_{i}"
                                )
                        
                        # Bulk download option
                        if len(face_images) > 1:
                            st.subheader("📦 Bulk Download")
                            
                            # Create ZIP file
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for i, img_bytes in enumerate(face_images):
                                    zip_file.writestr(f"face_{i+1}.png", img_bytes)
                            
                            zip_buffer.seek(0)
                            
                            st.download_button(
                                label="📦 Download All Faces (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name="extracted_faces.zip",
                                mime="application/zip"
                            )
    
    # Instructions and information
    with st.expander("ℹ️ How to use", expanded=False):
        st.markdown("""
        ### Instructions:
        1. **Upload an image** using the file uploader above
        2. **Click "Extract Faces"** to detect and extract faces
        3. **Download individual faces** or all faces as a ZIP file
        
        ### Supported formats:
        - JPG, JPEG, PNG
        
        ### Features:
        - Automatic face detection using OpenCV
        - Faces resized to 200x200 pixels
        - Individual and bulk download options
        - Support for multiple faces in one image
        
        ### Tips for better results:
        - Use images with clear, well-lit faces
        - Front-facing photos work best
        - Ensure faces are not too small in the original image
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and OpenCV")

if __name__ == "__main__":
    main()
