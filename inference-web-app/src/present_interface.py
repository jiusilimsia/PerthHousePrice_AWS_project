import base64
import logging
from typing import List
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from PIL import Image

logger = logging.getLogger(__name__)

def get_image_b64(image_path: str) -> str:
    """
    Converts an image file to base64 format.
    
    Parameters:
        image_path (str): Path to the image file.
        
    Returns:
        str: Image in base64 format.
    """
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logging.error('An error occurred while getting image b64: %s', e)
        raise

def get_video_b64(video_path: str) -> str:
    """
    Converts a video file to base64 format.
    
    Parameters:
        video_path (str): Path to the video file.
        
    Returns:
        str: Video in base64 format.
    """
    try:
        with open(video_path, 'rb') as video_file:
            return base64.b64encode(video_file.read()).decode('utf-8')
    except Exception as e:
        logging.error('An error occurred while getting video b64: %s', e)
        raise

def load_options(file_path: str) -> List[str]:
    """
    Loads options from a text file.
    
    Parameters:
        file_path (str): Path to the text file.
        
    Returns:
        list: List of options.
    """
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f]
    except Exception as e:
        logging.error('An error occurred while loading options from %s: %s', file_path, e)
        raise

def present_interface(model,preprocessor):
    """
    Renders the user interface.
    
    Parameters:
        user_info (Dict[str, str]): User's information for prediction.
        price (float): Predicted price.
    """
    user_info = {
        'SUBURB': None,
        'BEDROOMS': None,
        'BATHROOMS': None,
        'GARAGE': None,
        'LAND_AREA': None,
        'FLOOR_AREA': None,
        'NEAREST_STN': None,
        'LATITUDE': None,
        'NEAREST_SCH': None,
        'OTHERS_ROOMS_AREA': None,
        'GARAGE_AREA': None,
        'BATHROOMS_AREA': None,
        'BEDROOMS_AREA': None,
    }

    # Define your sliders with keys
    sliders = {
        'BEDROOMS': st.sidebar.slider('BEDROOMS', 0, 8, 3, step=1),
        'BATHROOMS': st.sidebar.slider('BATHROOMS', 0, 5, 2, step=1),
        'GARAGE': st.sidebar.slider('GARAGE', 0, 3, 1, step=1),
        'LAND_AREA': st.sidebar.slider('LAND_AREA', 0, 2000, 1000, step=50),
        'FLOOR_AREA': st.sidebar.slider('FLOOR_AREA', 0, 1000, 700, step=50),
        # 'OTHERS_ROOMS_AREA': st.sidebar.slider('OTHERS_ROOMS_AREA', 0, 2000, 1000, step=50),
        'GARAGE_AREA': st.sidebar.slider('GARAGE_AREA', 0, 250, 150, step=50),
        'BATHROOMS_AREA': st.sidebar.slider('BATHROOMS_AREA', 0, 250, 150, step=50),
        'BEDROOMS_AREA': st.sidebar.slider('BEDROOMS_AREA', 0, 200, 150, step=50),
    }

    suburb_options = load_options('config/suburb.txt')
    nearest_stn_options = load_options('config/nearest_stn.txt')
    nearest_sch_options = load_options('config/nearest_sch.txt')

    suburbs = st.sidebar.selectbox('SUBURB', suburb_options)
    nearest_stn = st.sidebar.selectbox('NEAREST_STN', nearest_stn_options)
    nearest_sch = st.sidebar.selectbox('NEAREST_SCH', nearest_sch_options)

    # Set the constant values for other features
    other_values = {
        'SUBURB': suburbs,
        'NEAREST_STN': nearest_stn,
        'LATITUDE': -31.95,
        'NEAREST_SCH': nearest_sch,
        'ADDRESS': ' ',
        'PRICE': 0,
        'DATE_SOLD': ' ',
        'OTHERS_ROOMS_AREA': 1000,
    }

    user_info.update(sliders)
    user_info.update(other_values)
    row_df = pd.DataFrame.from_records([user_info])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])

    # Now you can use user_input_df for prediction
    price = pipeline.predict(row_df)[0]
    # Embed the video using the HTML video tag
    # Set up a video background
    video_path = 'src/img/perth_video.mp4'  # replace with your video path
    video_b64 = get_video_b64(video_path)
    st.markdown(f"""
        <video id="myVideo" width="100%" height="300px" controls autoplay muted playsinline loop>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)

    st.header('Housing Information:')
    # Set up some CSS properties
    st.markdown("""
        <style>
            .key {
                color: #ff8f00;  /* bright yellow color */
                font-weight: bold;
            }
            .value {
                color: #ffffff;  /* light green color */
            }
        </style>
        """, unsafe_allow_html=True)

    # Initialize columns
    columns = st.columns(3)

    # Create a list of sliders to maintain their order
    slider_items = list(sliders.items())
    sidebox_items = [('Suburb', suburbs), ('Nearest Station',
                                           nearest_stn), ('Nearest School', nearest_sch)]
    all_items = slider_items + sidebox_items

    for i, (key, value) in enumerate(all_items):
        # Calculate column index
        column_index = i % 3
        key_display = key.lower().replace('_', ' ').title()
        value_display = value
        html = (f"<p class='key'>{key_display}:</p>"
                f"<p class='value'>{value_display}</p>")
        columns[column_index].markdown(html, unsafe_allow_html=True)
    # Show house price
    st.header('Predicted House Price: ')
    # Define corresponding images
    img_dict = {
        'Level 1': 'src/img/level1.png',
        'Level 2': 'src/img/level2.png',
        'Level 3': 'src/img/level3.png',
        'Level 4': 'src/img/level4.png',
        'Level 5': 'src/img/level5.png',
        'Level 6': 'src/img/level6.png',
        'Level 7': 'src/img/level7.png',
    }

    # Add a selectbox to choose level:
    if price <= 400000:
        selected_level = 'Level 1'
    elif 400000 <= price < 500000:
        selected_level = 'Level 2'
    elif 500000 <= price < 600000:
        selected_level = 'Level 3'
    elif 600000 <= price < 700000:
        selected_level = 'Level 4'
    elif 700000 <= price < 5000000:
        selected_level = 'Level 5'
    elif 5000000 <= price < 6500000:
        selected_level = 'Level 6'
    else:
        selected_level = 'Level 7'
    # Display the corresponding image
    if selected_level in img_dict:
        # Create two columns
        col1, col2 = st.columns(2)

        # First column: markdown with price
        col1.markdown(f"""
            <div style="
                color: #ea80fc;
                font-size: 40px;
                font-weight: bold;
                text-shadow: 3px 3px 6px #FF69B4;
                margin-bottom: 50px;
            ">
                $ {price:,.2f}
            </div>
        """, unsafe_allow_html=True)
        image_path = 'src/img/best-price.png'
        image_b64 = get_image_b64(image_path)

        col1.markdown(f"""
            <div style="margin-left: 50px;">
                <img src="data:image/png;base64,{image_b64}" width="150" />
            </div>
        """, unsafe_allow_html=True)
        # Second column: image
        image = Image.open(img_dict[selected_level])
        col2.image(image, width=300)
