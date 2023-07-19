import streamlit as st
from PIL import Image
from keras.models import load_model
import tensorflow

import numpy as np
from keras.models import Sequential

from keras.preprocessing import image
st.set_page_config(page_title="Fruit Health",page_icon="fruit-21.jpg")
page_bg_img="""
<style>
[data-testid="stAppViewContainer"]
{
background-color: #9bebff;
opacity: 1;
background-image: radial-gradient(#09c455 1.3px, #9bebff 1.3px);
background-size: 26px 26px;

}
[data-testid="stHeader"]
{
background-color: rgba(0,0,0,0);
}
"""
st.markdown(page_bg_img,unsafe_allow_html=True)


model = load_model('Fresh_Rotten_Fruits_MobileNetV2_Transfer_Learning2(98).h5')
lab = {0: 'freshapples',1:'freshbanana',2:'freshoranges',3:'rottenapples',4:'rottenbanana',5:'rottenoranges'}

def processed_img(img_path):
    img=tensorflow.keras.utils.load_img(img_path,target_size=(224,224,3))
    img=tensorflow.keras.preprocessing.image.img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(answer)
    return res

def run():
    #img1 = Image.open('WhatsApp Image 2023-06-22 at 10.45.28.jpg,'height)
    #img1 = img1.resize((350,350))
    #st.image(img1,use_column_width=False,)
    st.title("Fruit Health Analysis")
    #st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "13000 samples of  apples,banana,oranges"</h4>''',
               # unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Fruit of your choice:", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Get Report"):
            result = processed_img(save_image_path)
            st.success("Predicted Fruit is: "+result)
            #st.write('%s (%.2f%%)' % (lab['fresh'], lab['rotten']*100))
run()