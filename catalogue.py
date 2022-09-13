import streamlit as st
from PIL import Image
import streamlit as st
import random
import pickle
import numpy as np
import faiss
import random
import glob
import os
import time
import urllib.request
# from sklearn.externals import joblib

def load_index():

    '''
    load the index using faiss library
    '''
    with open("faiss_ivfpqindex.pkl", "rb") as f:
        ivfpq_index = faiss.deserialize_index(pickle.load(f))
    return ivfpq_index

def search(query):
  t =time.time()
  # print("query=",query)

  query_vector = train_dict[query]
  print(query_vector)
  print(type(query_vector))
  k = 5
  #  top_k = gpu_index.search(query_vector, k)
  # print(type(query_vector))

  # print(query_vector.shape)
  D, I = ivfpq_index.search(query_vector, k) # sanity check
  print("top_k=",D)
  print('totaltime: {}'.format(time.time()-t))
  results = [documents[_id] for _id in I.tolist()[0]]
  metrics = [dist for dist in D.tolist()[0]]
  print("inside search metrics=", metrics)

  return results,metrics

# callback to update emojis in Session State
# in response to the on_click event
def random_emoji():
    st.session_state.emoji = random.choice(emojis)

# initialize emoji as a Session State variable

if "emoji" not in st.session_state:
    st.session_state.emoji = "💖"

emojis = ["✔️", "🆒"]

def random_file():
    files = glob.glob('./16k_images/*.jpeg')
    array_file = random.choice(files)
    return array_file

# if "filename" not in st.session_state:
#     st.session_state.filename = glob.glob('./16k_images/B00BGEM4QC.jpeg')


# train_dict = joblib.load(urlopen("https://drive.google.com/open?id=1M7Dt7CpEOtjWdHv_wLNZdkHw5Fxn83vW"))
# train_dict = pickle.load(urllib.request.urlopen("https://drive.google.com/file/d/10N6WDPH9g3sGegtykVWUi9wuGpfHidy0/view?usp=sharing"),'rb')
train_dict = pickle.load(open('trainxception_embed.pickle', 'rb'))

documents = []
vectors = []
for k,v in train_dict.items():
  documents.append(k)
  vectors.append(v)
#len(documents),len(vectors)

# print(type(vectors))

array_vec = np.squeeze(np.asarray(vectors), axis=1)
# print(array_vec.shape)
ivfpq_index = load_index()




#Load the pkl files, model and index vector
#read them

#ivfpq = load_index()


placeholder = st.empty()

with st.form("my_form"):
    st.header("**Category:Shirts**")
    jpgfile = 'B071KCWVH5.jpeg'
    placeholder.image(os.path.join("./16k_images/",jpgfile))


    # Every form must have a submit button.
    submitted = st.form_submit_button("Like 💖")

    if submitted:
        results, metrics = search(jpgfile)
        if results is not None:
            for i,(result,metric) in enumerate(zip(results,metrics)):
                cols = st.columns(2)
                cols[0].image(os.path.join("./16k_images/",result), use_column_width=True)
                cols[0].text(metric)
                wished = cols[1].form_submit_button("Add Wishlist_"+ str(i)+"🆒")

                if wished:
                    st.success('These items are added to wishlist!', icon="✅")
        # elif other_submitted:

final_res_vec_file = pickle.load(open('explore.pickle', 'rb'))
files = final_res_vec_file[final_res_vec_file['cluster_labels'] == 1]['filename']
with st.form("form_2"):
    different = st.form_submit_button("Different category")
    if different:
        st.header("**New Arrivals coming soon**")
        for f_n in files[90:96]:
            st.image(os.path.join("./16k_images/",f_n+".jpeg"))
