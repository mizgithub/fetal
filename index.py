from struct import unpack
from typing_extensions import Protocol
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
import pickle
import csv
import os
def classifier(data):
    f = open('randomForestClassifier.pkl', 'rb')
    model = pickle.load(f)
    f.close()
    data = np.array(data)
    data = data.reshape(1, 13)
    result = model.predict_proba(data)
    result=  result[0]
    maxindex = np.argmax(result)
    
    label = ["Normal", "Suspect", "Pathological"]
    class_result = "<p>NB: The highlighted one is the highest probable felat heath status  </p><table style='width:100%'>"
    for i in range(0,len(result)):
        if i == maxindex:
            class_result+="<tr style='background-color:#999999;font-weight:bold'><td>"+label[i]+"</td><td>"+str(round(result[i]*100,2))+" %</td></tr>"
        else:
            class_result+="<tr><td>"+label[i]+"</td><td>"+str(round(result[i]*100,2))+" %</td></tr>"
    class_result+="</table>"

    return class_result
def ctg_processor(fhr):
    #baseline
    fhr_signal = []
    for i in fhr:
        x = i[0]
        if ~np.isnan(x):  # removinf nan
            fhr_signal.append(x)
    baseline = round(np.average(fhr_signal),0)

    #variablity
    variablity = np.std(fhr_signal)

    #prolongued deceleration
    deceleration = []
    x = []
    for i in fhr_signal:
        if i<baseline:
            x.append(i)
        else:
            deceleration.append(x)
            x=[]
    proloDecelPerM = 0
    for i in deceleration:
        dec = len(i)/(4*60)  #perminute
        if dec > 3:
            proloDecelPerM+=1
    number_of_prolongued_deceleration_per_sec = proloDecelPerM/(len(fhr_signal)/4)

    # acceleration
    acceleration_array = []
    x = []
    for i in fhr_signal:
        if i>(baseline+15): # greater than 15bpm
            x.append(i)
        else:
            acceleration_array.append(x)
            x = []
    num_accelerations = 0

    for i in acceleration_array:
        if len(i)/4 > 15:
            num_accelerations+=1
    acceleration_per_sec = num_accelerations/(len(fhr_signal)/4)

    # histogram
    
    hist, bin_edges = np.histogram(fhr_signal)
    # hsitogram min
    hist_min = np.min(bin_edges)
    #histogram max
    hist_max = np.max(bin_edges)
    #histogram width
    hist_width = hist_max - hist_min
    # histogram peaks
    hist_number_of_peaks = 0
    for i in bin_edges:
        if i > baseline:
            hist_number_of_peaks+=1
    # histogram number of zeros
    hist_num_zeros = 0
    for i in bin_edges:
        if i == 0:
            hist_num_zeros+=1
    # histogram mode
    hist_mode = stats.mode(bin_edges)[0][0]
    # histogram mean
    hist_mean = np.mean(bin_edges)

    # histogram median
    sorted_bin_edges = np.sort(bin_edges)
    middle = round(len(sorted_bin_edges)/2)
    hist_median = sorted_bin_edges[middle]

    # histogram variance
    hist_var = np.std(bin_edges)

    data = [baseline, acceleration_per_sec, number_of_prolongued_deceleration_per_sec, variablity,hist_width, hist_min, hist_max,hist_number_of_peaks,hist_num_zeros,hist_mode,hist_mean, hist_median, hist_var]

    classification_result = classifier(data)
    

    string_result = "<table style='font-size:12px'><tr><th>Features</th><th>Exctracted Value</th><th>Features</th><th>Exctracted Value</th></tr>"
    string_result+="<tr><td>1. Base line FHR</td><td style='color:yellow'>"+str(baseline)+"</td><td>8. FHR Variability </td><td style='color:yellow'>"+str(variablity)+"</td></tr>"
    string_result+="<tr><td>2. Prolongued deceleration</td><td style='color:yellow'>"+str(number_of_prolongued_deceleration_per_sec)+"</td><td>9. Acceleration</td><td style='color:yellow'>"+str(acceleration_per_sec)+"</td></tr>"
    string_result+="<tr><td>3. Histogram min</td><td style='color:yellow'>"+str(hist_min)+"</td><td>10. Histogram max</td><td style='color:yellow'>"+str(hist_max)+"</td></tr>"
    string_result+="<tr><td>4. Histogram Mode</td><td style='color:yellow'>"+str(hist_mode)+"</td><td>11. Histogram mean</td><td style='color:yellow'>"+str(hist_mean)+"</td></tr>"
    string_result+="<tr><td>5. Histogram median</td><td style='color:yellow'>"+str(hist_median)+"</td><td>12. Histogram number of peaks</td><td style='color:yellow'>"+str(hist_number_of_peaks)+"</td></tr>"
    string_result+="<tr><td>6. Histogram number of zeros</td><td style='color:yellow'>"+str(hist_num_zeros)+"</td><td>13. Histogram variance</td><td style='color:yellow'>"+str(hist_var)+"</td></tr>"
    string_result+="<tr><td>7. Histogram Width</td><td style='color:yellow'>"+str(hist_width)+"</td><td></td><td></td></tr>"

    string_result+="</table>"
    return string_result, classification_result
def header():
    col1, col2 = st.columns([3,9])
    with col1:
        st.image("./headerImage.png")
    with col2:    
        st.markdown("<h2>Fetal Health Status Classification System</h2>", unsafe_allow_html = True)
    st.markdown("<hr>", unsafe_allow_html = True)

def body():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col1, _,col2= st.columns([6,1,5])
    process_result = None
    with st.form(key = "signal uload"):
        with col1:
            file_path = st.file_uploader("Select CTG data (.mat)")
            sfrq = st.text_input("Enter sampling friquency in Hz")
        if file_path != None and sfrq!="":
            ctg_signal = loadmat(file_path)
        
            with col2:
                fhr  = ctg_signal['fhr']
                sample_signal = fhr[: int(sfrq)*180] # ten minute sample
                plt.plot(sample_signal)
                plt.title("Three minute sample signal")
                st.pyplot()      
            if st.form_submit_button("Start processing "):
                process_result,classification_result  = ctg_processor(fhr)
                with st.spinner("Please wait . . ."):
                    # st.sleep(1)
            
                    if process_result !=None:
                        col3, _,col4 = st.columns([6,1,5])
                        with col3:
                            st.markdown(process_result, unsafe_allow_html = True)
                        with col4:
                            st.success("Predicted fetal health status")
                            st.markdown(classification_result, unsafe_allow_html = True)

        
def ctg_def():
    st.subheader("CardioTocoGraph")         
def fetal_health_def():
    st.subheader("Fetal Health")
def parameters_def():
    st.subheader("Parameters")

def about_us():
    st.subheader("About us")
st.set_page_config(layout="wide")

header()

st.sidebar.title("Menu")

option = st.sidebar.radio("",["Home","Cardiotocograph","Fetal Health","Parameters"],index=0)
if option == "Home":
    body()
elif option == "Cardiotocograph":
    ctg_def()
elif option == "Fetal Health":
    fetal_health_def()
elif option == "Parameters":
    parameters_def()
else:
    body()

st.sidebar.markdown("<br><br><br>", unsafe_allow_html= True)
if st.sidebar.button("About us"):
    about_us()