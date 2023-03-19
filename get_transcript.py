# Use open ai whisper to get transcript for a video file

import whisper
import subprocess
from openai.embeddings_utils import get_embedding
from vidcaption import *

embedding_model = "text-embedding-ada-002"
video_file = "input-video.mp4"

# get audio from video file
audio_file = "audio.mp3"

## get audio from video file ##
def get_audio(video_file, audio_file):
    if(os.path.exists(audio_file)):
        return
    command = "ffmpeg -i " + video_file + " -ab 160k -ac 2 -ar 44100 -vn " + audio_file
    subprocess.call(command, shell=True)

## get transcript using open ai whisper ##
def get_transcript(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result

## get sentences from transcript ##
def get_sentences(text):
    sentences = text.split(".")
    return sentences

## get scores for each sentence ##
def get_scores(segments):
    scores = []
    for segment in segments:
        embedding = get_embedding(segment['text'], engine = embedding_model)
        embedding2 = get_embedding(segment['caption'], engine = embedding_model)
        # add the two embeddings
        embedding = list(np.array(embedding) + np.array(embedding2))
        scores.append(embedding)
    return scores

## trim video file based on start and end time ##
def trim_video(video_file, output_file, start_time, end_time):
    command = "ffmpeg -i " + video_file + " -ss " + start_time + " -to " + end_time + " -c copy " + output_file
    subprocess.call(command, shell=True)


## get transcript and scores for a video file ##
def formula(video_file, output_file = None):
    base_file_name = video_file.split(".")[0]
    # get audio from video file
    audio_file = base_file_name + ".mp3"
    get_audio(video_file, audio_file)

    # get transcript
    result = get_transcript(audio_file)

    segments = result['segments']


    if(output_file == None):
        return segments
    else:
        # export to json
        import json
        with open(output_file, "w") as f:
            json.dump(segments, f)


# Cosine similarity between two vectors
def cosine_similarity(a, b):
    import numpy as np
    from numpy import dot
    from numpy.linalg import norm
    return dot(a, b)/(norm(a)*norm(b))

# Cosine similarity between a vector and a list of vectors
# Returns a list of cosine similarity values
def cosine_similarity_fast(a, b_list):
    # a is a vector
    # b_list is a list of vectors
    import numpy as np
    from numpy import dot
    from numpy.linalg import norm

    return dot(a, b_list)/(norm(a)*norm(b_list, axis = 1))


# UI STUFF ----------------------------------------------------------------------------------------------#
## A program to that will tkinter GUI  ##
## It will have a browse file button to take a video file as input ##
## It will have a button to start the process ##
## It will have a search box to search for a keyword in the transcript ##
## We will use async function to run the process in the background ##
## We will use ttk widgets to create the GUI ##

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import os
import subprocess
import json
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)

import sv_ttk


# create a window (ttk) ------------------------------------------------ #
window = Tk()
style = ttk.Style(window)
window.title("Video Searcher")
sv_ttk.set_theme(theme = "light")
window.geometry('620x600')

segments_dict = {}
start_column = 1
start_row = 2
for i in range(10):
    window.rowconfigure(i, minsize=20)
    window.columnconfigure(i, minsize=20)


# Add margin in grid layout
style.configure("TButton", margin = "10 10 10 10")
style.configure("TLabel", margin = "10 10 10 10")
style.configure("TEntry", margin = "10 10 10 10")
# ---------------------------------------------------------------------- #



# Header a label --------------------------------------------------------------------#
lbl = ttk.Label(window, text="Video Search", font=("Arial", 15))
lbl.grid(column=start_column + 1, row=start_row - 1, sticky='w')
# -----------------------------------------------------------------------------------#

# Open video folder -----------------------------------------------------------------#
# 1. Allow user to browse for a folder
# 2. Iterate through all the files in the folder
# 3. Get transcript for each file
# 4. Store the transcript in a dictionary
# 5. Get the starting timestamps of each sentence
# 6. Pass it to the vidcaption function

# create a button to browse file
def browse_folder():
    global segments_dict

    # Prompt the user to select a folder
    folder = filedialog.askdirectory(initialdir = "/", title = "Select a Folder")

    # Iterate through all the files in the folder
    mp4_files = []
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            mp4_files.append(file)
    
    # Update the user 
    lbl_text = StringVar()
    lbl_text.set('Loading files...')
    lbl_file = ttk.Label(window, text=lbl_text, font=("Arial", 10))
    lbl_file.grid(column=start_column + 1, row=start_row + 5, sticky='w')

    # No files found
    if(mp4_files == []):
        lbl_text.set('No mp4 files found in the folder')
        # delte lbl_file
        lbl_file = None
        return
    
    # Get transcript and score for each file
    for file in mp4_files:
        file_path = os.path.join(folder, file)
        seg = formula(file_path)
        segments_dict[file_path] = seg
    

    # Update the user
    lbl_text.set('Loaded ' + str(len(mp4_files)) + ' files')

    # Make a dict of filenames as keys and start timestamps as list
    file_timestamps = {}

    for file_name, segments in segments_dict.items():
        timestamps = []
        for segment in segments:
            timestamps.append(segment['start'])
        file_timestamps[file_name] = timestamps
    
    # Get the captions for each video
    vid_caption_dict = get_vid_text(file_timestamps)

    for file_name, captions in vid_caption_dict.items():
        # captions are a dict with start timestamp as key and caption as value
        for segment in segments_dict[file_name]:
            segment['caption'] = captions[segment['start']]

        # Get the embedding for each segment
        segments = segments_dict[file_name]
        scores = get_scores(segments)


        # # convert sentences and scores to json
        for i in range(len(segments)):
            segment = segments[i]
            score = scores[i]
            segment['score'] = score

    # write out to json file
    with open("segments_dict.json", "w") as f:
        json.dump(segments_dict, f)

    


btn = ttk.Button(window, text="  Browse File  ", command=browse_folder)
btn.grid(column=start_column + 1, row=start_row + 1, sticky='w')

or_label = ttk.Label(window, text="OR", font=("Arial Bold", 10))
or_label.grid(column=start_column + 2, row=start_row + 1, sticky='w')

# create a search box
def search():
    keyword = txt.get()
    global segments

    max_list = []

    embedding_search = get_embedding(keyword, engine = embedding_model)

    for file_name, segments in segments_dict.items():
        cosine_scores = []

        for segment in segments:
            embedding = segment['score']
            cosine_scores.append(cosine_similarity(embedding_search, embedding))
        
        

        max_cosine_score = max(cosine_scores)
        max_cosine_score_index = cosine_scores.index(max_cosine_score)

        segment = segments[max_cosine_score_index]
        start_time = segment['start']
        end_time = segment['end']

        max_list.append((file_name, start_time, end_time, max_cosine_score))

    max_list.sort(key=lambda x: x[3], reverse=True)
    max_start_time = max_list[0][1]
    max_end_time = max_list[0][2]
    max_file_name = max_list[0][0]

    # Make labels for start time and end time
    lbl_file_name = ttk.Label(window, text="File Name: " + max_file_name, font=("Arial", 10))
    lbl_file_name.grid(column=start_column + 1, row=start_row + 5, sticky='w', columnspan=3)

    lbl_start_time = ttk.Label(window, text="Start Time: " + str(max_start_time), font=("Arial", 10))
    lbl_start_time.grid(column=start_column + 1, row=start_row + 7, sticky='w')

    lbl_end_time = ttk.Label(window, text="End Time: " + str(max_end_time), font=("Arial", 10))
    lbl_end_time.grid(column=start_column + 1, row=start_row + 9, sticky='w')

    # Make a bar plot of the cosine scores of the max_file_name
    cosine_scores = []
    for segment in segments_dict[max_file_name]:
        embedding_segment = segment['score']
        cosine_score = cosine_similarity(embedding_search, embedding_segment)
        cosine_scores.append(cosine_score)
    

    # Save the fig with transparent background
    # plt.savefig('cosine_scores.png', transparent=True)
    
    # Make canvas for the bar plot
    fig, ax= plt.subplots(figsize=(4, 3), facecolor='#fafafa')
    ax.bar(range(len(cosine_scores)), cosine_scores)
    ax.axis('tight')
    ax.axis('off')
    # Make y axis from 0.6 to 1
    ax.set_ylim(0.7, 1)
    # bbox tight
    plt.tight_layout()
    # Remove all border and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # set facecolor to fafafa
    ax.set_facecolor('#fafafa')


    # Remove y axis markings
    ax.set_yticks([])
    # Remove x axis markings
    ax.set_xticks([])

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(column=start_column + 1, row=start_row + 11, sticky='w', columnspan=4)


# Add a button to load segments directly from json file, with browse button
def load_segments():
    global segments_dict
    segments_file = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("json files","*.json"),("all files","*.*")))
    with open(segments_file) as f:
        segments_dict = json.load(f)

btn = ttk.Button(window, text="Browse Segments", command=load_segments)
btn.grid(column=start_column + 3, row=start_row + 1, sticky='w')

txt = ttk.Entry(window,width=40)
txt.grid(column=start_column + 1, row=start_row + 3, columnspan=2)

btn = ttk.Button(window, text="Search", command=search)
btn.grid(column=start_column + 3, row=start_row + 3)

window.mainloop()
