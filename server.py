#coding=utf-8
from flask import Flask
from flask import render_template
from flask import Response, request, jsonify
import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import joblib

import re
import sys
import getopt
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer


#UPLOAD_FOLDER = r'/Users/wenyifei/Desktop/file'   # 上传路径

UPLOAD_FOLDER = r'./static/file'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

newFile='YifeiWen'
import pdfplumber
#path = '/Users/wenyifei/Desktop/file/'
path = './static/file/'

import joblib
tf_jobs = joblib.load('./static/file/tf_jobs.pkl')
tf = joblib.load('./static/file/tf.pkl')
#jobs=pd.read_csv('./static/file/processed_data.csv')
current_id = 2
data = [
    {
        "id": 1,
        "name": "michael scott"
    },
    {
        "id": 2,
        "name": "jim halpert"
    },
]

# ROUTES

@app.route('/hi')
def hello():
   return 'Hi hi hi hi hi hi hi hi hi'

@app.route('/hello/<name>')
def hello_name(name=None):
    return render_template('hello_name.html', name=name)

@app.route('/')
def hello_world():
   return render_template('homePage.html')

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html', data=data)


# AJAX FUNCTIONS

# ajax for people.js
@app.route('/add_name', methods=['GET', 'POST'])
def add_name():
    global data 
    global current_id 

    json_data = request.get_json()   
    name = json_data["name"] 
    
    # add new entry to array with 
    # a new id and the name the user sent in JSON
    current_id += 1
    new_id = current_id 
    new_name_entry = {
        "name": name,
        "id":  current_id
    }
    data.append(new_name_entry)

    #send back the WHOLE array of data, so the client can redisplay it
    return jsonify(data = data)

########################################################################
#function added
def job_resume_matching(jobs,path):
    #define parse resume function
    def parse_resume(pdf_line):
        title = {'EDUCATION':[],'SKILLS':[],'EXPERIENCES':[],'AWARDS':[]}
        i = 0
        while i < len(pdf_line):
            flag = True
            if 'EDUCATION' in pdf_line[i].strip().upper().split() or 'EDUCATIONS' in pdf_line[i].strip().upper().split():
                j = i + 1
                while j < len(pdf_line) - 1:
                    title['EDUCATION'].append(pdf_line[j])
                    j += 1
                    i = j
                    for key in title:
                        if key in pdf_line[j].strip().upper().split() or key[0:-1] in pdf_line[j].strip().upper().split():
                            flag = False
                        if not flag:
                            break
                    if not flag:
                        break



            elif 'SKILL' in pdf_line[i].strip().upper().split() or 'SKILLS' in pdf_line[i].strip().upper().split():
                j = i + 1
                while j < len(pdf_line) - 1:
                    title['SKILLS'].append(pdf_line[j])
                    j += 1
                    i = j
                    for key in title:
                        if key in pdf_line[j].strip().upper().split() or key[0:-1] in pdf_line[j].strip().upper().split():
                            flag = False
                        if not flag:
                            break
                    if not flag:
                        break



            elif 'EXPERIENCE' in pdf_line[i].strip().upper().split() or 'EXPERIENCES' in pdf_line[i].strip().upper().split():
                j = i + 1
                while j < len(pdf_line) - 1:
                    title['EXPERIENCES'].append(pdf_line[j])
                    j += 1
                    i = j
                    for key in title:
                        if key in pdf_line[j].strip().upper().split() or key[0:-1] in pdf_line[j].strip().upper().split():
                            flag = False
                        if not flag:
                            break
                    if not flag:
                        break

            elif 'AWARDS' in pdf_line[i].strip().upper().split() or 'AWARDS' in pdf_line[i].strip().upper().split() or 'HORNOR' in pdf_line[i].strip().upper().split() or 'HORNORS' in pdf_line[i].strip().upper().split():
                while j < len(pdf_line) - 1:
                    title['EXPERIENCES'].append(pdf_line[j])
                    j += 1
                    i = j
                    for key in title:
                        if key in pdf_line[j].strip().upper().split() or key[0:-1] in pdf_line[j].strip().upper().split():
                            flag = False
                        if not flag:
                            break
                    if not flag:
                        break

            else:
                i += 1

        return title


    #extract pdf
    pdf = ''
    with pdfplumber.open(path) as rpdf:
        for i in range(len(rpdf.pages)):
            page = rpdf.pages[i]
        #first_page = rpdf.pages[0]
            pdf += page.extract_text()
    #1-page resume
    pdf_line = pdf.split('\n')

    #extract skills in resume
    title = parse_resume(pdf_line)
    resume_skill = ''
    for i in range(len(title['SKILLS'])):
        resume_skill += title['SKILLS'][i]

    #clean skill
    resume_skill = resume_skill.strip().lower()
    if ',' not in resume_skill and ';' not in resume_skill:
        #resume是以空格隔开skill的格式
        resume_skill = resume_skill.split(' ')
    else:
        #resume_skill = resume_skill.replace(' ','')
        resume_skill = re.split('\W+',resume_skill)
    ps = PorterStemmer()
    num_skill_resume = {}
    for word in resume_skill:
        if word not in stopwords.words('english'):
            if word not in num_skill_resume:
                num_skill_resume[ps.stem(word)] = 0

    #clean pdf
    for i in range(len(pdf)):
        cv_pdf = re.sub('[^a-zA-Z]', ' ', pdf)

    cv_pdf = cv_pdf.lower().split()
    temp = []
    for j in range(len(cv_pdf)):
        if not cv_pdf[j] in stopwords.words('english'):
            temp.append(ps.stem(cv_pdf[j]))
    cv_pdf = temp

    #count number of skills in resume
    for x in cv_pdf:
        if x in num_skill_resume:
            num_skill_resume[x] += 1

    #get pre-trained model (weight)
    weight_skills_job = joblib.load('./static/model/skill_weight_job.pkl')
    weight_skills_applicant = joblib.load('./static/model/skill_weight_applicant.pkl')

    #job-resume matching
    res = {}
    for i in range(len(jobs)):
        for skill in num_skill_resume:
            if (skill in jobs['skills'][i]) and  (skill in weight_skills_applicant) and (skill in weight_skills_job):
                if i not in res:
                    res[i] = weight_skills_job[skill] * weight_skills_applicant[skill] * num_skill_resume[skill]
                else:
                     res[i] += weight_skills_job[skill] * weight_skills_applicant[skill] * num_skill_resume[skill]

    #get top N results
    sorted_res = sorted(res.items(),key = lambda x : x[1] , reverse = True)

    return sorted_res

#######################################################################


@app.route('/pdf_upload', methods=['GET', 'POST'])
def upload_file():
    global newFile 
    if request.method == 'POST':
        file = request.files['photo']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #with pdfplumber.open(path + filename) as rpdf:
            jobs = joblib.load('./static/model/cleaned_jobs_data.pkl')
            sorted_res = job_resume_matching(jobs,path + filename)
            recommendRes = jobs.iloc[[sorted_res[0][0],sorted_res[1][0],sorted_res[2][0]]].reset_index(drop=True)
            #res['jobdescription'][0]
            #answer=jobs['Query'][sorted_similarity_index[0][1]]
            #answer = []
            #answer.append(jobs['Query'][sorted_similarity_index[0][1]])
            #answer.append(jobs['Query'][sorted_similarity_index[1][1]])
            #answer.append(jobs['Query'][sorted_similarity_index[2][1]])
            f_matrix = joblib.load('./static/model/preprocessed_jobs_tfidf.pkl')
            sim1 = {}
            sim2 = {}
            sim3 = {}
            for i in range(3):
                #job_index = jobs.iloc[sorted_res[i][0]]['index']
                job_index = sorted_res[i][0]
                for j in range(f_matrix .shape[0]):
                    if j == i:
                        continue
                    if i == 0:
                        sim1[j] = cosine_similarity(f_matrix[job_index,:],f_matrix[j,:])[0][0]
                    elif i == 1:
                        sim2[j] = cosine_similarity(f_matrix[job_index,:],f_matrix[j,:])[0][0]
                    else:
                        sim3[j] = cosine_similarity(f_matrix[job_index,:],f_matrix[j,:])[0][0]

            sorted_res1 = sorted(sim1.items(),key = lambda x : x[1] , reverse = True)
            sorted_res2 = sorted(sim2.items(),key = lambda x : x[1] , reverse = True)
            sorted_res3 = sorted(sim3.items(),key = lambda x : x[1] , reverse = True)

            '''
            answer1=jobs['Query'][sorted_similarity_index[0][1]]
            answer2=jobs['Query'][sorted_similarity_index[1][1]]
            answer3=jobs['Query'][sorted_similarity_index[2][1]]
            answer = answer1 + " " + answer2 + " " + answer3
            '''
            #kkk = dataset['Description'][sorted_similarity_index[0][1]]
            #print(kkk)
            answer = {}
            answer[0]= recommendRes['jobtitle'][0]
            answer[1]= recommendRes['jobtitle'][1]
            answer[2]= recommendRes['jobtitle'][2]
            answer[3] = recommendRes['jobdescription'][0]
            answer[4] = recommendRes['jobdescription'][1]
            answer[5] = recommendRes['jobdescription'][2]
            answer[6]= recommendRes['joblocation_address'][0]
            answer[7]= recommendRes['joblocation_address'][1]
            answer[8]= recommendRes['joblocation_address'][2]
            answer[9]= recommendRes['company'][0]
            answer[10]= recommendRes['company'][1]
            answer[11]= recommendRes['company'][2]
            answer[12]= recommendRes['skills'][0]
            answer[13]= recommendRes['skills'][1]
            answer[14]= recommendRes['skills'][2]
            res = jobs.iloc[[sorted_res1[0][0],sorted_res1[1][0],sorted_res1[2][0]]].reset_index(drop=True)
            answer[15] = res['jobtitle'][0]
            answer[16] = res['jobtitle'][1]
            answer[17] = res['jobtitle'][2]
            answer[18] = res['joblocation_address'][0]
            answer[19] = res['joblocation_address'][1]
            answer[20] = res['joblocation_address'][2]
            print(answer)
            print(answer[15])
            return answer   # 返回保存成功的信息
    return 'fail'

if __name__ == '__main__':
   app.run(debug = True)




