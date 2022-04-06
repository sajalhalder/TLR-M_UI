# TLR-M_UI
In this work, we propose a multi-task, multi-head attention transformer model. The model recommends the next POIs to the target users and predicts queuing time to access the POIs simultaneously by considering user's mobility behaviors. The proposed model utilizes POIs description-based user personalised interest that can also solve the new categorical POI cold start problem. Extensive experiments on six real datasets show that the proposed models outperform the state-of-the-art baseline approaches in terms of precision, recall, and F1-score evaluation metrics. The model also predicts and minimizes the queuing time effectively.


To use this code in your research work please cite the following paper.  


Sajal Halder, Kwan Hui Lim, Jeﬀrey Chan, and Xiuzhen Zhang. POI Recommendation with Queuing Time and User Interest Awareness. In Submission, 2022.

In this research work, we aim to answer the following research questions. 
  
     (i) Are users' interests important for recommending top-k POIs?  
     
    (ii) Are users' interests important for recommending top-k POIs and predicting queuing times simultaneously? 
    
    (iii) How does POI description based users' interests perform compared to the POI categorical based users' interests? 



# Implemtation Details
In this TLR-M_UI model implemenation, we have used transformer based attention machanism that has been implemented in python programing language. We use tensorflow, keras and attention machanism. 

Required Packages:

tensorflow: 2.4.1

pandas: 1.2.2

TLR_UI model has been implemented in TLR_UI-POIDes.py file

TLR-M_UI model has been implemented in TLRM_UI-POIDes.py file


Here we added only one dataset (Melbourne). If you are interested to know about more datasets email at sajal.halder@student.rmit.edu.au or sajal.csedu01@gmail.com
