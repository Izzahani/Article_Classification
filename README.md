![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

# Classification of Texts in Articles by using TensorFlow
 
 ## Summary
<p>Text documents are crucial for companies since they are one of the richest sources of data. Text documents frequently include important information that could impact investment flows or shape market patterns. As a result, companies frequently employ analysts to track trends through web publications, tweets on social media sites like Twitter, or newspaper articles. Some companies, however, might prefer to limit their attention to news about politics and technology. As a result, it is necessary to sort the articles into various categories.</p>
<p>Thus, this initiative is being performed to assist businesses in the categorization of article text.</p>

<p>Before moving further with model development for this project, there are 5 steps that must be completed.</p>
<p>1. Data Loading</p>
  <ol>- Upload the dataset using pandas</ol>
  <ol>- In this project, I use link as path to upload the dataset. However, pd.read_csv( <strong>your_path.csv</strong> ) still need to be applied to upload the dataset.</ol>
  <ol>- In this part, dataframe for features and target need to be declared. I declared text as features and category as target.</ol>
  
  
<p>2. Data Inspection</p>
   <ol>- Inspect the dataset to check whether the dataset contains NULL, duplicated data or any other unwanted things.</ol>
   <ol>- I used <strong>df.info()</strong> to find the amount of NULL in the data. </ol>
   <ol>- Then, I used <strong>df.duplicated().sum()</strong> to find duplicated data. I found 99 duplicated data that need to be cleaned.</ol>
   <ol>- Based on the data inspection I did, the dataset can be clean easily by only using regex. </ol>

<p>3. Data Cleaning</p>
   <ol>- Data cleaning is important to increase overall productivity and allow for the highest quality information in your decision-making.</ol>
   <ol>- I used Regex to remove unwanted words which then leave only the words with alphabets A-Z</ol>
   <ol>- The alphabets are then all converted in lower case.</ol>
   <ol>- All of the duplicated data has been removed in this part as well.</ol>

<p>4. Features Selection</p>
   <ol>- In this part, I have to declare feature and target again by using other dataframe variable to avoid overwrite.</ol>
          
<p>5. Data Pre-processing</p>
   <ol> <strong>For feature:</strong></ol>
   <ol>- Tokenizers is being used in this part to convert the text into numerical.</ol>
   <ol>- I used train sequences to convert the text to horizontal.</ol>
   <ol> <strong>For target:</strong></ol>
   <ol>- One Hot Encoder is being used to convert the outputs which are politicsNews and worldnews to 1.0 and 0.1 respectively.</ol>
   
 <p>Finally, <strong>Model Development</strong> can be done if all of the steps above has already finished.</p>
 <p> In Model Development, I did train-test split. Then, i used Embedding as an input layer. For hidden layers, I used LSTM.</p>
 <p> Then, the project is being compiled. This is my result:</p>
 <img src="https://github.com/Izzahani/Text_Classification/blob/main/Predictions1.png" alt="descriptive text">
 <p>These are the graph I got from TensorBoard</p>
 <p>For Epoch Acc</p>
 <img src="https://github.com/Izzahani/Text_Classification/blob/main/epoch_acc.png" alt="descriptive text">

 <p>For Epoch Loss</p>
 <img src="https://github.com/Izzahani/Text_Classification/blob/main/epoch%20loss.png" alt="descriptive text">
 
## Acknowledgement
Special thanks to [(https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv)](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv) :smile:

