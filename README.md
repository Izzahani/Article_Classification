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
   <ol>- I have used <strong>df.isna().sum()</strong> to find the amount of NaN in the data. </ol>
   <ol>- Then, I have used <strong>df.duplicated().sum()</strong> to find duplicated data. I found 99 duplicated data that need to be cleaned.</ol>
   <ol>- Based on the data inspection I did, duplicated data can be clean easily by only using regex. </ol>

<p>3. Data Cleaning</p>
   <ol>- Data cleaning is important to increase overall productivity and allow for the highest quality information in your decision-making.</ol>
   <ol>- I used Regex to remove unwanted words which then leave only the words with alphabets A-Z</ol>
   <ol>- The alphabets are then all converted in lower case.</ol>
   <ol>- All of the duplicated data has been removed in this part as well.</ol>
  
<p>Additionally, it is critical to find out the average word length in each sentence because we need to choose the right length when declaring the maximum length during train sequences in order to improve accuracy.</p>

<p>4. Features Selection</p>
   <ol>- In this part, I have to declare feature and target again by using other dataframe variable to avoid overwrite.</ol>
          
<p>5. Data Pre-processing</p>
   <ol> <strong>For feature:</strong></ol>
   <ol>- Tokenizers is being used in this part to convert the text into numerical.</ol>
   <ol>- I have used train sequences to convert the text to horizontal.</ol>
   <ol> <strong>For target:</strong></ol>
   <ol>- One Hot Encoder is being used to convert the outputs into numerical. 5 outputs need to be converted which are Sport, Tech, Business, Entertainment and 
Politics. </ol>
   
 <p>Finally, once all of the above steps have been completed, <strong>model development</strong> can begin.</p>
 <p> In Model Development, I did train-test split. Then, i used Embedding as an input layer. For hidden layers, I used LSTM, Dropout and Bidirectional.</p>
 <p>I used Bidirectional LSTMs because it can imporove my model performance.</p>
   <p align="center"><img src="https://github.com/Izzahani/Article_Classification/blob/main/model.png" alt="descriptive text">
 
 <p> Then, the project is being compiled. The result as shown in the picture below:</p>
  <p align="center"><img src="https://github.com/Izzahani/Article_Classification/blob/main/prediction.png" alt="descriptive text">
 <div align="center"><ol>The f1-score for accuracy based on the image is 81%. However, eliminating stop words can increase accuracy. Additionally, adding another dense layer might increase accuracy as well.</ol></div>
 
 <p>For Epoch Acc</p>
  <p align="center"><img src="https://github.com/Izzahani/Article_Classification/blob/main/epoch_acc.png" alt="epoch acc">
 <div align="center"><ol>According to the graph, the train model was coloured green. Between 1.5 and 2, the model started to get overfitted. Perhaps the Bidrectional LSTM layer is at cause. But when it reached to 2, it started to get close to optimum. So maybe the dropout layer is the reason. But at 3, it became overfitted again. I might need to add another dropout layer to reached the good git graph.</ol></div>

 <p>For Epoch Loss</p>
  <p align="center"><img src="https://github.com/Izzahani/Article_Classification/blob/main/epoch_loss.png" alt="epoch loss">
 
## Acknowledgement
Special thanks to [(https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv)](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv) :smile:

