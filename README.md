Why is 99% accuracy often the wrong goal in Machine Learning?

When building an Email & SMS Spam Classifier, I learned firsthand that standard metrics can be deceptive if they ignore the real-world cost of errors.

Here is the breakdown of the project and its core insights:

The Metric Dilemma (Precision > Accuracy): The dataset is naturally skewed (~87% Ham vs. ~13% Spam). 
A model that predicts "Ham" every time scores 87% accuracy while failing completely. More crucially, 
the error penalty is asymmetrical: a False Negative (spam hitting your inbox) is a minor annoyance, 
but a False Positive (a job offer or banking alert dumped into spam) is a critical failure. 
Maximizing Precision for the Spam class was the primary objective.

NLP Preprocessing Pipeline: Raw text was cleaned and normalized using NLTK (lowercasing, punctuation and special character removal, stop-word filtering)
and stemmed via PorterStemmer to collapse morphological variants (e.g., winning, wins, won  --> win).

TF-IDF Vector Space: Extracted the top 3,000 features using TF-IDF Vectorization to downweight ubiquitous filler 
words while prioritizing distinctive spam signatures.

Model Benchmarking: Evaluated 11+ algorithms on the same stratified test set (Logistic Regression, Support Vector Machines, 
Random Forest, XGBoost, and Stacking/Voting ensembles).
The Standout: Multinomial Naive Bayes (MNB) outperformed complex ensembles by achieving high overall accuracy paired with 100% Precision 
(zero False Positives) on the test evaluation.

The full pipeline and model artifacts were serialized using pickle and deployed into an interactive Streamlit dashboard for real-time message analysis.

#MachineLearning #NaturalLanguageProcessing #Python #ScikitLearn #TextClassification #ArtificialIntelligence
