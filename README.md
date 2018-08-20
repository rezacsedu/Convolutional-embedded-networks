# VariationDEC
The DEC implementation of our paper titled "Recurrent Deep Embedding Networks for Genotype Clustering and Ethnicity Prediction". Note that the journal version submission is ongoing. Then the CDEC version will be uploaded here. 

Pre-print link: https://arxiv.org/pdf/1805.12218.pdf

## How to use this repository
 Step 1: Feature extraction from the 1000 Genomes Project genetic variants data files in Scala, Adam and Spark. 
 For this, first download the VCF files (containing the variants) and and the panel file (containing the labels) from ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/. 
 
Step 2: Then go to https://github.com/rezacsedu/VariationDEC/tree/master/PopulationClustering_v2 and use the featureExtractor.scala
to extract the features and save as a DataFrame in CSV to be used by Keras-based DEC.
For this, make sure that you've configured Spark correctly on your machine. Alternatively, execute this script as a standalone Scala project from Eclipse or IntelliJ IDEA. 

Step 3: This is the DEC part in Keras/Python. Go to https://github.com/rezacsedu/VariationDEC/tree/master/DEC_GenotypeClustering_Keras. Then there are 2 Python scripts and a sample genetic variants featres in csv for the clustering and classification respectively. 

- genome.csv: is the sample genetic variants featres
- DEC_Genotype_Clustering.py: for the clustering 
- LSTM_EthnicityPrediction.py: for the classification 


