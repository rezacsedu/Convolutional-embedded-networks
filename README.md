This repository contains the implemnetation of our papers titled "Recurrent Deep Embedding Networks for Genotype Clustering and Ethnicity Prediction" and "Recurrent Deep Embedding Networks for Population Scale Clustering and Ancestry Inference". The former is available on "Arxiv as pre-print"(link: https://arxiv.org/pdf/1805.12218.pdf). The later has been submitted to IEEE/ACM Transactions on Computational Biology and Bioinformatics, which is under review. 

This repo will have two different implementations: i) Deep Embedding Networks(DEC) and Recurrent Deep Embedding Networks(CDEC) using ii) Spark and H2O implementations of our paper titled "Recurrent Deep Embedding Networks for Genotype Clustering and Ethnicity Prediction". 

## Implementation details
The proof of the concept of our approach is implemented in Spark, ADAM, and Keras. In particular, for the scalable and faster preprocessing of huge number of genetic variants across all the chromosomes (i.e. 870 GB of data), we used ADAM and Spark to convert the genetic variants from VCF format to Spark DataFrame. Then we convert Spark DataFrame into NumPy arrays. Finally, we use Keras to implement Conv-LSTM and CDEC networks for for Population Scale Clustering and Ancestry Inference, respectively. 

Experiments were carried out on a computing cluster having 32 cores, 64-bit Ubuntu 14.04 OS. Software stack consisting of Apache Spark v2.3.0, H2O v3.14.0.1, Sparkling Water v1.2.5, ADAM v0.22.0 and Keras v2.0.9 with TensorFlow backend. We compare approach with the state-of-the-art such as ADMIXTURE and VariationSpark. 

### CDEC implementation in Python with Keras
Refer to https://github.com/rezacsedu/Recurrent-Deep-Embedding-Networks/tree/master/CDEC for more details. Network training were carried out on a Nvidia TitanX GPU with CUDA and cuDNN enabled to make the overall pipeline faster. 

#### Step 1: Feature extraction using Scala, Adam, and Spark 
For this, first, download the VCF files (containing the variants) and the panel file (containing the labels) from ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/. 
 
Then go to https://github.com/rezacsedu/VariationDEC/tree/master/PopulationClustering_v2 and use the featureExtractor.scala
to extract the features and save as a DataFrame in CSV to be used by Keras-based DEC.

For this, make sure that you've configured Spark correctly on your machine. Alternatively, execute this script as a standalone Scala project from Eclipse or IntelliJ IDEA. 

#### Step 2: This is the CDEC part in Keras 
Go to https://github.com/rezacsedu/Recurrent-Deep-Embedding-Networks/tree/master/CDEC. Then there are several Python scripts and a sample genetic variants feature in csv for the clustering and classification, respectively. 

- genome.csv: is the sample genetic variants featres
- customlayers.py: for creating custom clustering layer in Keras 
- keras_unpooling.py: for performing conv unpooling operation for COnv autoencoder part of the network
- misc.py: contains the data preparation helper modules
- network.py: CDEC network creation for the clustering
- main.py: the main class that encapsulates all the steps. 

#### Instruction

#### Acknowledgement: 
This implementation slightly based on https://github.com/elieJalbout/Clustering-with-Deep-learning

### DEC implementation in Python
A modified version of Keras based DEC implementation (https://github.com/XifengGuo/DEC-keras) proposed by Ali F. et al. is used in our approach. Network training were carried out on a Nvidia TitanX GPU with CUDA and cuDNN enabled to make the overall pipeline faster. 

#### Step 1: Feature extraction using Scala, Adam and Spark 
For this, first, download the VCF files (containing the variants) and the panel file (containing the labels) from ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/. 
 
Then go to https://github.com/rezacsedu/VariationDEC/tree/master/PopulationClustering_v2 and use the featureExtractor.scala
to extract the features and save as a DataFrame in CSV to be used by Keras-based DEC.

For this, make sure that you've configured Spark correctly on your machine. Alternatively, execute this script as a standalone Scala project from Eclipse or IntelliJ IDEA. 

#### Step 2: This is the DEC part in Keras/Python 
Go to https://github.com/rezacsedu/VariationDEC/tree/master/DEC_GenotypeClustering_Keras. Then there are 2 Python scripts and a sample genetic variants feature in csv for the clustering and classification respectively. 

- genome.csv: is the sample genetic variants featres
- DEC_Genotype_Clustering.py: for the clustering 
- LSTM_EthnicityPrediction.py: for the classification 

### Spark and H2O implementation in Scala
For this, first download the VCF files (containing the variants) and the panel file (containing the labels) from ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/. Then go to https://github.com/rezacsedu/VariationDEC/tree/master/PopulationClustering_v2 and you'll see there Scala scripts as listed below: 

- PopGenomicsClassificationSpark.scala: this is the Spark implementation of ethnicity prediction
- PopStratClassification.scala: this is the H2O implementation of ethnicity prediction
- PopStratClustering.scala: this is the H2O/Spark implementation of the genotype clustering but using K-means prediction

For this, make sure that you've configured Spark and Adam (see https://github.com/bigdatagenomics/adam) correctly on your machine. Alternatively, execute this script as a standalone Scala project from Eclipse or IntelliJ IDEA.

### Citation request
    @inproceedings{karim2018recurrent,
        title={Recurrent Deep Embedding Networks for Genotype Clustering and Ethnicity Prediction},
        author={Karim, Md and Cochez, Michael and Beyan, Oya Deniz and Zappa, Achille and Sahay, Ratnesh and Decker, Stefan and Schuhmann, Dietrich-Rebholz and others},
        booktitle={arXiv preprint arXiv:1805.12218},
        year={2018}
    }

### Contributing
For any questions, feel free to open an issue or contact at rezaul.karim@rwth-aachen.de
