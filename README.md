# Software Introduction
This Java package corresponds to my research paper as follows:<br />
**Yang Qian, Junming Yin, Yezheng Liu, Yuanchun Jiang. TopicModel4J: A Java Package for Topic Models, to be submitted.**<br />
<br />
This package is about Topic Models for Natural Language Processing (NLP). And **it provides an easy-to-use interface for researchers and data analysts**.<br />

Motivations：I develop this Java package to promote related research about Topic Models for Natural Language Processing (NLP). <br />

When submitting my research paper to a journal, I will publicly release all the source code.<br />

# Jar Dependency
If you want to use this package, you need download some Java jars: commons-math3-3.5.jar, lingpipe-4.1.0.jar, stanford-corenlp-3.9.1-models.jar, stanford-corenlp-3.9.1-sources.jar, stanford-corenlp-3.9.1.jar. The stanford-corenlp 3.9.1 can be download from this website: http://central.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.9.1/.

# Data Preprocessing for NLP
This software can do the following text preprocessing:<br />
* (1) Split the sentence to words.<br />
* (2) Lowercase the words and preform lemmatization.<br />
* (3) Remove useless characters, URLs and stop words.<br />

The first example is as follows:<br />
```java
import java.util.ArrayList;
import com.topic.utils.FileUtil;
public class RawDataProcess {
	/**
	 * Functions:
	 * 
	 * (1) Split the sentence to words
	 * (2) Lowercase the words and preform lemmatization
	 * (3) Remove special characters (e.g., #, % and &), URLs and stop words
	 * 
	 * @author: Yang Qian
	 */
	public static void main(String[] args) {
			String line = "http://t.cn/RAPgR4n Artificial intelligence is a known phenomenons "
					+ "in the world today. Its root started to build years "
					+ "ago but the tree started to grow long after. Months ago when our beloved google assistant made her first "
					+ "call to book a haircut appointment in the Google IO event,";
			//get all word for a document
			ArrayList<String> words = new ArrayList<String>();
			//lemmatization using StanfordCoreNLP
			FileUtil.getlema(line, words);
			//remove noise words
			String text = FileUtil.RemoveNoiseWord(words);
			System.out.println(text);
	}
}
```
Running this code, we can obtain the following results:<br />
```java
artificial intelligence phenomenon world today root start build year ago tree start grow long month ago beloved google assistant make call book haircut appointment Google IO event
```
If we want deal a file which a line represent one document. For example,
```java
We present a new algorithm for domain adaptation improving upon a discrepancy minimization algorithm, (DM), previously shown to outperform a number of algorithms for this problem. 
We investigated the feature map inside deep neural networks (DNNs) by tracking the transport map. We are interested in the role of depth--why do DNNs perform better than shallow models?
```
We denote this file as 'rawdata'. And we can use the next code to deal with:
```java
import java.io.IOException;
import java.util.ArrayList;
import com.topic.utils.FileUtil;

public class RawDataProcessing {
	/**
	 * Functions:
	 * 
	 * (1) Split the sentence to words
	 * (2) Lowercase the words and preform lemmatization
	 * (3) Remove special characters (e.g., #, % and &), URLs and stop words
	 * 
	 * @author: Yang Qian
	 */
	public static void main(String[] args) throws IOException {
		//read data
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines("data/rawdata", docLines, "gbk");
		ArrayList<String> doclinesAfter = new ArrayList<String>();
		for(String line : docLines){
			//get all word for a document
			ArrayList<String> words = new ArrayList<String>();
			//lemmatization using StanfordCoreNLP
			FileUtil.getlema(line, words);
			//remove noise words
			String text = FileUtil.RemoveNoiseWord(words);
			doclinesAfter.add(text);
		}
		// write data
		FileUtil.writeLines("data/rawdata_process", doclinesAfter, "gbk");
	}
}
```

# Algorithm for NLP
The algorithms in this package contain **Latent Dirichlet Allocation (LDA), Biterm Topic Model (BTM),  Author-topic Model (ATM), Dirichlet Multinomial Mixture Model (DMM), Dual-Sparse Topic Model (DSTM), Labeled LDA, Link LDA, Sentence-LDA, Pseudo-document-based Topic Model (PTM), Hierarchical Dirichlet processes, Collaborative topic Model (CTM), Gaussian Lda and so on **. Now, I will intorduce how to use my package for running some algorithms.

## Latent Dirichlet Allocation (Collapsed Gibbs sampling)
Reference: (1) Griffiths T. Gibbs sampling in the generative model of latent dirichlet allocation[J]. 2002.<br />
           (2) Heinrich G. Parameter estimation for text analysis[R]. Technical report, 2005.<br />
The following code is to call the LDA algorithm for processing text:<br />
```java
import com.topic.model.GibbsSamplingLDA;

public class LDAGibbsSamplingTest {

	public static void main(String[] args) {
		GibbsSamplingLDA lda = new GibbsSamplingLDA("data/rawdata_process_lda", "gbk", 30, 0.1,
				0.01, 500, 50, "data/ldaoutput/");
		lda.MCMCSampling();

	}
}
```
Where the constructor method GibbsSamplingLDA() is:
```java
GibbsSamplingLDA(String inputFile, String inputFileCode, int topicNumber,
			double inputAlpha, double inputBeta, int inputIterations, int inTopWords,
			String outputFileDir)
```



##  Latent Dirichlet Allocation (Collapsed Variational Bayesian Inference)










