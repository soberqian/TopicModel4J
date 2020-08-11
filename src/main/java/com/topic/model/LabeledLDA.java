package com.topic.model;
/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *	  You should have received a copy of the GNU General Public License along with this 
 *	  program.
 */
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.topic.utils.FileUtil;
import com.topic.utils.FuncUtils;

/**
 * TopicModel4J: A Java package for topic models
 * 
 * Gibbs sampling for Labeled LDA
 * 
 * Reference:
 * Ramage D, Hall D, Nallapati R, et al. Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora[C]//Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing: Volume 1-Volume 1. Association for Computational Linguistics, 2009: 248-256.
 * 
 * Each document is associated with a set of labels
 * Each label with one topic in direct correspondence
 * 
 * @author: Yang Qian,Yuanchun Jian,Yidong Chai,Yezheng Liu,Jianshan Sun (HeFei University of Technology)
 * 
 */
public class LabeledLDA {
	public double alpha; // Hyper-parameter alpha
	public double beta; // Hyper-parameter beta
	public int iterations; // number of Gibbs sampling iterations
	public int topWords; // number of most probable words for each topic
	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int M; // number of documents in the corpus
	public int V; // number of words in the corpus
	public int [][] docword;//word index array
	// Labeled LDA related
	public Map<String, Integer> labelToIndexMap = new HashMap<String, Integer>(); //label to index
	public List<String> indexLabelMap = new ArrayList<String>();   //index to String label 
	public int [][] doclabel;//label index array
	public int L; // number of labels in the corpus---Topic Number
	public int[][] ndk; // document-topic count
	public int[] ndsum; //document-topic sum
	public int[][] nkw; //topic-word count
	public int[] nksum; //topic-word sum (total number of words assigned to a topic)
	public int[][] z;
	//output
	public int topWordsOutputNumber;
	public String outputFileDirectory; 
	public LabeledLDA(String inputFile, String inputFileCode, String separator,
			double inputAlpha, double inputBeta, int inputIterations, int inTopWords,
			String outputFileDir){
		//read data
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines(inputFile, docLines,inputFileCode);
		M = docLines.size();
		docword = new int[M][];
		doclabel = new int[M][];
		int j = 0;
		for(String line : docLines){
			List<String> words = new ArrayList<String>();
			List<String> links = new ArrayList<String>();
			FileUtil.tokenizeAndLowerCase(line.split("\t")[1], words);
			FileUtil.tokenizeEntity(line.split("\t")[0], links,separator);
			docword[j] = new int[words.size()];
			for(int i = 0; i < words.size(); i++){
				String word = words.get(i);
				if(!wordToIndexMap.containsKey(word)){
					int newIndex = wordToIndexMap.size();
					wordToIndexMap.put(word, newIndex);
					indexToWordMap.add(word);
					docword[j][i] = newIndex;
				} else {
					docword[j][i] = wordToIndexMap.get(word);
				}
			}
			doclabel[j] = new int[links.size()];
			for(int i = 0; i < links.size(); i++){
				String link = links.get(i);
				if(!labelToIndexMap.containsKey(link)){
					int newIndex = labelToIndexMap.size();
					 labelToIndexMap.put(link, newIndex);
					indexLabelMap.add(link);
					doclabel[j][i] = newIndex;
				} else {
					doclabel[j][i] = labelToIndexMap.get(link);
				}
			}
			j++;

		}
		V = indexToWordMap.size();
		L = indexLabelMap.size();
		alpha = inputAlpha;
		beta = inputBeta;
		iterations = inputIterations;
		topWordsOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		initialize();
	}
	
	/**
	 * Randomly initialize topic assignments
	 */
	/**
	 * Randomly initialize topic assignments
	 */
	public void initialize(){
		ndk = new int[M][L];
		ndsum = new int[M];
		nkw = new int[L][V];
		nksum = new int[L];
		z = new int[M][];
		for (int d = 0; d < M; d++) {
			int Nd = docword[d].length;  // the number of words in a document
			z[d] = new int[Nd];
			int[] dLabels = doclabel[d];
			for (int n = 0; n < Nd; n++) {
				int topic;
				if (dLabels.length > 0) {
					topic = dLabels[(int) (Math.random() * dLabels.length)];
                    z[d][n] = topic;
                } else {
                	topic = (int) (Math.random() * L);
                    z[d][n] = topic;
                }
				updateCount(d, topic, docword[d][n], +1);
			}
		}
	}
	public void MCMCSampling(){
		for (int iter = 1; iter <= iterations; iter++) {
			System.out.println("iteration : " + iter);
			gibbsOneIteration();
		}
		// output the result
		System.out.println("write topic word ..." );
		writeTopWordsWithProbability();
		System.out.println("write document topic ..." );
		writeDocumentTopic();
		
	}
	public void gibbsOneIteration() {
		for (int d = 0; d < M; d++) {
			for (int n = 0; n < z[d].length; n++) {
				int topic = z[d][n]; // get the old topic
				updateCount(d, topic, docword[d][n], -1); // update the count --1
				// possible number of topics for document d
		        int K_d = (doclabel[d] != null) ?  doclabel[d].length : L;
		        if (K_d == doclabel[d].length) {
		        	double[] p = new double[K_d];
					for (int k = 0; k < K_d; k++) {
						p[k] = (ndk[d][k] + alpha) / (ndsum[d] + K_d * alpha) * (nkw[k][docword[d][n]] + beta)
								/ (nksum[k] + V * beta);
					}
					int index = FuncUtils.rouletteGambling(p); //roulette gambling for updating the topic of a word
					topic = doclabel[d][index];
					z[d][n] = topic;
					updateCount(d, topic, docword[d][n], +1);  // update the count ++1
				}else {
					double[] p = new double[K_d];
					for (int k = 0; k < K_d; k++) {
						p[k] = (ndk[d][k] + alpha) / (ndsum[d] + K_d * alpha) * (nkw[k][docword[d][n]] + beta)
								/ (nksum[k] + V * beta);
					}
					int index = FuncUtils.rouletteGambling(p); //roulette gambling for updating the topic of a word
					topic = index;
					z[d][n] = topic;
					updateCount(d, topic, docword[d][n], +1);  // update the count ++1
				}
			}
		}
	}
	/**
	 * update the count 
	 * 
	 * @param probs ndk,ndsum,nkw,nksum
	 * @return
	 */
	void updateCount(int d, int topic, int word, int flag) {
		ndk[d][topic] += flag;
		ndsum[d] += flag;
		nkw[topic][word] += flag;
		nksum[topic] += flag;
	}
	/**
	 * obtain the parameter Theta
	 */
	public double[][] estimateTheta() {
		double[][] theta = new double[M][L];
		for (int d = 0; d < M; d++) {
			for (int k = 0; k < L; k++) {
				theta[d][k] = (ndk[d][k] + alpha) / (ndsum[d] + L * alpha);
			}
		}
		return theta;
	}
	/**
	 * obtain the parameter Phi
	 */
	public double[][] estimatePhi() {
		double[][] phi = new double[L][V];
		for (int k = 0; k < L; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nkw[k][w] + beta) / (nksum[k] + V * beta);
			}
		}
		return phi;
	}
	/**
	 * write top words with probability for each topic
	 */
	public void writeTopWordsWithProbability(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] phi = estimatePhi();
		int topicNumber = 1;
		for (double[] phi_z : phi) {
			sBuilder.append("Topic:" + topicNumber + "(" + indexLabelMap.get(topicNumber-1) + ")" + "\n" );
			for (int i = 0; i < topWordsOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_z);
				sBuilder.append(indexToWordMap.get(max_index) + " :" + phi_z[max_index] + "\n");
				phi_z[max_index] = 0;
			}
			sBuilder.append("\n");
			topicNumber++;
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "LabeledLDA_topic_word_" + L + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write theta for each document
	 */
	public void writeDocumentTopic(){
		double[][] theta = estimateTheta();
		StringBuilder sBuilder = new StringBuilder();
		for (int d = 0; d < theta.length; d++) {
			StringBuilder doc = new StringBuilder();
			for (int k = 0; k < theta[d].length; k++) {
				doc.append(theta[d][k] + "\t");
			}
			sBuilder.append(doc.toString().trim() + "\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "LabeledLDA_doc_topic" + L + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String[] args) {
		LabeledLDA llda = new LabeledLDA("data/rawdata_process_author", "gbk", ",", 0.1, 
				0.01, 500, 50, "data/ldaoutput/");
		llda.MCMCSampling();
	}
}
