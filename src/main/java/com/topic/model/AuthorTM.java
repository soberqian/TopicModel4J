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
 * Collapsed Gibbs sampling for author-topic model
 * 
 * Reference:
 * Rosen-Zvi M, Griffiths T, Steyvers M, et al. The author-topic model for authors and documents[C]//Proceedings of the 20th conference on Uncertainty in artificial intelligence. AUAI Press, 2004: 487-494.
 * Python: https://github.com/jyscardioid/neo-author-topic-model
 * @author: Yang Qian,Yuanchun Jian,Yidong Chai,Yezheng Liu,Jianshan Sun (HeFei University of Technology)
 */

public class AuthorTM
{
	public double alpha; // Hyper-parameter alpha
	public double beta; // Hyper-parameter beta
	public int K; // number of topics
	public int iterations; // number of Gibbs sampling iterations
	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int [][] docword;//word index array
	public int M; // number of documents in the corpus
	public int V; // number of words in the corpus
	//author-topic model related
	public Map<String, Integer> AuthorToIndexMap = new HashMap<String, Integer>();;  //author to index
	public List<String> indexAuthorMap = new ArrayList<String>();    //index to author
	public int [][] docAuthor; // authors of documents
	public int A;  //number of authors in the corpus
	public int[][] nak; //the number of times author a is assigned to topic k
	public int[] naksum; //total authors is assigned to topic k
	public int[][] nkw; //topic-word count
	public int[] nksum; //topic-word sum (total number of words assigned to a topic)
	public int[][] z;  //topic assignment
	public int[][] x;  //author assignment
	//output
	public int topWordsOutputNumber;
	public String outputFileDirectory; 
	public AuthorTM(String inputFile, String inputFileCode, String separator, int topicNumber,
			double inputAlpha, double inputBeta, int inputIterations, int inTopWords,
			String outputFileDir){
		//read data
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines(inputFile, docLines,inputFileCode);
		M = docLines.size();
		docword = new int[M][];
		docAuthor = new int[M][];
		int j = 0;
		for(String line : docLines){
			List<String> words = new ArrayList<String>();
			List<String> authors = new ArrayList<String>();
			FileUtil.tokenizeAndLowerCase(line.split("\t")[1], words);
			FileUtil.tokenizeEntity(line.split("\t")[0], authors,separator);
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
			docAuthor[j] = new int[authors.size()];
			for(int i = 0; i < authors.size(); i++){
				String author = authors.get(i);
				if(!AuthorToIndexMap.containsKey(author)){
					int newIndex = AuthorToIndexMap.size();
					AuthorToIndexMap.put(author, newIndex);
					indexAuthorMap.add(author);
					docAuthor[j][i] = newIndex;
				} else {
					docAuthor[j][i] = AuthorToIndexMap.get(author);
				}
			}
			j++;

		}
		alpha = inputAlpha;
		beta = inputBeta;
		V = indexToWordMap.size();
		A = indexAuthorMap.size();
		K = topicNumber;
		M = docword.length;
		iterations = inputIterations;
		topWordsOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		initialize();
	}
	/**
	 * Randomly initialize topic and author assignments
	 */
	public void initialize(){
		nkw = new int[K][V];
		nksum = new int[K];
		nak = new int[A][K];
		naksum = new int[A];
		z = new int[M][];
		x = new int[M][];
		for (int d = 0; d < M; d++) {
			int NWords = docword[d].length;  // the number of words in a document
			z[d] = new int[NWords];
			x[d] = new int[NWords];
			for (int w = 0; w < NWords; w++) {
				int topic = (int) (Math.random() * K);
				int author = (int) (Math.random() * A);
				z[d][w] = topic;
				x[d][w] = author;
				updateCount(topic, docword[d][w], +1, true);
				updateCount(author,topic, +1, false);
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
		writeTopWords();
		System.out.println("write author topic ..." );
		writeAuthorTopic();
		System.out.println("write ranked topic author ..." );
		writeTopicAuthor();
	}
	public void gibbsOneIteration() {
		for (int d = 0; d < docword.length; d++) {
			for (int n = 0; n < z[d].length; n++) {
				int topic = z[d][n]; // get the old topic
				int author = x[d][n];  //get the old author
				updateCount(topic, docword[d][n], -1, true); // update the count --1
				updateCount(author, topic, -1, false); // update the count --1
				double[][] p = new double[K][A];
				for (int k = 0; k < K; k++) { //��С�ڴ�
					for (int a = 0; a < A; a++) {
						p[k][a] = (nkw[k][docword[d][n]] + beta)*( nak[a][k] + alpha) /(nksum[k] + V * beta)/ (naksum[a] + K * alpha);
					}
				}
				int idx = FuncUtils.rouletteGambling(p); //roulette gambling for updating z and x
				topic = idx % K;
				author = idx / K;
				updateCount(topic, docword[d][n], +1, true);  // update the count ++1
				updateCount(author, topic, +1, false); 
				z[d][n] = topic;
				x[d][n] = author;
			}
		}
	}
	/**
	 * update the count 
	 * 
	 * @param nkw, nksum , nak, naksum
	 * @return
	 */
	void updateCount(int topicOrAuthor, int wordOrTopic, int flag, boolean zOrX) {
		if (zOrX == true) { //update z related
			nkw[topicOrAuthor][wordOrTopic] += flag;
			nksum[topicOrAuthor] += flag;
		}else {
			nak[topicOrAuthor][wordOrTopic] += flag;
			naksum[topicOrAuthor] += flag;
		}
	}
	/**
	 * obtain the parameter Theta
	 */
	public double[][] estimateTheta() {
		double[][] theta = new double[A][K];
		for (int a = 0; a < A; a++) {
			for (int k = 0; k < K; k++) {
				theta[a][k] = (nak[a][k] + alpha) / (naksum[a] + K * alpha);
			}
		}
		return theta;
	}
	/**
	 * obtain the parameter Phi
	 */
	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
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
			sBuilder.append("Topic:" + topicNumber + "\n");
			for (int i = 0; i < topWordsOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_z);
				sBuilder.append(indexToWordMap.get(max_index) + " :" + phi_z[max_index] + "\n");
				phi_z[max_index] = 0;
			}
			sBuilder.append("\n");
			topicNumber++;
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "authorTM_topic_word" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write top words with probability for each topic
	 */
	public void writeTopWords(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] phi = estimatePhi();
		for (double[] phi_z : phi) {
			for (int i = 0; i < topWordsOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_z);
				sBuilder.append(indexToWordMap.get(max_index) + "\t");
				phi_z[max_index] = 0;
			}
			sBuilder.append("\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "authorTM_topic_wordnop_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write theta
	 */
	public void writeAuthorTopic(){
		double[][] theta = estimateTheta();
		StringBuilder sBuilder = new StringBuilder();
		for (int a = 0; a < theta.length; a++) {
			StringBuilder doc = new StringBuilder();
			doc.append(indexAuthorMap.get(a) + "\t");
			for (int k = 0; k < theta[a].length; k++) {
				doc.append(theta[a][k] + "\t");
			}
			sBuilder.append(doc.toString().trim() + "\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "authorTM_author_topic_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * rank topic author 
	 */
	public void writeTopicAuthor(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] thetaTrans = FuncUtils.arrayTrans(estimateTheta());
		int topicNumber = 1;
		for (double[] theta_z : thetaTrans) {
			sBuilder.append("Topic:" + topicNumber + "\n");
			for (int i = 0; i < topWordsOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(theta_z);
				sBuilder.append(indexAuthorMap.get(max_index) + " :" + theta_z[max_index] + "\n");
				theta_z[max_index] = 0;
			}
			sBuilder.append("\n");
			topicNumber++;
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "authorTM_topic_author_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String args[]) throws Exception{
		AuthorTM lda = new AuthorTM("/home/qianyang/dualsparse/rawdata_process_author", "gbk", ",", 25, 0.1,
				0.01, 500, 50, "/home/qianyang/dualsparse/output/");
		lda.MCMCSampling();
	}
}
