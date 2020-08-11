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
import java.util.Random;
import com.topic.utils.FileUtil;
import com.topic.utils.FuncUtils;

/**
 * TopicModel4J: A Java package for topic models
 * 
 * Sampling based on the Chinese restaurant franchise
 * 
 * Reference:
 * Teh Y W, Jordan M I, Beal M J, et al. Sharing clusters among related groups: Hierarchical Dirichlet processes[C]//Advances in neural information processing systems. 2005: 1385-1392.
 * https://github.com/arnim/HDP
 * 
 * @author: Yang Qian,Yuanchun Jian,Yidong Chai,Yezheng Liu,Jianshan Sun (HeFei University of Technology)
 */
public class HDP {
	public double gamma; // Hyper-parameter gamma
	public double beta; // Hyper-parameter beta
	public double alpha; // Hyper-parameter alpha
	public int K; // number of topics
	public int iterations; // number of Gibbs sampling iterations
	public int topWords; // number of most probable words for each topic
	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int [][] docword;//word index array
	public int M; // number of documents in the corpus
	public int V; // number of words in the corpus
	//HDP related
	public int ndtable[]; //total number of tables in document d
	public int ndw_table[][]; //the d_w table assignment 
	public int ndtable_wordCount[][]; //the total number of words that assignment to the table t in the document d
	public int d_tableToTopic[][]; //the topic assignment for table t in document d
	public int[] nk_table; //total numbers of table to a topic
	public int[][] nkw; //topic-word count
	public int[] nksum; //topic-word sum (total number of words assigned to a topic)
	public int totalTablesNum; 
	public double[] f;
	public double[] p;
	//output
	public int topWordsOutputNumber;
	public String outputFileDirectory; 
	public HDP(String inputFile, String inputFileCode, int initTopicNumber,
			double inputAlpha, double inputBeta, double inputGamma, int inputIterations, int inTopWords,
			String outputFileDir){
		//read data
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines(inputFile, docLines,inputFileCode);
		M = docLines.size();
		docword = new int[M][];
		int j = 0;
		for(String line : docLines){
			List<String> words = new ArrayList<String>();
			FileUtil.tokenizeAndLowerCase(line, words);
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
			j++;

		}
		V = indexToWordMap.size();
		alpha = inputAlpha;
		beta = inputBeta;
		gamma = inputGamma;
		K = initTopicNumber;
		iterations = inputIterations;
		topWordsOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		initialize();
	}
	/**
	 * Randomly initialize topic assignments
	 */
	public void initialize(){
		ndw_table = new int[M][V];
		ndtable_wordCount = new int[M][2];//table number is 2
		nk_table = new int[K + 1];
		ndtable = new int[M];
		nkw = new int[K + 1][V];
		nksum = new  int[K + 1];
		d_tableToTopic = new int[M][2]; //table number is 2
		f = new double[K * 2];
		p = new double[K * 2];
		//To ensure that each topic is assigned words
		for (int d = 0; d < K; d++) { 
			for (int n = 0; n < docword[d].length; n++)
				updateCount(d, n, 0, d); 
		} // all topics have now one document
		for (int d = K; d < docword.length; d++) {
			int k = (int) (Math.random() * K);
			for (int n = 0; n < docword[d].length; n++) 
				updateCount(d, n, 0, k);
		} // the words in the remaining documents are now assigned too
	}

	public void MCMCSampling(){
		for (int iter = 1; iter <= iterations; iter++) {
			System.out.println("iteration : " + iter + "\tK:" + K);
			gibbsOneIteration();
			defragment();
		}
		// output the result
		System.out.println("write topic word ..." );
		writeTopWordsWithProbability();
		System.out.println("write document topic ..." );
		writeDocumentTopic();
	}
	public void gibbsOneIteration() {
		for (int d = 0; d < M; d++) {
			for (int n = 0; n < docword[d].length; n++) {
				updateCountDecrease(d, n); // update the count --1
				//sample table and topic
				int table = sampleTable(d, n);
				if (table == ndtable[d]){ // new Table
					updateCount(d, n, table, sampleTopic()); // sampling its Topic
				}else{
					updateCount(d, n, table, d_tableToTopic[d][table]); // existing Table
				}
			}
		}
	}
	/**
	 * sample a table for a word
	 * 
	 * new table or old table
	 * 
	 * if new table, choose a topic for this table
	 */
	int sampleTable(int d, int n) {
		double pSum = 0.0;
		f = ensureCapacity(f, K);
		p = ensureCapacity(p, ndtable[d]);
		double fNew = gamma / V;
		for (int k = 0; k < K; k++) {
			f[k] = (nkw[k][docword[d][n]] + beta) / 
					(nksum[k] + V*beta);
			fNew += nk_table[k] * f[k];
		}
		for (int tab = 0; tab < ndtable[d]; tab++) {
			if (ndtable_wordCount[d][tab] > 0) 
				pSum += ndtable_wordCount[d][tab] * f[d_tableToTopic[d][tab]];
			p[tab] = pSum;
		}
		pSum += alpha * fNew / (totalTablesNum + gamma); // Probability for t = tNew
		p[ndtable[d]] = pSum;
		double u =  (new Random()).nextDouble() * pSum;
		int j;
		for (j = 0; j <= ndtable[d]; j++)
			if (u < p[j]) 
				break;	// which table assignment for the word
		return j;
	}
	/**
	 * 
	 * if new table, choose a topic for this table
	 * 
	 */
	private int sampleTopic() {
		double pSum = 0.0;
		p = ensureCapacity(p, K);
		for (int k = 0; k < K; k++) {
			pSum += nk_table[k] * f[k];
			p[k] = pSum;
		}
		pSum += gamma/ V;
		p[K] = pSum;
		double u = (new Random()).nextDouble() * pSum;
		int k;
		for (k = 0; k <= K; k++)
			if (u < p[k])
				break;
		return k;
	}
	/**
	 * 
	 * remove the empty topic
	 * 
	 */
	public void defragment() {
		int[] kOldToKNew = new int[K];
		int  newK = 0;
		for (int k = 0; k < K; k++) {
			if (nksum[k] > 0) {
				kOldToKNew[k] = newK;
				swap(nksum, newK, k);
				swap(nk_table, newK, k);
				swap(nkw, newK, k);
				newK++;
			} 
		}
		K = newK;
		for (int d = 0; d < docword.length; d++) 
			defragmentTable(d, kOldToKNew);
	}
	/**
	 * 
	 * remove the empty table
	 * 
	 */
	public void defragmentTable(int d, int[] kOldToKNew) {
		int[] tOldToTNew = new int[ndtable[d]];
		int[] tableWordCount = ndtable_wordCount[d];
		int newtablesNum = 0;
		for (int tab = 0; tab < ndtable[d]; tab++){
			if (ndtable_wordCount[d][tab] > 0){
				tOldToTNew[tab] = newtablesNum;
				d_tableToTopic[d][newtablesNum] = kOldToKNew[d_tableToTopic[d][tab]];
				swap(tableWordCount, newtablesNum, tab);
				newtablesNum ++;
			} else 
				d_tableToTopic[d][tab] = -1;
		}
		ndtable_wordCount[d] = tableWordCount;
		ndtable[d] = newtablesNum;
		for (int n = 0; n < docword[d].length; n++)
			ndw_table[d][n] = tOldToTNew[ndw_table[d][n]];
	}
	//renumber the arr
	void swap(int[] arr, int arg1, int arg2){
		int t = arr[arg1]; 
		arr[arg1] = arr[arg2]; 
		arr[arg2] = t; 
	}
	//renumber the arr
	void swap(int[][] arr, int arg1, int arg2) {
		int[] t = arr[arg1]; 
		arr[arg1] = arr[arg2]; 
		arr[arg2] = t; 
	}
	/**
	 * update the count 
	 * ndw_table;
	 * 
	 * 
	 * @param d	document id
	 * @param w the index of the word
	 * @param table the table to which the word is assigned to
	 * @param k the topic to which the word is assigned to
	 */
	public void updateCount(int d, int n, int table, int k) {
		ndw_table[d][n] = table;
		ndtable_wordCount[d][table]++;
		nksum[k]++;
		nkw[k][docword[d][n]] ++; 
		if (ndtable_wordCount[d][table] == 1) { // a new table is created
			ndtable[d]++; 
			d_tableToTopic[d][table] = k;
			totalTablesNum++;
			nk_table[k]++;
			d_tableToTopic[d] = ensureCapacity(d_tableToTopic[d],ndtable[d]);
			ndtable_wordCount[d] = ensureCapacity(ndtable_wordCount[d], ndtable[d]);
			if (k == K) { // a new topic is created
				K++; 
				nk_table = ensureCapacity(nk_table, K); 
				nkw = ensureCapacity(nkw, K);
				nksum = ensureCapacity(nksum, K);
			}
		}
	}
	/**
	 * update the count when removing the word w
	 * 
	 * @param d document id
	 * @param n word index
	 */
	public void updateCountDecrease(int d, int n){
		// get the old table and topic
		int table = ndw_table[d][n];
		int k = d_tableToTopic[d][table];
		ndtable_wordCount[d][table]--;
		nksum[k]--; 		
		nkw[k][docword[d][n]] --;
		if (ndtable_wordCount[d][table] == 0) { // table is removed
			totalTablesNum--; 
			nk_table[k]--; 
			//docState.tableToTopic[table] --; 
		}
	}

	public static  int[] ensureCapacity(int[] arr, int min) {
		int length = arr.length;
		if (min < length)
			return arr;
		int[] arr2 = new int[min*2];
		System.arraycopy(arr, 0, arr2, 0, length);
		return arr2;
	}
	public static double[] ensureCapacity(double[] arr, int min){
		int length = arr.length;
		if (min < length)
			return arr;
		double[] arr2 = new double[min*2];
		System.arraycopy(arr, 0, arr2, 0, length);
		return arr2;
	}
	public static  int[][] ensureCapacity(int[][] array, int min) { 
		int length = array.length;
		if (min < length)
			return array;
		int[][] arr = new int[2*min][array[0].length];
		for(int c = 0; c< array.length; c++) {  
			System.arraycopy(array[c], 0, arr[c], 0, array[c].length);  
		}  
		return arr;  
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
	 * obtain the parameter Theta
	 */
	public double[][] estimateTheta() {
		//compute ndk
		double[][] ndk = new double[M][K];
		for (int d = 0; d < M; d++) {
			for (int tab = 0; tab < ndtable[d]; tab++) {
				int k = d_tableToTopic[d][tab];
				int wordCountTable = ndtable_wordCount[d][tab];
				ndk[d][k] += wordCountTable;
			}
		}
		double[][] theta = new double[M][K];
		for (int d = 0; d < M; d++) {
			for (int k = 0; k < K; k++) {
				theta[d][k] = (ndk[d][k] + alpha) / (docword[d].length + K * alpha);
			}
		}
		return theta;
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
			FileUtil.writeFile(outputFileDirectory + "HDP_topic_word_" + K + ".txt", sBuilder.toString(),"gbk");
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
		for (int i = 0; i < K; i++) {
			sBuilder.append("Topic" + (i+1) + "\t");
		}
		for (int d = 0; d < theta.length; d++) {
			StringBuilder doc = new StringBuilder();
			for (int k = 0; k < theta[d].length; k++) {
				doc.append(theta[d][k] + "\t");
			}
			sBuilder.append(doc.toString().trim() + "\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "HDP_doc_topic" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String[] args) {
		HDP hdp = new HDP("data/rawdata_process_lda", "gbk", 10, 1, 0.01,
				0.1, 1000, 50, "data/ldaoutput/");
		hdp.MCMCSampling();
	}
}
