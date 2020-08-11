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
 * Collapsed Gibbs Sampling for DPMM
 * 
 * This model can be used to cluster short text.
 * 
 * Reference:
 * Yin J, Wang J. A model-based approach for text clustering with outlier detection[C]//2016 IEEE 32nd International Conference on Data Engineering (ICDE). IEEE, 2016: 625-636.
 * https://github.com/junyachen/GSDPMM
 * @author: Yang Qian,Yuanchun Jian,Yidong Chai,Yezheng Liu,Jianshan Sun (HeFei University of Technology)
 */
public class DPMM {
	public double alpha; // Hyper-parameter alpha
	public double beta; // Hyper-parameter beta
	public int K; // number of clusters
	public int iterations; // number of Gibbs sampling iterations
	public int topWords; // number of most probable words for each cluster
	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int [][] docword;//word index array
	public int M; // number of documents in the corpus
	public int V; // number of words in the corpus
	//DPMM
	public int[] z; //cluster assignment for each document
	public int[] kd; //number of documents assigned to a cluster
	public int[][] nkw; //cluster-word count
	public int[] nksum; //cluster-word sum (total number of words assigned to a cluster)
	//Example: given a document of "a a b a b c d c". We have: 1 2 1 3 2 1 1 2
	public List<List<Integer>> docWordIndexCount;  
	//output
	public int topWordsOutputNumber;
	public String outputFileDirectory; 
	public DPMM(String inputFile, String inputFileCode, int initClusterNumber,
			double inputAlpha, double inputBeta, int inputIterations, int inTopWords,
			String outputFileDir){
		//read data
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines(inputFile, docLines,inputFileCode);
		M = docLines.size();
		docword = new int[M][];
		docWordIndexCount = new ArrayList<List<Integer>>();
		int j = 0;
		for(String line : docLines){
			List<String> words = new ArrayList<String>();
			FileUtil.tokenizeAndLowerCase(line, words);
			docword[j] = new int[words.size()];
			HashMap<String, Integer> wordCount = new HashMap<String, Integer>();
			List<Integer> wordIndexCount = new ArrayList<Integer>();
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
				int times = 0;
				if (wordCount.containsKey(word)) {
					times = wordCount.get(word);
				}
				times += 1;
				wordCount.put(word, times);
				wordIndexCount.add(times);
			}
			docWordIndexCount.add(wordIndexCount);
			j++;
		}
		V = indexToWordMap.size();
		alpha = inputAlpha;
		beta = inputBeta;
		K = initClusterNumber;  //initialize topic number
		iterations = inputIterations;
		topWordsOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		initialize();
	}
	/**
	 * Randomly initialize topic assignments
	 */
	public void initialize(){
		z = new int[M];
		nkw = new int[K][V];
		nksum = new int[K];
		kd = new int[K];
		for (int d = 0; d < M; d++) {
			int cluster = (int) (Math.random() * K);
			z[d] = cluster; 
			updateCount(d,cluster,false);
		}
	}
	public void MCMCSampling(){
		for (int iter = 1; iter <= iterations; iter++) {
			System.out.println("iteration : " + iter);
			gibbsOneIteration();
			defragment();
			System.out.println("K is:\t" + K);
		}
		// output the result
		System.out.println("write cluster word ..." );
		writeTopWordsWithProbability();
		System.out.println("write cluster distribution ..." );
		writeClusterDistri();
		System.out.println("write cluster for each document ..." );
		writeDocCluster();

	}
	public void gibbsOneIteration() {
		for (int d = 0; d < M; d++) {
			double[] p = new double[K + 1];
			int cluster = z[d];
			//decrease the count
			updateCount(d,cluster,true);
			if (kd[cluster] < 0) {
				defragment();
			}
			//sample the new cluster
			for (int k = 0; k < K; k++) {
				p[k] = (kd[k])/(M - 1 + alpha );
				for (int n = 0; n < docword[d].length; n ++) {
					p[k] *= (nkw[k][docword[d][n]] + beta + docWordIndexCount.get(d).get(n) - 1)
							/ (nksum[k] + V * beta + n);
				}
			}
			p[K] = (alpha)/(M - 1 +  alpha);
			for (int n = 0; n < docword[d].length; n ++) {
				p[K] *= (beta + docWordIndexCount.get(d).get(n) - 1)
						/ (V * beta + n);
			}
			cluster = FuncUtils.rouletteGambling(p);
			//increase the count
			z[d] = cluster;
			if (cluster < K) { //old cluster
				updateCount(d,cluster,false);
			}else {  //new cluster
				K++;
				kd = ensureCapacity(kd,1);
				nkw = ensureCapacity(nkw, 1, 0);
				nksum = ensureCapacity(nksum, 1);
				updateCount(d,cluster,false);
			}
		}
	}
	//remove the topic without words
	public void defragment() {
		int[] kOldToKNew = new int[K];
		int newK = 0;
		for (int k = 0; k < K; k++) {
			if (kd[k] > 0) {
				kOldToKNew[k] = newK;  //
				swap(kd, newK, k);
				swap(nkw, newK, k);
				swap(nksum, newK, k);
				newK++;
			} else {

			}
		}
		K = newK;
		for (int dIndex = 0; dIndex < K; dIndex++) {
			z[dIndex] = kOldToKNew[z[dIndex]];
		}
	}
	//renumber the arr
	public void swap(int[] arr, int arg1, int arg2){
		int t = arr[arg1]; 
		arr[arg1] = arr[arg2]; 
		arr[arg2] = t; 
	}
	//renumber the arr
	public  void swap(int[][] arr, int arg1, int arg2) {
		int[] t = arr[arg1]; 
		arr[arg1] = arr[arg2]; 
		arr[arg2] = t; 
	}
	//when creating a new cluster, we will ensure capacity of the arr
	public  int[] ensureCapacity(int[] arr,int i) {
		int length = arr.length;
		int[] arr2 = new int[length  + i];
		System.arraycopy(arr, 0, arr2, 0, length);
		return arr2;
	}
	//when creating a new cluster, we will ensure capacity of the arr
	public  int[][] ensureCapacity(int[][] array,int i,int j) {  
		int[][] arr = new int[array.length +i][array[0].length +j];       //��չ  
		for(int c = 0; c< array.length; c++) {  
			System.arraycopy(array[c], 0, arr[c], 0, array[c].length);  //���鿽��  
		}  
		return arr;  
	} 
	/**
	 * update the count 
	 * 
	 * @param probs kd,nkw,nksum
	 * @return
	 */
	void updateCount(int d, int cluster, boolean flag) {
		if (flag) {  // decrease the count
			kd[cluster]--;
			for(int n = 0; n < docword[d].length; n ++) {
				nkw[cluster][docword[d][n]]--;
				nksum[cluster]--;
			}
		}else { // increase the count
			kd[cluster]++;
			for(int n = 0; n < docword[d].length; n ++) {
				nkw[cluster][docword[d][n]]++;
				nksum[cluster]++;
			}
		}
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
	 * obtain the parameter theta
	 */
	public double[] estimateTheta() {
		double[] theta = new double[K];
		for (int k = 0; k < K; k++) {
			theta[k] = (kd[k] + alpha) / (M - 1 + K * alpha);
		}
		return theta;
	}
	/**
	 * write top words with probability for each cluster
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
			FileUtil.writeFile(outputFileDirectory + "DPMM_cluster_word_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write theta for each cluster
	 */
	public void writeClusterDistri(){
		double[] theta = estimateTheta();
		StringBuilder sBuilder = new StringBuilder();
		for (int k = 0; k < K; k++) {
			sBuilder.append(theta[k] + "\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "DPMM_theta_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write cluster for each document
	 */
	public void writeDocCluster(){
		StringBuilder sBuilder = new StringBuilder();
		for(int d = 0; d < M; d++){
			int cluster = z[d];
			sBuilder.append(cluster + "\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "DPMM_doc_cluster" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String[] args) {
		DPMM dmm = new DPMM("data/shortdoc.txt", "gbk", 5, 0.1,
				0.01, 1500, 50, "data/ldaoutput/");
		dmm.MCMCSampling();
	}

}
