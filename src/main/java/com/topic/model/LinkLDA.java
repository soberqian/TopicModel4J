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
 * Collapsed Gibbs sampling in the generative model of Link LDA
 * 
 * Reference:
 * Erosheva E, Fienberg S, Lafferty J. Mixed-membership models of scientific publications[J]. Proceedings of the National Academy of Sciences, 2004, 101(suppl 1): 5220-5227.
 * Su S, Wang Y, Zhang Z, et al. Identifying and tracking topic-level influencers in the microblog streams[J]. Machine Learning, 2018, 107(3): 551-578.
 * (Probabilistic inference formula):Ling G, Lyu M R, King I. Ratings meet reviews, a combined approach to recommend[C]//Proceedings of the 8th ACM Conference on Recommender systems. ACM, 2014: 105-112.
 * @author: Yang Qian,Yuanchun Jian,Yidong Chai,Yezheng Liu,Jianshan Sun (HeFei University of Technology)
 */
public class LinkLDA {
	public double alpha; // Hyper-parameter alpha
	public double beta; // Hyper-parameter beta
	public double gamma; // Hyper-parameter gamma
	public int K; // number of topics
	public int iterations; // number of Gibbs sampling iterations
	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int [][] docword;//word index array
	public int M; // number of documents in the corpus
	public int V; // number of words in the corpus
	public int[][] ndk; // document-topic count
	public int[] ndsum; //document-topic sum
	public int[][] nkw; //topic-word count
	public int[] nksum_w; //topic-word sum (total number of words assigned to a topic)
	public int[][] z_word; //topic assignment for word
	//for link LDA
	public Map<String, Integer> linkToIndexMap = new HashMap<String, Integer>(); //link to index
	public List<String> indexLinkMap = new ArrayList<String>();   //index to String link 
	public int [][] doclink;//link index array
	public int L; // number of links in the corpus
	public int[][] z_link;  //topic assignment for link
	public int[][] nkl;  //topic-link count
	public int[] nksum_l; //topic-link sum 
	//output
	public int topWordsAndLinksOutputNumber;
	public String outputFileDirectory; 
	public String outputFilecode; 
	public LinkLDA(String inputFile, String inputFileCode, String separator, int topicNumber,
			double inputAlpha, double inputBeta,double inputGamma, int inputIterations, int inTopWords,
			String outputFileDir){
		//read data
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines(inputFile, docLines,inputFileCode);
		M = docLines.size();
		docword = new int[M][];
		doclink = new int[M][];
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
			doclink[j] = new int[links.size()];
			for(int i = 0; i < links.size(); i++){
				String link = links.get(i);
				if(!linkToIndexMap.containsKey(link)){
					int newIndex = linkToIndexMap.size();
					linkToIndexMap.put(link, newIndex);
					//�����ظ��Ĵ�����뼯��
					indexLinkMap.add(link);
					//���ĵ��е�ÿ�����ʽ��б��
					doclink[j][i] = newIndex;
				} else {
					//������ظ��ĵ��ʣ������ʵı�Ÿ�docWords
					doclink[j][i] = linkToIndexMap.get(link);
				}
			}
			j++;

		}
		V = indexToWordMap.size();
		L = indexLinkMap.size();
		alpha = inputAlpha;
		beta = inputBeta;
		gamma = inputGamma;
		K = topicNumber;
		iterations = inputIterations;
		topWordsAndLinksOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		outputFilecode = inputFileCode;
		initialize();
	}
	/**
	 * Randomly initialize topic assignments
	 */
	public void initialize(){
		int D = docword.length;
		ndk = new int[D][K];
		ndsum = new int[D];
		nkw = new int[K][V];
		nksum_w = new int[K];
		z_word = new int[D][];
		z_link = new int[D][];
		nkl = new int[K][L];
		nksum_l = new int[K];
		for (int d = 0; d < D; d++) {
			int NWord = docword[d].length;  // the number of words in a document
			z_word[d] = new int[NWord];
			for (int n = 0; n < NWord; n++) {
				int topic = (int) (Math.random() * K);
				z_word[d][n] = topic;
				updateCount(d, topic, docword[d][n], +1, 0);
			}
			int NLink = doclink[d].length;
			z_link[d] = new int[NLink];
			for (int n = 0; n < NLink; n++) {
				int topic = (int) (Math.random() * K);
				z_link[d][n] = topic;
				updateCount(d, topic, doclink[d][n], +1, 1);
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
//		writeTopWords();
		System.out.println("write document topic ..." );
		writeDocumentTopic();
		System.out.println("write topic link ...");
		writeTopLinksWithProbability();
//		writeTopLinks();
	}
	public void gibbsOneIteration() {
		for (int d = 0; d < docword.length; d++) {
			//for words of this document
			for (int n = 0; n < z_word[d].length; n++) {
				int topic = z_word[d][n]; // get the old topic
				updateCount(d, topic, docword[d][n], -1, 0); // update the count --1
				double[] p = new double[K];
				for (int k = 0; k < K; k++) {
					p[k] = (ndk[d][k] + alpha) / (ndsum[d] + K * alpha) * (nkw[k][docword[d][n]] + beta)
							/ (nksum_w[k] + V * beta);
				}
				topic = FuncUtils.rouletteGambling(p); //roulette gambling for updating the topic of a word
				z_word[d][n] = topic;
				updateCount(d, topic, docword[d][n], +1, 0);  // update the count ++1
			}
			//for links of this document
			for (int n = 0; n < z_link[d].length; n++) {
				int topic = z_link[d][n]; // get the old topic
				updateCount(d, topic, doclink[d][n], -1, 1); // update the count --1
				double[] p = new double[K];
				for (int k = 0; k < K; k++) {
					p[k] = (ndk[d][k] + alpha) / (ndsum[d] + K * alpha) * (nkl[k][doclink[d][n]] + gamma)
							/ (nksum_l[k] + L * gamma);
				}
				topic = FuncUtils.rouletteGambling(p); //roulette gambling for updating the topic of a word
				z_link[d][n] = topic;
				updateCount(d, topic, doclink[d][n], +1, 1);  // update the count ++1
			}
		}
	}
	/**
	 * update the count for word or link of assignment
	 * 
	 * @param d
	 * @return
	 */
	void updateCount(int d, int topic, int wordOrLink, int flagCount, int flagWordOrLink) {
		//word update
		if (flagWordOrLink == 0) {
			ndk[d][topic] += flagCount;
			ndsum[d] += flagCount;
			nkw[topic][wordOrLink] += flagCount;
			nksum_w[topic] += flagCount;
		}else {  //link update
			ndk[d][topic] += flagCount;
			ndsum[d] += flagCount;
			nkl[topic][wordOrLink] += flagCount;
			nksum_l[topic] += flagCount;
		}
	}
	/**
	 * obtain the parameter Theta
	 */
	public double[][] estimateTheta() {
		double[][] theta = new double[docword.length][K];
		for (int d = 0; d < docword.length; d++) {
			for (int k = 0; k < K; k++) {
				theta[d][k] = (ndk[d][k] + alpha) / (ndsum[d] + K * alpha);
			}
		}
		return theta;
	}
	/**
	 * obtain the parameter Phi for words
	 */
	public double[][] estimatePhi_word() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nkw[k][w] + beta) / (nksum_w[k] + V * beta);
			}
		}
		return phi;
	}
	/**
	 * obtain the parameter Phi for words
	 */
	public double[][] estimatePhi_link() {
		double[][] phi = new double[K][L];
		for (int k = 0; k < K; k++) {
			for (int l = 0; l < L; l++) {
				phi[k][l] = (nkl[k][l] + gamma) / (nksum_l[k] + L * gamma);
			}
		}
		return phi;
	}
	/**
	 * write top words with probability for each topic
	 */
	public void writeTopWordsWithProbability(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] phi = estimatePhi_word();
		int topicNumber = 1;
		for (double[] phi_z : phi) {
			sBuilder.append("Topic:" + topicNumber + "\n");
			for (int i = 0; i < topWordsAndLinksOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_z);
				sBuilder.append(indexToWordMap.get(max_index) + " :" + phi_z[max_index] + "\n");
				phi_z[max_index] = 0;
			}
			sBuilder.append("\n");
			topicNumber++;
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "LinkLDA_topic_word_" + K + ".txt", sBuilder.toString(),outputFilecode);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write top words for each topic
	 */
	public void writeTopWords(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] phi = estimatePhi_word();
		for (double[] phi_z : phi) {
			for (int i = 0; i < topWordsAndLinksOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_z);
				sBuilder.append(indexToWordMap.get(max_index) + "\t");
				phi_z[max_index] = 0;
			}
			sBuilder.append("\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "LinkLDA_topic_wordnop_" + K + ".txt", sBuilder.toString(),outputFilecode);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write top words with probability for each topic
	 */
	public void writeTopLinksWithProbability(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] phi = estimatePhi_link();
		int topicNumber = 1;
		for (double[] phi_z : phi) {
			sBuilder.append("Topic:" + topicNumber + "\n");
			for (int i = 0; i < topWordsAndLinksOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_z);
				sBuilder.append(indexLinkMap.get(max_index) + " :" + phi_z[max_index] + "\n");
				phi_z[max_index] = 0;
			}
			sBuilder.append("\n");
			topicNumber++;
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "LinkLDA_topic_link_" + K + ".txt", sBuilder.toString(),outputFilecode);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write top words for each topic
	 */
	public void writeTopLinks(){
		StringBuilder sBuilder = new StringBuilder();
		double[][] phi = estimatePhi_link();
		for (double[] phi_z : phi) {
			for (int i = 0; i < topWordsAndLinksOutputNumber; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_z);
				sBuilder.append(indexLinkMap.get(max_index) + "\t");
				phi_z[max_index] = 0;
			}
			sBuilder.append("\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "LinkLDA_topic_linknop_" + K + ".txt", sBuilder.toString(),outputFilecode);
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
			FileUtil.writeFile(outputFileDirectory + "LinkLDA_doc_topic_" + K + ".txt", sBuilder.toString(),outputFilecode);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String args[]) throws Exception{
		LinkLDA linklda = new LinkLDA("data/rawdata_process_link", "gbk", "--", 50, 0.1,
				0.01,0.01, 200, 50, "data/linkldaoutput/");
		linklda.MCMCSampling();
	}
}
