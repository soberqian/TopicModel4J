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
 * Gibbs sampling for BTM
 * 
 * Reference:
 * Cheng X, Yan X, Lan Y, et al. Btm: Topic modeling over short texts[J]. IEEE Transactions on Knowledge and Data Engineering, 2014, 26(12): 2928-2941.
 * Yan X, Guo J, Lan Y, et al. A biterm topic model for short texts[C]//Proceedings of the 22nd international conference on World Wide Web. ACM, 2013: 1445-1456.
 * 
 * @author: Yang Qian,Yuanchun Jian,Yidong Chai,Yezheng Liu,Jianshan Sun (HeFei University of Technology)
 */

public class BTM
{
	public double alpha; // Hyper-parameter alpha
	public double beta; // Hyper-parameter beta
	public int K; // number of topics
	public int iterations; // number of iterations
	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int M; // number of documents in the corpus
	public int V; // number of words in the corpus
	public int [][] docword;//word index array
	//biterm realted
	public int[][] biterms;
	public int windowSize;
	public int[] z;
	public int[][] nkw; 
	public int[] nkw_sum; 
	public int[] nk;
	//output
	public int topWordsOutputNumber;
	public String outputFileDirectory; 
	public BTM(String inputFile, String inputFileCode, int topicNumber,
			double inputAlpha, double inputBeta, int inputIterations, int inTopWords, int windowS,
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
		K = topicNumber;
		windowSize = windowS;
		iterations = inputIterations;
		topWordsOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		//generate biterms
		biterms = generateBiterms(docword, windowSize);
		//initialize
		initialize();
	}
	/**
	 * Randomly assign the topic for each biterm
	 */
	public void initialize(){
		//Biterm size
		int NB = biterms.length;
		//biterm realted
		z = new int[NB];
		nkw = new int[K][V];
		nkw_sum = new int[K];
		nk = new int[K];
		for (int b = 0; b < NB; ++b) {
			int topic = (int) (Math.random() * K);
			z[b] = topic;
			nkw[topic][biterms[b][0]]++;
			nkw[topic][biterms[b][1]]++;
			nk[topic]++;
			nkw_sum[topic] += 2;
		}
	}
	public void MCMCSampling(){
		for (int iter = 1; iter <= iterations; iter++) {
			System.out.println("iteration : " + iter);
			gibbsOneIteration();
		}
		// output the result
		writeTopWordsWithProbability();
		writeTopicDistri();
		writeTopicDocument();
//		writeTopWords();
		

	}
	public void gibbsOneIteration() {
		for (int i = 0; i < biterms.length; i++) {
			int topic = z[i];
			updateCount(i, topic, 0);
			double[] p = new double[K];
			for (int k = 0; k < K; ++k) {
				p[k] = (nk[k] + alpha) * ((nkw[k][biterms[i][0]] + beta) / (nkw_sum[k] + V * beta))
						* ((nkw[k][biterms[i][1]] + beta) / (nkw_sum[k] + V * beta + 1));

			}
			topic = FuncUtils.rouletteGambling(p); //roulette gambling for updating the topic of a word
			z[i] = topic;
			updateCount(i, topic, 1);
		}
	}
	/**
	 * update the count nkw, nk and nkw_sum
	 * 
	 * @param biterm
	 * @param topic
	 * @param flag
	 * @return null
	 */
	void updateCount(int biterm, int topic, int flag) {
		if (flag == 0) {
			nkw[topic][biterms[biterm][0]]--;
			nkw[topic][biterms[biterm][1]]--;
			nk[topic]--;
			nkw_sum[topic] -= 2;
		}else {
			nkw[topic][biterms[biterm][0]]++;
			nkw[topic][biterms[biterm][1]]++;
			nk[topic]++;
			nkw_sum[topic] += 2;
		}

	}
	/**
	 * obtain the parameter theta
	 */
	public double[] estimateTheta() {
		double[] theta = new double[K];
		for (int k = 0; k < K; k++) {
			theta[k] = (nk[k] + alpha) / (biterms.length + K * alpha);
		}
		return theta;
	}
	/**
	 * obtain the parameter phi
	 */
	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phi[k][w] = (nkw[k][w] + beta) / (nkw_sum[k] + V * beta);
			}
		}
		return phi;
	}
	/**
	 * evaluating the topic posterior P(z|d) for document d
	 */
	public double[][] estimatePdz() {
		double[][] phi = estimatePhi();
		double[] theta = estimateTheta();
		double[][] pdz = new double[docword.length][K];
		System.out.println(docword.length);
		for (int i = 0; i < docword.length; i++) {
			int[] document = docword[i];
			int[][] bitermsDoc = generateBitermsForOneDoc(document, windowSize);
			double pzb[] = new double[K];
			for(int b = 0 ;b < bitermsDoc.length; b++){
				double sum = 0.0;
				for( int k=0; k < K; k++){
					pzb[k] = theta[k] * phi[k][bitermsDoc[b][0]] * phi[k][bitermsDoc[b][1]];
					sum += pzb[k];
				}
				//normalize pzb
				for (int k=0; k < K; k++) {
					pzb[k] = pzb[k]/sum;
					pdz[i][k] += pzb[k];	
				}
			}
		}
		//normalize pdz
		for (int i = 0; i < pdz.length; i++) {
			for (int k = 0; k < pdz[i].length; k++) {
				pdz[i][k] = pdz[i][k]/generateBitermsForOneDoc(docword[i], windowSize).length;
			}
		}
		return pdz;
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
			FileUtil.writeFile(outputFileDirectory + "BTM_topic_word_" + K + ".txt", sBuilder.toString(),"gbk");
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
			FileUtil.writeFile(outputFileDirectory + "BTM_topic_wordnop_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write theta for each topic
	 */
	public void writeTopicDistri(){
		double[] theta = estimateTheta();
		StringBuilder sBuilder = new StringBuilder();
		for (int k = 0; k < K; k++) {
			sBuilder.append(theta[k] + "\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "BTM_topic_theta_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write topic components for each document
	 */
	public void writeTopicDocument(){
		double[][] pdz = estimatePdz();
		StringBuilder sBuilder = new StringBuilder();
		for (int i = 0; i < K; i++) {
			sBuilder.append("Topic" + (i+1) + "\t");
		}
		for (int d = 0; d < pdz.length; d++) {
			StringBuilder doc = new StringBuilder();
			for (int k = 0; k < pdz[d].length; k++) {
				doc.append(pdz[d][k] + "\t");
			}
			sBuilder.append(doc.toString().trim() + "\n");
		}
		try {
			FileUtil.writeFile(outputFileDirectory + "BTM_doc_topic_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * generate biterms
	 * @param documents 
	 * @param windowSize 
	 * @return biterms
	 */
	public int[][] generateBiterms(int[][] documents, int windowSize) {
		List<int[]> list = new ArrayList<int[]>();
		for (int d = 0; d < documents.length; ++d) {
			for (int i = 0; i < documents[d].length - 1; ++i) {
				for (int j = i + 1; j < Math.min(i + windowSize, documents[d].length); ++j) {
					list.add(new int[]{documents[d][i], documents[d][j]});
				}
			}
		}
		int[][] biterms = new int[list.size()][2];
		list.toArray(biterms);
		return biterms;
	}
	/**
	 * generate biterms for a document
	 * @param documents 
	 * @param windowSize 
	 * @return biterms
	 */
	public int[][] generateBitermsForOneDoc(int[] document, int windowSize) {
		List<int[]> list = new ArrayList<int[]>();
		for (int i = 0; i < document.length - 1; ++i) {
			for (int j = i + 1; j < Math.min(i + windowSize, document.length); ++j) {
				list.add(new int[]{document[i], document[j]});
			}
		}
		int[][] biterms = new int[list.size()][2];
		list.toArray(biterms);
		return biterms;
	}
	public static void main(String args[]) throws Exception{
		BTM btm = new BTM("data/shortdoc.txt", "utf-8", 50, 0.1,
				0.01, 100, 50, 50, "data/ldaoutput/");
		btm.MCMCSampling();
	}
}
