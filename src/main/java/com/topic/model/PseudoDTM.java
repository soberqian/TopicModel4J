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
 * Collapsed Gibbs sampling in the generative model of Pseudo-document-based Topic Model
 * 
 * Reference:
 * Zuo Y, Wu J, Zhang H, et al. Topic modeling of short texts: A pseudo-document view[C]//Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. ACM, 2016: 2105-2114.
 *
 *https://github.com/maybefeicun/ptm
 *
 * @author: Yang Qian,Yuanchun Jian,Yidong Chai,Yezheng Liu,Jianshan Sun (HeFei University of Technology)
 */
public class PseudoDTM {

	public double alpha; // Hyper-parameter alpha
	public double beta; // Hyper-parameter beta
	public int K; // number of topics
	public int iterations; // number of iterations
	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int M; // number of documents in the corpus
	public int V; // number of words in the corpus
	public int [][] docword;//word index array
	//PTM realted
	public double lambada;
	public int P;  //the number of pseudo documents
	public int ml[]; //the number of short texts assigned to the lth pseudo document
	public int nlk[][];  //the number of tokens assigned to topic z in pseudo document l
	public int nlk_sum[];  //the total number of tokens in pseudo document l
	public int[][] nkw;  //topic-word count
	public int[] nkw_sum;  //topic-word sum (total number of words assigned to a topic)
	public int l[];  //pseudo document assignments l
	public int z[][]; //topic assignments z
	//for each document
	public int[][] ndk;
	public int[] ndsum;
	//output
	public int topWordsOutputNumber;
	public String outputFileDirectory; 
	public PseudoDTM(String inputFile, String inputFileCode, int pDocumentNumber, int topicNumber,
			double inputAlpha, double inputBeta, double inputLambada, int inputIterations, int inTopWords,
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
		lambada = inputLambada;
		K = topicNumber;
		P = pDocumentNumber;
		iterations = inputIterations;
		topWordsOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		initialize();
	}
	/**
	 * Randomly initialize topic assignments
	 */
	public void initialize(){
		ml = new int[P];
		nlk = new int[P][K];
		nlk_sum = new int[P];
		nkw = new int[K][V];
		nkw_sum = new int[K];
		l = new int[M];
		z = new int[M][]; 
		for (int d = 0; d < M; d++) {
			int Nd = docword[d].length;  // the number of words in a document
			z[d] = new int[Nd];
			int pDocumentAssign =  (int) (Math.random() * P);;
			l[d] = pDocumentAssign;
			ml[pDocumentAssign] ++; 
			for (int n = 0; n < Nd; n++) {
				int topic = (int) (Math.random() * K);
				z[d][n] = topic;
				updateCount(d, pDocumentAssign, topic, docword[d][n], +1);
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
		System.out.println("write pseudo-document and document topic ..." );
		writePDocumentTopic();
		writeDocumentTopic();

	}
	public void gibbsOneIteration() {
		//Sampling pseudo document assignments l
		for (int d = 0; d < M; d++) {
			int pDocumentAssign = l[d]; 
			ml[pDocumentAssign] --; 
			List<Integer> zIndexCount = new ArrayList<Integer>();
			Map<Integer, Integer> zCount = new HashMap<Integer, Integer>();
			for(int i = 0; i < z[d].length; i++){
				int topic = z[d][i]; 
				int times = 0;
				if (zCount.containsKey(topic)) {
					times = zCount.get(topic);
				}
				times += 1;
				zCount.put(topic, times);
				zIndexCount.add(times);
				nlk[pDocumentAssign][topic] --;
				nlk_sum[pDocumentAssign] --;
			}
			double[] p = new double[P];
			for (int pAss = 0; pAss < P; pAss++) {
				p[pAss] = (ml[pAss] + lambada)/(M - 1 + P * lambada);
				for (int n = 0; n < z[d].length; n ++) {
					p[pAss] *= (nlk[pAss][z[d][n]] + alpha + zIndexCount.get(n) - 1)
							/ (nlk_sum[pAss] + K * alpha + n);
				}
			}
			pDocumentAssign = FuncUtils.rouletteGambling(p);
			l[d]  = pDocumentAssign; 
			//increase the count
			ml[pDocumentAssign] ++;
			for (int n = 0; n < z[d].length; n++) {
				int topic = z[d][n];
				nlk[pDocumentAssign][topic] ++;
				nlk_sum[pDocumentAssign] ++;
			}
		}
		//Sampling topic assignments z
		for (int d = 0; d < M; d++) {
			for (int n = 0; n < z[d].length; n++) {
				int pDocumentAssign = l[d]; 
				int topic = z[d][n]; 
				updateCount(d, pDocumentAssign, topic, docword[d][n], -1);
				//update the topic
				double[] p = new double[K];
				for (int k = 0; k < K; k++) {
					p[k] = (nlk[pDocumentAssign][k] + alpha) / (nlk_sum[pDocumentAssign] + K *alpha ) *
							(nkw[k][docword[d][n]] + beta) / (nkw_sum[k] + V * beta);
				}
				topic = FuncUtils.rouletteGambling(p); //roulette gambling for updating the topic of a word
				z[d][n] = topic;
				updateCount(d, pDocumentAssign, topic, docword[d][n], 1);
			}
		}
	}
	/**
	 * update the count 
	 * nlk nlk_sum nkw nkw_sum
	 * @param d, l, topic, w, flag
	 * @return
	 */
	void updateCount(int d, int l, int topic, int w, int flag) {
		nlk[l][topic] += flag;
		nlk_sum[l] += flag;
		nkw[topic][w] += flag;
		nkw_sum[topic] += flag;
	}
	/**
	 * obtain the parameter theta for each pseudo document
	 */
	public double[][] estimateThetaP() {
		double[][] theta = new double[P][K];
		for (int pAss = 0; pAss < P; pAss++) {
			for (int k = 0; k < K; k++) {
				theta[pAss][k] = (nlk[pAss][k] + alpha) / (nlk_sum[pAss] + K * alpha);
			}
		}
		return theta;
	}
	/**
	 * obtain the parameter theta for each document
 	 */
	public double[][] estimateThetaD() {
		ndk = new int[M][K];
		ndsum = new int[M];
		for (int d = 0; d < M; d++) {
			for (int n = 0; n < docword[d].length; n++) {
				ndk[d][z[d][n]] ++;
				ndsum[d] ++;
			}
		}
		double[][] theta = new double[M][K];
		for (int d = 0; d < M; d++) {
			for (int k = 0; k < K; k++) {
				theta[d][k] = (ndk[d][k] + alpha) / (ndsum[d] + K * alpha);
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
				phi[k][w] = (nkw[k][w] + beta) / (nkw_sum[k] + V * beta);
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
			FileUtil.writeFile(outputFileDirectory + "PseudoDTM_topic_word_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write theta for each pseudo document
	 */
	public void writePDocumentTopic(){
		double[][] theta = estimateThetaP();
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
			FileUtil.writeFile(outputFileDirectory + "PseudoDTM_pseudo_topic" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write theta for each pseudo document
	 */
	public void writeDocumentTopic(){
		double[][] theta = estimateThetaD();
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
			FileUtil.writeFile(outputFileDirectory + "PseudoDTM_doc_topic" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String[] args) {
		PseudoDTM ptm = new PseudoDTM("data/car.txt", "gbk", 300, 50, 0.1, 0.1,
				0.01, 500, 50, "data/ldaoutput/");
		ptm.MCMCSampling();
	}

}
