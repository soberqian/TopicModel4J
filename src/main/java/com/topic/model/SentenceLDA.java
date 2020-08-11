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
 * Collapsed Gibbs sampling for Sentence-LDA
 * 
 * Reference:
 * Jo Y, Oh A H. Aspect and sentiment unification model for online review analysis[C]//Proceedings of the fourth ACM international conference on Web search and data mining. ACM, 2011: 815-824.
 * B��schken J, Allenby G M. Sentence-based text analysis for customer reviews[J]. Marketing Science, 2016, 35(6): 953-975.
 * 
* @author: Yang Qian,Yuanchun Jian,Yidong Chai,Yezheng Liu,Jianshan Sun (HeFei University of Technology)
 */
public class SentenceLDA {
	public double alpha; // Hyper-parameter alpha
	public double beta; // Hyper-parameter beta
	public int K; // number of topics
	public int iterations; // number of Gibbs sampling iterations
	public int topWords; // number of most probable words for each topic
	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int M; // number of documents in the corpus
	public int V; // number of words in the corpus
	//sentence LDA 
	public int S; // number of sentence in the corpus
	public int [][][] docsenword;//document-sentence-word
	public int[][] ndk; // the number of sentences that are assigned topic k in review d
	public int[] ndsum; //the number of sentences of document d
	public int[][] nkw; //topic-word count
	public int[] nksum; //topic-word sum (total number of words assigned to a topic)
	public int[][] z; //the topic assignment of the sentence in document d
	//Example: given a sentence of "a a b a b c d c". We have: 1 2 1 3 2 1 1 2
	public List<List<List<Integer>>> docSenWordIndexCount;   //doc-sen-wordIndex = number
	//output
	public int topWordsOutputNumber;
	public String outputFileDirectory; 
	public SentenceLDA(String inputFile, String inputFileCode, int topicNumber,
			double inputAlpha, double inputBeta, int inputIterations, int inTopWords,
			String outputFileDir){
		//read data
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines(inputFile, docLines,inputFileCode);
		M = docLines.size();
		docsenword = new int[M][][];
		docSenWordIndexCount = new ArrayList<List<List<Integer>>>();
		int j = 0;
		for(String line : docLines){
			//get the sentence of each document
			List<String> sentences = new ArrayList<String>();
			FileUtil.splitToSentence(line, sentences, "--");
			docsenword[j] = new int[sentences.size()][];
			List<List<Integer>> docSentenceIndexCount = new ArrayList<List<Integer>>();
			//sentence to words
			for (int i = 0; i < sentences.size(); i++) {
				List<String> words = new ArrayList<String>();
				FileUtil.tokenizeAndLowerCase(line, words);
				docsenword[j][i] = new int[words.size()];
				HashMap<String, Integer> wordCount = new HashMap<String, Integer>();
				List<Integer> wordIndexCount = new ArrayList<Integer>();
				for(int w = 0; w < words.size(); w++){
					String word = words.get(w);
					if(!wordToIndexMap.containsKey(word)){
						int newIndex = wordToIndexMap.size();
						wordToIndexMap.put(word, newIndex);
						indexToWordMap.add(word);
						docsenword[j][i][w] = newIndex;
					} else {
						docsenword[j][i][w] = wordToIndexMap.get(word);
					}
					//Example: given a sentence of "a a b a b c d c". We have: 1 2 1 3 2 1 1 2
					int times = 0;
					if (wordCount.containsKey(word)) {
						times = wordCount.get(word);
					}
					times += 1;
					wordCount.put(word, times);
					wordIndexCount.add(times);
				}
				docSentenceIndexCount.add(wordIndexCount);
			}
			docSenWordIndexCount.add(docSentenceIndexCount);
			j++;

		}
		V = indexToWordMap.size();
		alpha = inputAlpha;
		beta = inputBeta;
		K = topicNumber;
		iterations = inputIterations;
		topWordsOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		initialize();
	}
	/**
	 * Randomly initialize topic assignment of each sentence in one document
	 */
	public void initialize(){
		ndk = new int[M][K];
		ndsum = new int[M];
		nkw = new int[K][V];
		nksum = new int[K];
		z = new int[M][];
		for (int d = 0; d < M; d++) {
			int Ns = docsenword[d].length;  // the number of sentences in a document
			z[d] = new int[Ns];
			for (int s = 0; s < Ns; s++) {
				int topic = (int) (Math.random() * K);
				z[d][s] = topic;
				updateCount(d, topic, s, +1);
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
			for (int s = 0; s < z[d].length; s++) {
				int topic = z[d][s]; // get the old topic
				updateCount(d, topic, s, -1); // update the count --1
				double[] p = new double[K];
				for (int k = 0; k < K; k++) {
					p[k] = (ndk[d][k] + alpha) / (ndsum[d] + K * alpha);
					//for each word in the sentence
					for (int w = 0; w < docsenword[d][s].length; w ++) {
						p[k] *= (nkw[k][docsenword[d][s][w]] + beta + docSenWordIndexCount.get(d).get(s).get(w) - 1)
								/ (nksum[k] + V * beta + s);
					}
				}
				topic = FuncUtils.rouletteGambling(p); //roulette gambling for updating the topic of a word
				z[d][s] = topic;
				updateCount(d, topic, s, +1);  // update the count ++1
			}
		}
	}
	/**
	 * update the count 
	 * 
	 * @param probs ndk,ndsum,nkw,nksum
	 * @return
	 */
	void updateCount(int d, int topic, int s ,int flag) {
		ndk[d][topic] += flag;
		ndsum[d] += flag;
		for (int w = 0; w < docsenword[d][s].length; w++) {
			nkw[topic][docsenword[d][s][w]] += flag;
			nksum[topic] += flag;
		}
	}
	/**
	 * obtain the parameter Theta
	 */
	public double[][] estimateTheta() {
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
			FileUtil.writeFile(outputFileDirectory + "SentenceLDA_topic_word_" + K + ".txt", sBuilder.toString(),"gbk");
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
			FileUtil.writeFile(outputFileDirectory + "SentenceLDA_doc_topic" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String args[]) throws Exception{
		SentenceLDA sentenceLda = new SentenceLDA("data/rawdata_sentenceLDA", "gbk", 50, 0.1,
				0.01, 500, 50, "data/ldaoutput/");
		sentenceLda.MCMCSampling();
	}
}
