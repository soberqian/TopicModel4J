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
 * Collapsed Variational Bayesian Inference for LDA
 * 
 * Reference:
 * Teh Y W, Newman D, Welling M. A collapsed variational Bayesian inference algorithm for latent Dirichlet allocation[C]//Advances in neural information processing systems. 2007: 1353-1360.
 * Asuncion A, Welling M, Smyth P, et al. On smoothing and inference for topic models[C]//Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009: 27-34.
 * 
 * @author: Yang Qian,Yuanchun Jian,Yidong Chai,Yezheng Liu,Jianshan Sun (HeFei University of Technology)
 */

public class CVBLDA
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
	//variational parameters
	public double[][] nmk; //�ĵ�d������k���ɵĵ�����Ŀ(����)nmk+cmk
	public double[] nm; //
	public double[][] nkw; //����k�����ĵ���w����Ŀ(����) K*V
	public double[] nkw_sum; //����k�������ܵ�����Ŀ(����)
	public double[][][] gamma_word; 
	//output
	public int topWordsOutputNumber;
	public String outputFileDirectory; 
	public CVBLDA(String inputFile, String inputFileCode, int topicNumber,
			double inputAlpha, double inputBeta, int inputIterations, int inTopWords,
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
		iterations = inputIterations;
		topWordsOutputNumber = inTopWords;
		outputFileDirectory = outputFileDir;
		initialize();
	}
	/**
	 * Randomly initialize the parameter
	 */
	public void initialize(){
		//�ĵ�����
		int D = docword.length;
		//variational parameters
		nmk = new double[D][K];
		nm = new double[D];
		nkw = new double[K][V];
		nkw_sum = new double[K];
		gamma_word = new double[D][][]; 
		//ѭ��ÿƪ�ĵ�
		for (int d = 0; d < D; d++) {
			// �ĵ���������
			int Nd = docword[d].length;
			gamma_word[d] = new double[Nd][K];
			for(int n = 0; n < Nd; n ++) {
				gamma_word[d][n] = FuncUtils.getGaussianSample(K, 0.5, 0.5);
				double gamma_norm = 0;
				for(int k = 0; k < K; k ++) {
					gamma_norm += Math.exp(gamma_word[d][n][k]);
				}
				for(int k = 0; k < K; k ++) {
					gamma_word[d][n][k] = Math.exp(gamma_word[d][n][k]) / gamma_norm;
					nkw_sum[k] += gamma_word[d][n][k];
					nmk[d][k] += gamma_word[d][n][k];
					nkw[k][docword[d][n]] += gamma_word[d][n][k];
					nm[d] += gamma_word[d][n][k];
				}
			}
		}

	}
	public void CVBInference(){
		for (int iter = 1; iter <= iterations; iter++) {
			System.out.println("iteration : " + iter);
			iterateCVB0Update();
		}
		// output the result
		System.out.println("write topic word ..." );
		writeTopWordsWithProbability();
		System.out.println("write document topic ..." );
		writeDocumentTopic();
		writeTopWords();

	}
	public void iterateCVB0Update() {
		int D = docword.length;
		for(int d = 0; d < D; d ++) {
			for(int n = 0; n < docword[d].length; n ++) {
				double norm_w = 0;
				double[] gamma_w = new double[K];
				for(int k = 0; k < K; k ++) {
					gamma_w[k] = gamma_word[d][n][k];
					gamma_word[d][n][k] = (updateCount(d, n, k, 0, d) + alpha)*
							(beta + updateCount(d, n, k, docword[d][n], -1))/(V * beta + updateCount(d, n, k, 0, -1));
					norm_w += gamma_word[d][n][k];
				}
				for(int k = 0; k < K; k ++) {
					gamma_word[d][n][k] /= norm_w;
					//update
					nkw_sum[k] += gamma_word[d][n][k] - gamma_w[k];
					nmk[d][k] += gamma_word[d][n][k] - gamma_w[k];
					nkw[k][docword[d][n]] += gamma_word[d][n][k] - gamma_w[k];
					nm[d] += gamma_word[d][n][k] - gamma_w[k];
				}
			}
		}
	}
	/**
	 * update the count 
	 * expect the word d_n
	 * @param 
	 * @return
	 */
	public double updateCount(int d, int n, int k, int wsdn, int doc) {
		if(wsdn == 0 && doc == -1)
			return nkw_sum[k] - gamma_word[d][n][k];
		else if(doc == -1)
			return nkw[k][wsdn] - gamma_word[d][n][k];
		else
			return nmk[doc][k] - gamma_word[d][n][k];
	}
	/**
	 * obtain the parameter Theta
	 */
	public double[][] estimateTheta() {
		double[][] theta = new double[docword.length][K];
		for (int d = 0; d < docword.length; d++) {
			for (int k = 0; k < K; k++) {
				theta[d][k] = (nmk[d][k] + alpha) / (nm[d] + K * alpha);
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
			FileUtil.writeFile(outputFileDirectory + "topic_word_CVB_" + K + ".txt", sBuilder.toString(),"gbk");
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
			FileUtil.writeFile(outputFileDirectory + "topic_wordnop_CVB_" + K + ".txt", sBuilder.toString(),"gbk");
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
			FileUtil.writeFile(outputFileDirectory + "doc_topic_CVB_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String args[]) throws Exception{
		CVBLDA lda = new CVBLDA("data/rawdata_process", "gbk", 30, 0.1,
				0.01, 200, 50, "data/ldaoutput/");
		lda.CVBInference();
	}
}
