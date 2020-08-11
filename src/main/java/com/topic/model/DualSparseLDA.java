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

import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;

import com.topic.utils.FileUtil;
import com.topic.utils.FuncUtils;

/**
 * TopicModel4J: A Java package for topic models
 * 
 * Collapsed Variational Bayesian Inference for Dual-Sparse Topic Model
 * 
 * Reference:
 * Lin T, Tian W, Mei Q, et al. The dual-sparse topic model: mining focused topics and focused terms in short text[C]//Proceedings of the 23rd international conference on World wide web. ACM, 2014: 539-550.
 * 
 * @author: Yang Qian,Yuanchun Jian,Yidong Chai,Yezheng Liu,Jianshan Sun (HeFei University of Technology)
 */
public class DualSparseLDA {
	public double s; // Hyper-parameter for a
	public double r; // Hyper-parameter for a
	public double x; // Hyper-parameter for b
	public double y; // Hyper-parameter for b
	public double gamma; // Hyper-parameter
	public double gamma_bar; // Hyper-parameter
	public double pi; // Hyper-parameter
	public double pi_bar; // Hyper-parameter
	public int K; // number of topics
	public int iterations; // number of iterations
	public Map<String, Integer> wordToIndexMap = new HashMap<String, Integer>();;  //word to index
	public List<String> indexToWordMap = new ArrayList<String>();    //index to String word 
	public int M; // number of documents in the corpus
	public int V; // number of words in the corpus
	public int [][] docword;//word index array
	//variational parameters for DsparseTM
	public double[][] a_mk; //�ĵ�����ѡ����
	public double a_sum[];  //�ĵ�����ĸ���(����)
	public double[][] nmk; //�ĵ�d������k���ɵĵ�����Ŀ(����)
	public double[] nm; //�ĵ�d�����Ĵ���Ŀ(����)
	public double[][] nkw; //����k�����ĵ���w����Ŀ(����) 
	public double[] nkw_sum; //����k�������ܵ�����Ŀ(����)
	public double[][] b_kv; //�����ѡ����
	public double b_sum[];  //�����Ӧ�Ĵʸ���(����)
	public double[][][] gamma_word; 
	//output
	public int topWordsOutputNumber;
	public String outputFileDirectory; 
	public DualSparseLDA(String inputFile, String inputFileCode, int topicNumber,
			double inputS, double inputR, double inputX, double inputY, 
			double inputGamma, double inputGamma_bar,double inputPi, double inputPi_bar,
			int inputIterations, int inTopWords,
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
		s = inputS;
		r = inputR;
		x = inputX;
		y = inputY;
		gamma = inputGamma;
		gamma_bar = inputGamma_bar;
		pi = inputPi;
		pi_bar = inputPi_bar;
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
		JDKRandomGenerator rand = new JDKRandomGenerator();
		rand.setSeed(System.currentTimeMillis());
		BetaDistribution betaDist_a = new BetaDistribution(rand, s , r);
		BetaDistribution betaDist_b = new BetaDistribution(rand, x , y);
		int D = docword.length;
		//variational parameters
		a_mk = new double[D][K];
		a_sum = new double[D]; 
		nmk = new double[D][K];
		nm = new double[D];
		nkw = new double[K][V];
		nkw_sum = new double[K];
		b_kv = new double[K][V];
		b_sum = new double[K];
		gamma_word = new double[D][][]; 
		for (int d = 0; d < D; d++) {
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
			double[] a_sigma = new double[K];
			double a_norm = 0;
			for(int k = 0; k < K; k ++) {
				a_sigma[k] = 0.5;
				a_mk[d][k] = betaDist_a.sample();
				a_norm += a_mk[d][k];
			}
			for (int k = 0; k < K; k ++) {
				a_mk[d][k] /= a_norm;
				a_sum[d] += a_mk[d][k];
			}
		}
		for (int k = 0; k < K; k++) {
			double[] b_sigma = new double[V];
			double b_norm = 0;
			for (int v = 0; v < V; v++) {
				b_sigma[v] = 0.5;
				b_kv[k][v] = betaDist_b.sample();
				b_norm += b_kv[k][v];
			}
			for (int v = 0; v < V; v++) {
				b_kv[k][v] /= b_norm;
				b_sum[k] += b_kv[k][v];
			}
		}

	}
	public void CVBInference(){
		for (int iter = 1; iter <= iterations; iter++) {
			System.out.println("iteration : " + iter);
			iterateCVB0Update();
		}
		//output the result
		System.out.println("write topic word ..." );
		writeTopWordsWithProbability();
//		writeTopWords();
		System.out.println("write document topic ..." );
		writeDocumentTopic();
		System.out.println("write the sparsity ratio ..." );
		writeSparsityRatio();
	}
	public void iterateCVB0Update() {
		int D = docword.length;
		//update a_mk
		for(int d = 0; d < D; d ++) {
			double[] prev_a = new double[K];
			for(int k = 0; k < K; k ++) {
				prev_a[k] = a_mk[d][k];
				double Am = count_Am(d,k);
				double log_a1 = FuncUtils.logOn2(s + Am) + 
						FuncUtils.logOn2Gamma(nmk[d][k] + pi + pi_bar) + FuncUtils.log2betaf(pi + pi*Am + K*pi_bar, nm[d] + pi*Am + K*pi_bar);
				double log_a0 = FuncUtils.logOn2(r + K - 1.0 - Am) +
						FuncUtils.logOn2Gamma(pi + pi_bar) + FuncUtils.log2betaf(pi*Am + K*pi_bar, nm[d] + pi*Am + pi + K*pi_bar);
				if (FuncUtils.exponential2(log_a1) > 1024) {
					a_mk[d][k] = Double.MAX_VALUE/(Double.MAX_VALUE + FuncUtils.exponential2(log_a0));
				}else {
					a_mk[d][k] = FuncUtils.exponential2(log_a1)/(FuncUtils.exponential2(log_a1) + FuncUtils.exponential2(log_a0));
				}
			}
			for(int k = 0; k < K; k ++) {
				a_sum[d] +=  a_mk[d][k] - prev_a[k]; 
			}
		}
		//update b_kv
		for(int k = 0; k < K; k ++) {
			double[] prev_b = new double[V];
			for(int v = 0; v < V; v++) {
				prev_b[v] = b_kv[k][v];
				double bk = count_Bk(k,v);
				double log_b1 = FuncUtils.logOn2(x + bk) + 
						FuncUtils.logOn2Gamma(nkw[k][v] + gamma + gamma_bar) + FuncUtils.log2betaf(gamma + gamma*bk + V*gamma_bar, nkw_sum[k] + gamma*bk + V*gamma_bar);
				double log_b0 = FuncUtils.logOn2(y + V - 1.0 - bk) +
						FuncUtils.logOn2Gamma(gamma + gamma_bar) + FuncUtils.log2betaf(gamma*bk + V*gamma_bar, nkw_sum[k] + gamma + gamma*bk + V*gamma_bar );
				if (FuncUtils.exponential2(log_b1) > 1024) {
					b_kv[k][v] = Double.MAX_VALUE/(Double.MAX_VALUE + FuncUtils.exponential2(log_b0));
				}else {
					b_kv[k][v] = FuncUtils.exponential2(log_b1)/(FuncUtils.exponential2(log_b1) + FuncUtils.exponential2(log_b0));
				}
			}
			for(int v = 0; v < V; v++) {
				b_sum[k] +=  b_kv[k][v] - prev_b[v]; 
			}
		}
		//update gamma_word
		for(int d = 0; d < D; d ++) {
			for(int n = 0; n < docword[d].length; n ++) {
				double norm_w = 0;
				double[] prev_gamma_w = new double[K];
				for(int k = 0; k < K; k ++) {
					prev_gamma_w[k] = gamma_word[d][n][k];
					gamma_word[d][n][k] = (mean_count_gamma_w(d, n, k, 0, d) +  + a_mk[d][k]*pi + pi_bar)*
							(gamma_bar + gamma*b_kv[k][docword[d][n]] + mean_count_gamma_w(d, n, k, docword[d][n], -1))
							/(V * gamma_bar + gamma * b_sum[k] + mean_count_gamma_w(d, n, k, 0, -1));
					norm_w += gamma_word[d][n][k];
				}
				for(int k = 0; k < K; k ++) {
					gamma_word[d][n][k] /= norm_w;
					//maintain
					nkw_sum[k] += gamma_word[d][n][k] - prev_gamma_w[k];
					nmk[d][k] += gamma_word[d][n][k] - prev_gamma_w[k];
					nkw[k][docword[d][n]] += gamma_word[d][n][k] - prev_gamma_w[k];
					nm[d] += gamma_word[d][n][k] - prev_gamma_w[k];
				}
			}
		}
	}
	//a_sumͳ
	private double count_Am(int d, int k) {
		return a_sum[d] - a_mk[d][k];
	}
	//b_sumͳ
	private double count_Bk(int k, int v) {
		return b_sum[k] - b_kv[k][v];
	}
	private double mean_count_gamma_w(int ex_d, int ex_n, int k, int wsdn, int doc) {
		if(wsdn == 0 && doc == -1)
			return nkw_sum[k] - gamma_word[ex_d][ex_n][k];
		else if(doc == -1)
			return nkw[k][wsdn] - gamma_word[ex_d][ex_n][k];
		else
			return nmk[doc][k] - gamma_word[ex_d][ex_n][k];
	}
	//Theta
	public double[][] estimateTheta() {
		double[][] theta = new double[docword.length][K];
		for (int d = 0; d < docword.length; d++) {
			for (int k = 0; k < K; k++) {
				theta[d][k] = (nmk[d][k] + a_mk[d][k]*pi + pi_bar) / (nm[d] + pi*a_sum[k] +  K * pi_bar);
			}
		}
		return theta;
	}
	//Phi
	public double[][] estimatePhi() {
		double[][] phi = new double[K][V];
		for (int k = 0; k < K; k++) {
			for (int v = 0; v < V; v++) {
				phi[k][v] = (nkw[k][v] + b_kv[k][v]*gamma + gamma_bar) / (nkw_sum[k] + gamma*b_sum[k] +  V * gamma_bar);
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
			FileUtil.writeFile(outputFileDirectory + "dualSLDA_topic_word_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write top words for each topic
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
			FileUtil.writeFile(outputFileDirectory + "dualSLDA_topic_wordnop_" + K + ".txt", sBuilder.toString(),"gbk");
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
			FileUtil.writeFile(outputFileDirectory + "dualSLDA_doc_topic_" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * write sparsity ratio
	 */
	public void writeSparsityRatio(){
		double[] sparsityratio_DT = estimateSparsityRatioDT();
		double Asparsityratio_DT = estimateAverSparsityRatioDT();
		StringBuilder sBuilder = new StringBuilder();
		for (int d = 0; d < sparsityratio_DT.length; d++) {
			sBuilder.append(sparsityratio_DT[d] + "\n");
		}
		sBuilder.append("average saprse ratio of doc_topic:" + Asparsityratio_DT + "\n");
		try {
			FileUtil.writeFile(outputFileDirectory + "dualSLDA_sparseRatio_DT" + K + ".txt", sBuilder.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}

		double[] sparsityratio_TV = estimateSparsityRatioTV();
		double Asparsityratio_TV = estimateAverSparsityRatioTV();
		StringBuilder sBuilder1 = new StringBuilder();
		for (int k = 0; k < sparsityratio_TV.length; k++) {
			sBuilder1.append(sparsityratio_TV[k] + "\n");
		}
		sBuilder1.append("average saprse ratio of topic_word:" + Asparsityratio_TV + "\n");
		try {
			FileUtil.writeFile(outputFileDirectory + "dualSLDA_sparseRatio_TV" + K + ".txt", sBuilder1.toString(),"gbk");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	//sparsity ratio for doc-topic
	public double[] estimateSparsityRatioDT() {
		double[] sparsityratio = new double[docword.length];
		for (int d = 0; d < docword.length; d++) {
			double ratio_m = 0.0;
			for (int k = 0; k < K; k++) {
				ratio_m += a_mk[d][k];
			}
			sparsityratio[d] = 1 - ratio_m/K;
		}
		return sparsityratio;
	}
	//average sparsity ratio for doc-topic
	public double estimateAverSparsityRatioDT() {
		double aver_sparsityratio = 0.0;
		for (int d = 0; d < docword.length; d++) {
			aver_sparsityratio +=  (1 - a_sum[d]/K);
		}
		return aver_sparsityratio/docword.length;
	}
	//sparsity ratio for topic-word
	public double[] estimateSparsityRatioTV() {
		double[] sparsityratio = new double[K];
		for (int k = 0; k < K; k++) {
			double ratio_k = 0.0;
			for (int v = 0; v < V; v++) {
				ratio_k += b_kv[k][v];
			}
			sparsityratio[k] = 1 - ratio_k/V;
		}
		return sparsityratio;
	}
	//average sparsity ratio for topic-word
	public double estimateAverSparsityRatioTV() {
		double aver_sparsityratio = 0.0;
		for (int k = 0; k < K; k++) {
			aver_sparsityratio +=  (1 - b_sum[k]/V);
		}
		return aver_sparsityratio/K;
	}
	public static void main(String[] args) {
		DualSparseLDA slda = new DualSparseLDA("data/shortdoc.txt", "gbk", 10, 1.0, 1.0, 1.0, 1.0, 0.1, 1E-12, 0.1, 1E-12, 500, 60, "data/dualsparse/");
		slda.CVBInference();
	}
}
