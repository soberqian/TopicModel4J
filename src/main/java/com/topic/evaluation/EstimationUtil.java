package com.topic.evaluation;
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
import com.topic.utils.FuncUtils;
/**
 * TopicModel4J: A Java package for topic models
 * 
 * Collapsed Gibbs sampling for author-topic model
 * 
 * Reference:
 * Mimno D, Wallach H M, Talley E, et al. Optimizing semantic coherence in topic models[C]//Proceedings of the conference on empirical methods in natural language processing. Association for Computational Linguistics, 2011: 262-272.
 * https://github.com/yao8839836/PTM
 */
public class EstimationUtil {
	/**
	 * 
	 * @param docs
	 * @param phi
	 * @param top_words_size
	 * @return
	 */
	public static double average_coherence(int[][] docs, double[][] phi, int topN_word) {

		double total_coherence = 0;
		for (double[] phi_t : phi) {
			int[] topWords_N = new int[topN_word];
			for (int i = 0; i < topN_word; i++) {
				int max_index = FuncUtils.maxValueIndex(phi_t);
				topWords_N[i] = max_index;
				phi_t[max_index] = 0;
			}
			double coherence_score = 0.0;
			for (int m = 1; m < topWords_N.length; m++) {
				for (int l = 0; l < m; l++) {
					if (topWords_N[m] != topWords_N[l]){
						coherence_score += Math.log((double) (DocumentFrequency(docs, topWords_N[m], topWords_N[l]) + 1)
								/ DocumentFrequency(docs, topWords_N[l]));
					}else{
						coherence_score += Math.log((double) 2 / DocumentFrequency(docs, topWords_N[l]));
					}
				}
			}
			total_coherence += coherence_score;
		}
		double average_coherence = total_coherence / phi.length;
		return average_coherence;
	}
	/**
	 * 
	 * @param documents
	 * @param word
	 * @return count
	 */
	public static int DocumentFrequency(int[][] documents, int word) {
		int count = 0;
		for (int i = 0; i < documents.length; i++) {
			for (int j = 0; j < documents[i].length; j++) {
				if (documents[i][j] == word) {
					count++;
					break;
				}

			}
		}
		return count;
	}
	
	/**
	 * 
	 * @param documents
	 * @param word_i
	 * @param word_j
	 * @return count
	 */
	public static int DocumentFrequency(int[][] documents, int word_i, int word_j) {
		int count = 0;
		for (int i = 0; i < documents.length; i++) {
			boolean exsit_i = false;
			boolean exsit_j = false;
			for (int j = 0; j < documents[i].length; j++) {
				if (documents[i][j] == word_i) {
					exsit_i = true;
					break;
				}
			}
			for (int j = 0; j < documents[i].length; j++) {
				if (documents[i][j] == word_j) {
					exsit_j = true;
					break;
				}
			}
			if (exsit_i && exsit_j)
				count++;
		}
		return count;
	}
}
