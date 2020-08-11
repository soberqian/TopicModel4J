package com.topic.utils;
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
import java.util.Random;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.MathArrays;
/**
 * TopicModel4J: A Java package for topic models
 * 
 * @author: Yang Qian,Yezheng Liu,Yuanchun Jiang (HeFei University of Technology)
 */
public class FuncUtils {
	/**
	 * Sample a value from a double array
	 * 
	 * @param probs 
	 * @return
	 */
	public static int rouletteGambling(double[] prob){
		int topic = 0;
		for (int k = 1; k < prob.length; k++) {
			prob[k] += prob[k - 1];
		}
		double u = Math.random() * prob[prob.length - 1];
		for (int t = 0; t < prob.length; t++) {
			if (u < prob[t]) {
				topic = t;
				break;
			}
		}
		return topic;
	}
	/**
	 * Sample a value from a double array
	 * 
	 * @param probs 
	 * @return
	 */
	public static int rouletteGambling(double[][] prob){
		int K = prob.length;
		int A = prob[0].length;
		double[] pr_sum = new double[K * A];
		for (int k = 0; k < K; k++) {
			for (int a = 0; a < A; a++) {
				pr_sum[k  + a*K] = prob[k][a];
			}
		}
		int idx = rouletteGambling(pr_sum);
		return idx;
	}
	/**
	 * transpose of two-dimensional array
	 * 
	 * @param prob 
	 * @return
	 */
	public static double[][] arrayTrans(double[][] prob){
		double[][] pro_new =  new double[prob[0].length][prob.length];
		for(int i = 0; i < prob[0].length; i++){
			for(int j = 0; j < prob.length; j++){
				pro_new[i][j] = prob[j][i];
			}
		}
		return pro_new;
	}
	/**
	 * get the index of max value in an array
	 * 
	 * @param array
	 * @return the index of max value
	 */
	public static int maxValueIndex(double[] array) {
		double max = array[0];
		int maxVIndex = 0;
		for (int i = 1; i < array.length; i++) {
			if (array[i] > max) {
				max = array[i];
				maxVIndex = i;
			}
		}
		return maxVIndex;
	}

	public static double[] getGaussianSample(int K, double mean, double deviation) {
		Random r = new Random();
		double[] sample = new double[K];
		for(int k = 0; k < K; k ++) {
			sample[k] = r.nextGaussian() * Math.sqrt(deviation) + mean;
		}
		return sample;
	}
	/**sample from multivariate normal distribution
	 * @param mean
	 * @param identity array
	 * ****/
	public static double [] sampleFromMultivariateDistribution(double [] mean, double [][] variance){
		MultivariateNormalDistribution cc = new MultivariateNormalDistribution(mean, variance);
		double[] sampleValues = cc.sample();
		return sampleValues;
	}
	/**generate identity array
	 * [[100.   0.   0.   0.   0.]
	 * [  0. 100.   0.   0.   0.]
	 * [  0.   0. 100.   0.   0.]
	 * [  0.   0.   0. 100.   0.]
	 * [  0.   0.   0.   0. 100.]]
	 * @param identity 100
	 * @param dimension 5
	 * ****/
	public static double [][] generateIdentityArray(double identity,int dimension){
		double arr[][] = new double[dimension][dimension];
		for(int i = 0; i < arr.length; i++) {
			arr[i][i] = identity;  
		}
		return arr;
	}
	/**generate mean of multivariate normal distribution
	 * [0. 0. 0. 0. 0.]
	 * @param mean 0 
	 * @param dimension 4
	 * ****/
	public static double [] generateMeanArray(double mean, int dimension){
		double arr[] = new double[dimension];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = mean;
		}
		return arr;
	}
	/**generate random array
	 * 
	 * @param row
	 * @param column
	 * @param value
	 * if value=1.0, we can get:
	 * [[0.024027   0.168236   0.25728113  0.55045508]
	 * [ 0.263268324  0.28504307   0.290910533  0.16077806]
	 * [ 0.0800969   0.3252087   0.28674675   0.307947605]
	 * ****/
	public static double [][] generateRandomSumOneArray(int row, int column,double value){
		RandomGenerator rg = new JDKRandomGenerator();
		double array[][] = new double[row][column];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				array[i][j] = rg.nextDouble();
			}
			array[i] = MathArrays.normalizeArray(array[i],value);
		}
		return array;
	}
	/**get the inverse matrix of the input Matrix  A
	 * 
	 * @param RealMatrix A
	 * 
	 * ****/
	public static RealMatrix inverseMatrix(RealMatrix A) {
		RealMatrix result = new LUDecomposition(A).getSolver().getInverse();
		return result; 
	}
	/**The probability density function of the Multivariate t distributions
	 * Reference: Conjugate Bayesian analysis of the Gaussian distribution
	 * 
	 * 
	 * @param ArrayRealVector dataPoint
	 * @param ArrayRealVector meansVector
	 * @param RealMatrix covarianceMatrix
	 * @param double degreesOfFreedom
	 * @return The probability value
	 * 
	 * 
	 * @author Qianyang 
	 * ****/
	public static double multivariateTDensity(ArrayRealVector dataPoint, ArrayRealVector meansVector, RealMatrix covarianceMatrix, double degreesOfFreedom){
		LUDecomposition covariance = new LUDecomposition(covarianceMatrix);
		double logprob_left = Gamma.logGamma((degreesOfFreedom + dataPoint.getDimension())/2.0) - 
				(Gamma.logGamma(degreesOfFreedom / 2.0) + 0.5 * Math.log(covariance.getDeterminant()) + 
						dataPoint.getDimension()/2.0 * (Math.log(degreesOfFreedom) + Math.log(Math.PI)));		
		// compute x-u
		ArrayRealVector var = dataPoint.add(meansVector.mapMultiplyToSelf(-1.0));
		// (x-u) to  matrix
		RealMatrix realMatrix = new Array2DRowRealMatrix(var.getDataRef());
		//compute left
		double logprob_right = Math.log(1 + realMatrix.transpose().multiply(new LUDecomposition(covarianceMatrix).getSolver().getInverse())
				.multiply(realMatrix).getData()[0][0]/degreesOfFreedom);
		return Math.exp(logprob_left -(degreesOfFreedom + dataPoint.getDimension())/2.0 * logprob_right);
	}
	/**Arrays  Search
	 * 
	 * @param arr
	 * @param targetValue
	 * @return boolean
	 * For example: arr = new int[] { 3, 5, 7, 11, 13 }
	 * targetValue = 3
	 * the return will be true
	 * ****/
	public static boolean arrSearch(int[] arr, int targetValue) {
		for (Integer s : arr) {
			if (s.equals(targetValue))
				return true;
		}
		return false;
	}
	public static double logOn2Gamma(double value) {
		return com.aliasi.util.Math.log2Gamma(value);
	}
	public static double logOn2(double value) {
		return com.aliasi.util.Math.log2(value);
	}
	public static double log2betaf(double a,double b){
		double beta = logOn2Gamma(a)+ logOn2Gamma(b)-logOn2Gamma(a+b);
		return beta;
	}
	public static double exponential2(double a){
		return java.lang.Math.pow(2.0, a);
	}
}
