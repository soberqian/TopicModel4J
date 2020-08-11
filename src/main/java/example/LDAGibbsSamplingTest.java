package example;

import com.topic.evaluation.EstimationUtil;
import com.topic.model.GibbsSamplingLDA;

public class LDAGibbsSamplingTest {

	public static void main(String[] args) {
		GibbsSamplingLDA lda = new GibbsSamplingLDA("data/rawdataProcessAbstracts", "gbk", 30, 0.1,
				0.01, 500, 5, "data/ldagibbsoutput/");
		lda.MCMCSampling();
		double Ac5 = EstimationUtil.average_coherence(lda.docword, lda.estimatePhi(), 5);
		double Ac10 = EstimationUtil.average_coherence(lda.docword, lda.estimatePhi(), 10);
		double Ac20 = EstimationUtil.average_coherence(lda.docword, lda.estimatePhi(), 20);
		System.out.println("average_coherence_5:\t" + Ac5);
		System.out.println("average_coherence_10:\t" + Ac10);
		System.out.println("average_coherence_20:\t" + Ac20);
	}
}
