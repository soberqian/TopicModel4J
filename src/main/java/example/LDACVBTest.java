package example;

import com.topic.model.CVBLDA;
public class LDACVBTest {
	public static void main(String[] args) {
		CVBLDA lda = new CVBLDA("data/rawdata_process", "gbk", 30, 0.1,
				0.01, 200, 50, "data/ldaoutput/");
		lda.CVBInference();
	}
}
