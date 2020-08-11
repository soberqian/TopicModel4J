package example;

import com.topic.model.CVBLDA;

public class CVBLDATest {

	public static void main(String[] args) {
		CVBLDA cvblda = new CVBLDA("data/rawdataProcessAbstracts", "gbk", 30, 0.1,
				0.01, 1000, 5, "data/ldacvboutput/");
		cvblda.CVBInference();
	}
}
