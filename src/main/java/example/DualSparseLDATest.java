package example;

import com.topic.model.DualSparseLDA;

public class DualSparseLDATest {

	public static void main(String[] args) {
		DualSparseLDA dualSLDA = new DualSparseLDA("data/amReviewsGift", "gbk", 12, 1.0, 1.0, 1.0, 1.0, 0.1, 1E-12, 0.1, 1E-12, 20, 20, "data/dstmoutput/");
		dualSLDA.CVBInference();
	}
}
