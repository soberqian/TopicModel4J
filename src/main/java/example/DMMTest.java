package example;

import com.topic.model.DMM;

public class DMMTest {

	public static void main(String[] args) {
		DMM dmm = new DMM("data/amReviews", "gbk", 20, 0.1,
				0.01, 1000, 15, "data/dmmoutput/");
		dmm.MCMCSampling();
	}
}
