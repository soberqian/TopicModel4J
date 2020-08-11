package example;

import com.topic.model.DPMM;

public class DPMMTest {

	public static void main(String[] args) {
		DPMM dpmm = new DPMM("data/amReviews", "gbk", 3, 0.01,
				0.1, 1000, 20, "data/dpmmoutput/");
		dpmm.MCMCSampling();
	}
}
