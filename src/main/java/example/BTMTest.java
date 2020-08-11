package example;

import com.topic.model.BTM;

public class BTMTest {

	public static void main(String[] args) {
		BTM btm = new BTM("data/amReviews", "gbk", 20, 0.1,
				0.01, 1000, 30, 10, "data/btmoutput/");
		btm.MCMCSampling();
	}
}
