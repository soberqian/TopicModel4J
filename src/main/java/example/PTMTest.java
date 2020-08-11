package example;

import com.topic.model.PseudoDTM;

public class PTMTest {

	public static void main(String[] args) {
		PseudoDTM ptm = new PseudoDTM("data/amReviews", "gbk", 1000, 30, 0.1, 0.1,
				0.01, 1000, 20, "data/ptmoutput/");
		ptm.MCMCSampling();
	}

}
