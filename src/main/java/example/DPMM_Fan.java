package example;

import com.topic.model.DPMM;

public class DPMM_Fan {

	public static void main(String[] args) {
		DPMM dmm = new DPMM("data/shortdoc.txt", "gbk", 5, 0.1,
				0.01, 1500, 50, "data/ldaoutput/");
		dmm.MCMCSampling();

	}

}
