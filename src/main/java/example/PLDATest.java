package example;

import com.topic.model.PLDA;

public class PLDATest {

	public static void main(String[] args) {
//		PLDA plda = new PLDA("data/rawdata_process_author", "gbk", "--", 3, 0.1,
//				0.01, 500, 50, "data/ldaoutput/");
//		plda.MCMCSampling();
		PLDA plda = new PLDA("data/programmableweb.txt", "gbk", ",", 2, 0.1,
				0.01, 1000, 20, "data/pLDAoutput/");
		plda.MCMCSampling();

	}
}
