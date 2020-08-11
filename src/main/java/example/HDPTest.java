package example;

import com.topic.model.HDP;

public class HDPTest {

	public static void main(String[] args) {
		HDP hdp = new HDP("data/rawdataProcessAbstracts", "gbk", 3, 0.1, 0.01,
				0.1, 1000, 20, "data/hdpoutput/");
		hdp.MCMCSampling();
	}
}
