package example;

import com.topic.model.LabeledLDA;

public class LabeledLDATest {

	public static void main(String[] args) {
		LabeledLDA llda = new LabeledLDA("data/programmableweb.txt", "gbk", ",", 0.1,
				0.01, 1000, 5, "data/labeledLDAoutput/");
		llda.MCMCSampling();
	}
}
