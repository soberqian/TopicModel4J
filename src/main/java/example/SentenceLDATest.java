package example;

import com.topic.model.SentenceLDA;

public class SentenceLDATest {

	public static void main(String[] args) {
		SentenceLDA sentenceLda = new SentenceLDA("data/amR_process", "gbk", 10, 0.1,
				0.01, 1000, 5, "data/senLDAoutput/");
		sentenceLda.MCMCSampling();
	}
}
